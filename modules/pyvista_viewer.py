"""PyVista 3D viewer

Two views:
    1. Center points over vesselness point cloud.
    2. Skeleton with radius (vessel thickness).

"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
    _PV_OK = True
except ImportError:
    _PV_OK = False
    logger.warning("pyvista not installed - 3D verification viewer unavailable")


# Visual config
_SKELETON_LEFT_COLOR = "#00ff88"
_SKELETON_RIGHT_COLOR = "#ffaa00"
_SKELETON_LINE_WIDTH = 2.0

_BIF_COLOR = "#ff3333"
_BIF_SIZE = 5.0
_EP_COLOR = "#66ccff"
_EP_SIZE = 3.0

_VESS_CMAP = "hot"
_VESS_OPACITY_RANGE = (0.05, 0.6)
_VESS_POINT_SIZE = 3.0
_VESS_MAX_POINTS = 800_000
_VESS_THRESHOLD = 0.01

_BG_TOP = "#0a0a1e"
_BG_BOT = "#000000"

_TUBE_MIN_RADIUS = 0.3
_TUBE_DEFAULT_RADIUS = 0.5
_TUBE_N_SIDES = 8


# Helpers

def _graph_to_pv_lines(G, spacing: np.ndarray) -> Optional["pv.PolyData"]:
    """Convert NetworkX graph edges -> PyVista PolyData with line cells."""
    all_pts = []
    all_lines = []
    offset = 0

    sp = np.asarray(spacing, dtype=np.float64)

    for _u, _v, data in G.edges(data=True):
        path_mm = data.get("path_mm")
        if path_mm and len(path_mm) >= 2:
            pts = np.array(path_mm, dtype=np.float64)
        else:
            path_vox = data.get("path")
            if path_vox and len(path_vox) >= 2:
                pts = np.array(path_vox, dtype=np.float64) * sp
            else:
                u_mm = G.nodes[_u].get("pos_mm")
                v_mm = G.nodes[_v].get("pos_mm")
                if u_mm is None or v_mm is None:
                    continue
                pts = np.array([u_mm, v_mm], dtype=np.float64)

        n = len(pts)
        all_pts.append(pts)
        for i in range(n - 1):
            all_lines.extend([2, offset + i, offset + i + 1])
        offset += n

    if not all_pts:
        return None
    points = np.vstack(all_pts)
    lines = np.array(all_lines, dtype=np.int64)
    return pv.PolyData(points, lines=lines)


def _graph_to_pv_lines_with_radius(
    G, spacing: np.ndarray,
) -> Optional["pv.PolyData"]:
    """Convert graph edges -> single PolyData with per-point 'radius' scalars.

    One tube() call on the result gives varying-radius tubes.
    """
    sp = np.asarray(spacing, dtype=np.float64)
    all_pts: list = []
    all_radii: list = []
    all_lines: list = []
    offset = 0

    for _u, _v, data in G.edges(data=True):
        path_mm = data.get("path_mm")
        if path_mm and len(path_mm) >= 2:
            pts = np.array(path_mm, dtype=np.float64)
        else:
            path_vox = data.get("path")
            if path_vox and len(path_vox) >= 2:
                pts = np.array(path_vox, dtype=np.float64) * sp
            else:
                u_mm = G.nodes[_u].get("pos_mm")
                v_mm = G.nodes[_v].get("pos_mm")
                if u_mm is None or v_mm is None:
                    continue
                pts = np.array([u_mm, v_mm], dtype=np.float64)

        if len(pts) < 2:
            continue

        radius = max(float(data.get("mean_radius", _TUBE_DEFAULT_RADIUS)),
                     _TUBE_MIN_RADIUS)

        n = len(pts)
        all_pts.append(pts)
        all_radii.append(np.full(n, radius, dtype=np.float32))
        # polyline cell: [n_pts, idx0, idx1, ..., idx_n-1]
        all_lines.append(np.array([n] + list(range(offset, offset + n)),
                                  dtype=np.int64))
        offset += n

    if not all_pts:
        return None

    points = np.vstack(all_pts)
    radii = np.concatenate(all_radii)
    lines = np.concatenate(all_lines)

    poly = pv.PolyData(points, lines=lines)
    poly["radius"] = radii
    return poly


def _graph_special_nodes(G, spacing: np.ndarray):
    """Return (bifurcation_points, endpoint_points) as numpy arrays in mm."""
    sp = np.asarray(spacing, dtype=np.float64)
    bifs, eps = [], []
    for _n, data in G.nodes(data=True):
        pos_mm = data.get("pos_mm")
        if pos_mm is None:
            pos = data.get("pos")
            if pos is None:
                continue
            pos_mm = tuple(float(p * s) for p, s in zip(pos, sp))

        deg = G.degree(_n)
        if deg >= 3:
            bifs.append(pos_mm)
        elif deg == 1:
            eps.append(pos_mm)

    bif_arr = np.array(bifs, dtype=np.float64) if bifs else None
    ep_arr = np.array(eps, dtype=np.float64) if eps else None
    return bif_arr, ep_arr


def _vesselness_to_cloud(
    vesselness: np.ndarray,
    spacing: np.ndarray,
    threshold: float = _VESS_THRESHOLD,
    max_points: int = _VESS_MAX_POINTS,
) -> Optional["pv.PolyData"]:
    """Subsample vesselness > threshold -> PyVista point cloud with scalars."""
    sp = np.asarray(spacing, dtype=np.float64)
    mask = vesselness > threshold
    coords = np.argwhere(mask)
    n = len(coords)
    if n == 0:
        return None

    vals = vesselness[mask]

    if n > max_points:
        probs = vals ** 2
        probs /= probs.sum()
        idx = np.random.choice(n, max_points, replace=False, p=probs)
        coords = coords[idx]
        vals = vals[idx]

    pts = coords.astype(np.float64) * sp
    cloud = pv.PolyData(pts)
    cloud["vesselness"] = vals.astype(np.float32)
    return cloud


# API

def show_graph_vs_vesselness(
    graph_left,
    graph_right,
    vesselness: np.ndarray,
    spacing: Tuple[float, float, float],
    show_bifurcations: bool = True,
    show_endpoints: bool = False,
    window_size: Tuple[int, int] = (1920, 1080),
) -> None:
    """Open two PyVista windows:
    1. Skeleton over vesselness point cloud.
    2. Skeleton with vessel radius.
    """
    if not _PV_OK:
        logger.warning("PyVista unavailable - skipping 3D verification viewer.")
        return

    sp = np.asarray(spacing, dtype=np.float64)

    logger.info("Building PyVista verification scene...")

    pv.set_plot_theme("dark")

    # View 1
    pl = pv.Plotter(window_size=window_size, title="Szkielet - linie")
    pl.set_background(_BG_BOT, top=_BG_TOP)

    cloud = _vesselness_to_cloud(vesselness, sp)
    if cloud is not None:
        logger.info("  Vesselness cloud: %d points", cloud.n_points)
        vmax = float(np.percentile(vesselness[vesselness > 0], 90))
        pl.add_mesh(
            cloud,
            scalars="vesselness",
            cmap=_VESS_CMAP,
            clim=[_VESS_THRESHOLD, max(vmax, 0.1)],
            opacity=_VESS_OPACITY_RANGE,
            point_size=_VESS_POINT_SIZE,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={"title": "Vesselness", "n_labels": 3},
        )

    for G, color in (
        (graph_left,  _SKELETON_LEFT_COLOR),
        (graph_right, _SKELETON_RIGHT_COLOR),
    ):
        mesh = _graph_to_pv_lines(G, sp)
        if mesh is not None:
            logger.info("  Lines: %d points, %d segments", mesh.n_points, mesh.n_cells)
            pl.add_mesh(
                mesh,
                color=color,
                line_width=_SKELETON_LINE_WIDTH,
                render_lines_as_tubes=False,
                lighting=False,
            )

        bif_pts, ep_pts = _graph_special_nodes(G, sp)
        if show_bifurcations and bif_pts is not None:
            pl.add_mesh(
                pv.PolyData(bif_pts),
                color=_BIF_COLOR,
                point_size=_BIF_SIZE,
                render_points_as_spheres=True,
            )
        if show_endpoints and ep_pts is not None:
            pl.add_mesh(
                pv.PolyData(ep_pts),
                color=_EP_COLOR,
                point_size=_EP_SIZE,
                render_points_as_spheres=True,
            )

    pl.camera_position = "xz"
    pl.camera.azimuth = 30
    pl.camera.elevation = 20
    pl.reset_camera()

    # View 2
    pl2 = pv.Plotter(window_size=window_size, title="Szkielet - grubość naczyń")
    pl2.set_background(_BG_BOT, top=_BG_TOP)

    has_scalar_bar = False
    for G, label in (
        (graph_left,  "left"),
        (graph_right, "right"),
    ):
        logger.info("  Building tubes [%s]...", label)
        lines_with_r = _graph_to_pv_lines_with_radius(G, sp)
        if lines_with_r is None:
            continue

        logger.info(
            "  [%s] Lines: %d pts, radius range: %.2f – %.2f mm",
            label, lines_with_r.n_points,
            float(lines_with_r["radius"].min()),
            float(lines_with_r["radius"].max()),
        )

        tube_mesh = lines_with_r.tube(
            scalars="radius",
            n_sides=_TUBE_N_SIDES,
        )
        logger.info("  [%s] Tubes: %d pts", label, tube_mesh.n_points)

        show_bar = not has_scalar_bar
        pl2.add_mesh(
            tube_mesh,
            scalars="radius",
            cmap="plasma",
            show_scalar_bar=show_bar,
            scalar_bar_args={"title": "Promień (mm)", "n_labels": 5},
            lighting=True,
            smooth_shading=True,
        )
        has_scalar_bar = True

    pl2.camera_position = "xz"
    pl2.camera.azimuth = 30
    pl2.camera.elevation = 20
    pl2.reset_camera()

    logger.info("Opening PyVista viewers...")
    pl.show(interactive_update=True)
    pl2.show()
