"""Napari interactive viewer for pipeline stage inspection.

"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import napari; fall back gracefully
try:
    import napari
    _NAPARI_OK = True
except ImportError:
    _NAPARI_OK = False
    logger.warning("napari not installed. Install with: pip install napari PyQt5")


def _graph_to_points(G, attr: str = "pos") -> np.ndarray:
    """Return (N, 3) array of node positions."""
    try:
        import networkx as nx
    except ImportError:
        return np.zeros((0, 3))
    pts = [data[attr] for _, data in G.nodes(data=True) if attr in data]
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


def _graph_to_lines(G, shape: tuple) -> List[np.ndarray]:
    """Return list of (N, 3) arrays - one per edge, using path attribute."""
    lines = []
    for u, v, data in G.edges(data=True):
        path = data.get("path")
        if path and len(path) >= 2:
            lines.append(np.array(path, dtype=np.float32))
        else:
            u_pos = G.nodes[u].get("pos")
            v_pos = G.nodes[v].get("pos")
            if u_pos is not None and v_pos is not None:
                lines.append(np.array([u_pos, v_pos], dtype=np.float32))
    return lines


def _strahler_colormap(orders: dict) -> np.ndarray:
    """Map Strahler orders -> RGBA colours using matplotlib jet.

    Returns:
        (N, 4) float32 array in [0, 1].
    """
    import matplotlib.cm as cm
    vals = np.array(list(orders.values()), dtype=np.float32)
    if vals.max() > vals.min():
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        norm_vals = np.zeros_like(vals)
    cmap = cm.get_cmap("RdYlBu_r")
    colours = cmap(norm_vals).astype(np.float32)
    return colours


class PipelineViewer:
    """Interactive napari viewer wrapper for pipeline stage inspection.

    Args:
        spacing:  Voxel spacing (z, y, x) in mm.
        three_d:  If True, enable 3D rendering mode.
    """

    def __init__(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        three_d: bool = False,
    ) -> None:
        self.spacing = tuple(spacing)
        self.three_d = three_d
        self._viewer: Optional[Any] = None

    def _get_viewer(self):
        """Lazily create or return the napari viewer."""
        if not _NAPARI_OK:
            raise ImportError("napari is not installed")
        if self._viewer is None:
            self._viewer = napari.Viewer(
                title="Pulmonary Vascular Tree Pipeline",
                ndisplay=3 if self.three_d else 2,
            )
        return self._viewer

    def show(self) -> None:
        """Start the napari event loop (blocks until window is closed)."""
        if not _NAPARI_OK:
            logger.warning("Cannot show napari: not installed.")
            return
        try:
            v = self._get_viewer()
            napari.run()
            self._viewer = None  # reset for next stage
        except RuntimeError as exc:
            if "X server" in str(exc) or "display" in str(exc).lower():
                logger.warning("Headless server - napari cannot open a window.")
            else:
                logger.warning("napari error: %s", exc)

    def stage0_raw_ct(self, hu_iso: np.ndarray, spacing: np.ndarray) -> None:
        """Show raw CT volume."""
        try:
            v = self._get_viewer()
            v.add_image(
                hu_iso, name="CT (HU)",
                colormap="gray",
                contrast_limits=(-1000, 400),
                scale=tuple(spacing),
                rendering="mip",
            )
        except Exception as exc:
            logger.warning("napari stage0 error: %s", exc)

    def stage1_segmentation(
        self,
        hu_iso: np.ndarray,
        stages_dict: Dict[str, np.ndarray],
        spacing: np.ndarray,
    ) -> None:
        """Show CT with segmentation stage overlays.

        stages_dict keys: body, lungs0_thresh, lungs1_aircut, lungs2_post, final
        """
        _STAGE_COLORS = {
            "body":          (1.0, 1.0, 0.0, 1.0),   # yellow
            "lungs0_thresh": (0.0, 1.0, 1.0, 1.0),   # cyan
            "lungs1_aircut": (1.0, 0.0, 1.0, 1.0),   # magenta
            "lungs2_post":   (0.0, 1.0, 0.0, 1.0),   # green
            "final":         (1.0, 0.0, 0.0, 1.0),   # red
            "refined":       (1.0, 0.0, 0.0, 1.0),   # red
        }
        try:
            v = self._get_viewer()
            v.add_image(
                hu_iso, name="CT (HU)",
                colormap="gray",
                contrast_limits=(-1000, 400),
                scale=tuple(spacing),
            )
            for stage_name, mask in stages_dict.items():
                color = _STAGE_COLORS.get(stage_name, (1.0, 1.0, 1.0, 1.0))
                lyr = v.add_labels(
                    mask.astype(np.uint8),
                    name=f"[S1] {stage_name}",
                    opacity=0.35,
                    scale=tuple(spacing),
                )
                try:
                    lyr.color = {1: color}
                except Exception:
                    try:
                        lyr.color_dict = {1: color}
                    except Exception:
                        pass
                lyr.visible = (stage_name in ("final", "refined"))
        except Exception as exc:
            logger.warning("napari stage1 error: %s", exc)

    def stage2_diffusion(
        self,
        hu_iso: np.ndarray,
        smoothed_hu: np.ndarray,
        lung_mask: np.ndarray,
        spacing: np.ndarray,
    ) -> None:
        """Show original CT vs diffusion-smoothed CT with lung mask."""
        try:
            v = self._get_viewer()
            p_lo = float(np.percentile(hu_iso, 0.5))
            p_hi = float(np.percentile(hu_iso, 99.5))
            v.add_image(hu_iso,    name="CT oryginał", colormap="gray",
                        contrast_limits=(p_lo, p_hi), scale=tuple(spacing))
            v.add_image(smoothed_hu, name="Po dyfuzji", colormap="gray",
                        contrast_limits=(p_lo, p_hi), scale=tuple(spacing))
            lyr = v.add_labels(lung_mask.astype(np.uint8), name="Maska płuc",
                               opacity=0.2, scale=tuple(spacing))
            lyr.visible = False
        except Exception as exc:
            logger.warning("napari stage2 error: %s", exc)

    def stage3_vesselness(
        self,
        hu_iso: np.ndarray,
        vesselness: np.ndarray,
        lung_mask: np.ndarray,
        spacing: np.ndarray,
    ) -> None:
        """Show CT + continuous vesselness map."""
        try:
            v = self._get_viewer()
            v.add_image(hu_iso, name="CT (HU)", colormap="gray",
                        contrast_limits=(-1000, 400), scale=tuple(spacing))
            v.add_image(vesselness, name="Vesselness [0,1]",
                        colormap="magma",
                        contrast_limits=(0.0, 1.0),
                        blending="additive",
                        scale=tuple(spacing))
            lm = v.add_labels(lung_mask.astype(np.uint8), name="Maska płuc",
                              opacity=0.1, scale=tuple(spacing))
            lm.visible = False
        except Exception as exc:
            logger.warning("napari stage3 error: %s", exc)

    def stage4_skeleton(
        self,
        hu_iso: np.ndarray,
        skeleton_left: np.ndarray,
        skeleton_right: np.ndarray,
        edt_left: np.ndarray,
        edt_right: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
        spacing: np.ndarray,
    ) -> None:
        """Show CT with skeleton overlays, EDT volumes and growth masks."""
        try:
            v = self._get_viewer()
            sp = tuple(spacing)
            v.add_image(hu_iso, name="CT (HU)", colormap="gray",
                        contrast_limits=(-1000, 400), scale=sp)

            # Skeleton as coloured labels (left=1, right=2)
            skel_combined = np.zeros_like(skeleton_left, dtype=np.uint8)
            skel_combined[skeleton_left > 0] = 1
            skel_combined[skeleton_right > 0] = 2
            v.add_labels(skel_combined, name="Szkielet L/R", scale=sp)

            # EDT (hidden by default)
            el = v.add_image(edt_left,  name="EDT lewe płuco",
                             colormap="viridis", blending="additive", scale=sp)
            er = v.add_image(edt_right, name="EDT prawe płuco",
                             colormap="viridis", blending="additive", scale=sp)
            el.visible = False
            er.visible = False

            # Growth masks (hidden by default)
            lm_l = v.add_labels(left_mask.astype(np.uint8),  name="Maska lewe",
                                opacity=0.1, scale=sp)
            lm_r = v.add_labels(right_mask.astype(np.uint8), name="Maska prawe",
                                opacity=0.1, scale=sp)
            lm_l.visible = False
            lm_r.visible = False

        except Exception as exc:
            logger.warning("napari stage4 error: %s", exc)

    def stage5_graph(
        self,
        hu_iso: np.ndarray,
        graph_left,
        graph_right,
        spacing: np.ndarray,
    ) -> None:
        """Show CT with graph edges and nodes (endpoints/junctions)."""
        try:
            v = self._get_viewer()
            sp = tuple(spacing)
            v.add_image(hu_iso, name="CT (HU)", colormap="gray", scale=sp)

            for G, side, edge_colour in (
                (graph_left,  "lewe",  "cyan"),
                (graph_right, "prawe", "orange"),
            ):
                lines = _graph_to_lines(G, hu_iso.shape)
                if lines:
                    v.add_shapes(
                        lines, shape_type="path",
                        edge_color=edge_colour, edge_width=1.5,
                        name=f"Naczynia ({side})", scale=sp,
                    )

                endpoints  = []
                junctions  = []
                for node, data in G.nodes(data=True):
                    pos = data.get("pos")
                    if pos is None:
                        continue
                    deg = G.degree(node)
                    if deg == 1:
                        endpoints.append(np.array(pos, dtype=np.float32))
                    elif deg >= 3:
                        junctions.append(np.array(pos, dtype=np.float32))

                if endpoints:
                    v.add_points(
                        np.stack(endpoints), name=f"Końcówki ({side})",
                        face_color="lime", size=3, scale=sp, n_dimensional=True,
                    )
                if junctions:
                    v.add_points(
                        np.stack(junctions), name=f"Rozgałęzienia ({side})",
                        face_color="red", size=4, scale=sp, n_dimensional=True,
                    )
        except Exception as exc:
            logger.warning("napari stage5 error: %s", exc)

    def stage6_morphometry(
        self,
        hu_iso: np.ndarray,
        graph_left,
        graph_right,
        strahler_left: dict,
        strahler_right: dict,
        spacing: np.ndarray,
    ) -> None:
        """Show CT with Strahler-coloured nodes and radius-coloured edges."""
        try:
            import matplotlib.cm as cm
            v = self._get_viewer()
            sp = tuple(spacing)
            v.add_image(hu_iso, name="CT (HU)", colormap="gray", scale=sp)

            for G, strahler, side in (
                (graph_left,  strahler_left,  "lewe"),
                (graph_right, strahler_right, "prawe"),
            ):
                # Nodes coloured by Strahler order
                positions, colours, sizes = [], [], []
                for node, data in G.nodes(data=True):
                    pos = data.get("pos")
                    if pos is None:
                        continue
                    order = strahler.get(node, 1)
                    positions.append(np.array(pos, dtype=np.float32))
                    sizes.append(float(1 + order))

                if positions:
                    strahler_sub = {n: strahler.get(n, 1)
                                    for n, d in G.nodes(data=True) if d.get("pos") is not None}
                    colours_arr = _strahler_colormap(strahler_sub)
                    v.add_points(
                        np.stack(positions),
                        name=f"Strahler ({side})",
                        face_color=colours_arr,
                        size=np.array(sizes),
                        scale=sp, n_dimensional=True,
                    )

                # Edges coloured by mean radius - single layer per side
                all_radii = [d.get("mean_radius", 0.0) for _, _, d in G.edges(data=True)]
                max_r = max(all_radii) if all_radii else 1.0
                cmap = cm.get_cmap("plasma")

                edge_paths = []
                edge_colours = []
                for u_node, v_node, data in G.edges(data=True):
                    path = data.get("path")
                    if not path or len(path) < 2:
                        continue
                    r_norm = data.get("mean_radius", 0.0) / (max_r + 1e-9)
                    edge_paths.append(np.array(path, dtype=np.float32))
                    edge_colours.append(cmap(r_norm))

                if edge_paths:
                    v.add_shapes(
                        edge_paths, shape_type="path",
                        edge_color=np.array(edge_colours, dtype=np.float32),
                        edge_width=1.5,
                        name=f"Naczynia radius ({side})",
                        scale=sp,
                    )
        except Exception as exc:
            logger.warning("napari stage6 error: %s", exc)
