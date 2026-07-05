"""Stage 06 - Morphometric analysis of vascular trees.

Pipeline:
    A) Strahler Ordering & Generation Numbers
    B) FWHM Radius Measurement (EDT-based)
    C) Structural Validation:
       1. Fractal Dimension (box-counting 3D)
       2. Diameter Index (DI)
       3. Murray's Law (nonlinear optimisation)
       4. Horton Ratios (R_B, R_L, R_D)
       5. Tortuosity
       6. Asymmetry Ratio (Zamir 1999)

Output:
    outputs/morphometry_LEFT.csv
    outputs/morphometry_RIGHT.csv
    outputs/morphometry_summary.json

Cache keys:
    metrics_left  : dict
    metrics_right : dict
"""
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def run(
    cache: CacheManager,
    graph_left: nx.Graph,
    graph_right: nx.Graph,
    edt_left: np.ndarray,
    edt_right: np.ndarray,
    spacing: np.ndarray,
    output_dir: str = "outputs",
    fractal_scales: list = None,
    **_: object,
) -> dict:
    """Compute morphometric metrics for left and right vascular trees.

    Args:
        cache:         CacheManager instance.
        graph_left:    Acyclic NetworkX Graph for left lung.
        graph_right:   Acyclic NetworkX Graph for right lung.
        edt_left:      EDT volume for left lung float32.
        edt_right:     EDT volume for right lung float32.
        spacing:       Voxel spacing [z,y,x] mm.
        output_dir:    Directory for CSV/JSON output.
        fractal_scales: Box sizes for fractal dimension.

    Returns:
        dict with metrics_left and metrics_right.
    """
    if fractal_scales is None:
        fractal_scales = [2, 4, 8, 16, 32, 64]

    params = {
        "fractal_scales": fractal_scales,
        "nodes_left":     graph_left.number_of_nodes(),
        "nodes_right":    graph_right.number_of_nodes(),
    }
    cached = cache.get("stage_06_morphometric_analysis", params)
    if cached is not None:
        return cached

    t0 = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, dict] = {}

    for side, G, edt in (
        ("LEFT",  graph_left,  edt_left),
        ("RIGHT", graph_right, edt_right),
    ):
        ts = time.time()
        logger.info("Morphometry [%s] …", side)

        root = _find_root_node(G, edt)
        strahler = _compute_strahler(G, root)
        depth    = _compute_generation(G, root)

        for node in G.nodes:
            G.nodes[node]["strahler"]   = strahler.get(node, 1)
            G.nodes[node]["generation"] = depth.get(node, 0)

        _update_edge_radii(G, edt)

        # Fractal dim on 1-voxel skeleton
        volume_shape = edt.shape
        skeleton_arr = _paths_to_skeleton(G, volume_shape)
        fd_skel_res = _fractal_dimension(skeleton_arr, fractal_scales)
        fd = fd_skel_res["d_f"]

        # Fractal dim on vessel volume (skeleton inflated by EDT radius)
        logger.info("[%s] Building vessel volume for volumetric FD...", side)
        vessel_vol = _paths_to_vessel_volume(G, edt)
        fd_vol_res = _fractal_dimension(vessel_vol, fractal_scales)
        fd_vol = fd_vol_res["d_f"]
        logger.info("[%s] FD_skel=%.3f  FD_vol=%.3f  (vol voxels=%d)",
                    side, fd, fd_vol, int(vessel_vol.sum()))

        di   = _diameter_index(G, strahler)
        murray_alpha = _murray_law(G)
        horton = _horton_ratios(G, strahler)
        tort   = _tortuosity_stats(G, spacing)

        bif_counts = _bifurcation_counts(G)
        asym       = _asymmetry_ratio(G)

        n_components = nx.number_connected_components(G)
        n_cycles = (
            G.number_of_edges()
            - G.number_of_nodes()
            + n_components
        )

        metrics = {
            "fractal_dimension":   fd,
            "fractal_dimension_err": fd_skel_res["d_f_err"],
            "fractal_dimension_volumetric": fd_vol,
            "fractal_dimension_volumetric_err": fd_vol_res["d_f_err"],
            "fractal_dimension_volumetric_intercept": fd_vol_res["intercept"],
            "fractal_dimension_volumetric_scales": fd_vol_res["scales"],
            "fractal_dimension_volumetric_counts": fd_vol_res["counts"],
            "diameter_index":      di,
            "murray_alpha":        murray_alpha,
            "horton_R_B":          horton.get("R_B"),
            "horton_R_L":          horton.get("R_L"),
            "horton_R_D":          horton.get("R_D"),
            "horton_R_B_sd":       horton.get("R_B_sd"),
            "horton_R_L_sd":       horton.get("R_L_sd"),
            "horton_R_D_sd":       horton.get("R_D_sd"),
            "horton_n_R_B":        horton.get("n_R_B"),
            "horton_n_R_L":        horton.get("n_R_L"),
            "horton_n_R_D":        horton.get("n_R_D"),
            "horton_R_B_vals":     horton.get("R_B_vals", []),
            "horton_R_L_vals":     horton.get("R_L_vals", []),
            "horton_R_D_vals":     horton.get("R_D_vals", []),
            "tortuosity_mean":     tort["mean"],
            "tortuosity_std":      tort["std"],
            "tortuosity_max":      tort["max"],
            "n_nodes":             G.number_of_nodes(),
            "n_edges":             G.number_of_edges(),
            "max_strahler":        max(strahler.values()) if strahler else 0,
            "n_bifurcations_3way": bif_counts.get(3, 0),
            "n_bifurcations_4way": bif_counts.get(4, 0),
            "n_bifurcations_5plus": sum(v for k, v in bif_counts.items() if k >= 5),
            "n_bifurcations_total": sum(bif_counts.values()),
            "asymmetry_ratio_mean":   asym["mean"],
            "asymmetry_ratio_std":    asym["std"],
            "asymmetry_ratio_median": asym["median"],
            "n_connected_components": n_components,
            "n_cycles":            n_cycles,
        }
        all_metrics[side] = metrics
        logger.info(
            "[%s] FD=%.3f FD_vol=%.3f DI=%.3f Murray_α=%.3f δ=%.3f bif3=%d bif4=%d comp=%d cyc=%d  (%.1fs)",
            side, fd, fd_vol, di, murray_alpha if murray_alpha else float("nan"),
            asym["mean"],
            bif_counts.get(3, 0), bif_counts.get(4, 0),
            n_components, n_cycles, time.time() - ts,
        )

        rows = []
        for u, v, data in G.edges(data=True):
            rows.append({
                "node_u":            str(u),
                "node_v":            str(v),
                "length_mm":         data.get("length", data.get("length_mm", 0.0)),
                "mean_radius_mm":    data.get("mean_radius", 0.0),
                "radius_min_mm":     data.get("radius_min_mm", 0.0),
                "radius_max_mm":     data.get("radius_max_mm", 0.0),
                "diameter_mean_mm":  data.get("diameter_mean_mm", 0.0),
                "strahler_u":        strahler.get(u, 1),
                "strahler_v":        strahler.get(v, 1),
                "tortuosity":        data.get("tortuosity", float("nan")),
            })
        if rows:
            df = pd.DataFrame(rows)
            csv_path = out_dir / f"morphometry_{side}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved %s (%d rows)", csv_path, len(df))

        _save_graph(G, out_dir / f"graph_{side}.graphml", side)

    summary_path = out_dir / "morphometry_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Saved %s", summary_path)

    result = {
        "metrics_left":  all_metrics.get("LEFT",  {}),
        "metrics_right": all_metrics.get("RIGHT", {}),
    }
    cache.put(
        "stage_06_morphometric_analysis",
        params,
        result,
        numpy_keys=[],   # dicts -> pickle
    )
    logger.info("Stage 06 done  total=%.1fs", time.time() - t0)
    return result


def _save_graph(G: nx.Graph, path: Path, side: str) -> None:
    """Save graph as GraphML with all node/edge attributes.

    """
    G_export = G.copy()

    for node, data in G_export.nodes(data=True):
        pos = data.get("pos")
        if pos is not None:
            data["pos_z"] = float(pos[0])
            data["pos_y"] = float(pos[1])
            data["pos_x"] = float(pos[2])
            del data["pos"]
        pos_mm = data.get("pos_mm")
        if pos_mm is not None:
            data["pos_mm_z"] = float(pos_mm[0])
            data["pos_mm_y"] = float(pos_mm[1])
            data["pos_mm_x"] = float(pos_mm[2])
            del data["pos_mm"]
        for k, v in list(data.items()):
            if isinstance(v, (list, tuple, np.ndarray)):
                del data[k]

    for u, v, data in G_export.edges(data=True):
        # path arrays are too large for GraphML; kept in per-edge CSV
        for key in ("path", "path_mm"):
            if key in data:
                del data[key]
        for k, val in list(data.items()):
            if isinstance(val, (list, tuple, np.ndarray)):
                del data[k]

    nx.write_graphml(G_export, str(path))
    logger.info("Saved GraphML: %s (%d nodes, %d edges)", path,
                G_export.number_of_nodes(), G_export.number_of_edges())


def _find_root_node(G: nx.Graph, edt: np.ndarray):
    """Return node with maximum EDT radius (thickest vessel = root)."""
    best_node = None
    best_r = -1.0
    for node, data in G.nodes(data=True):
        pos = data.get("pos")
        if pos is None:
            continue
        r = float(edt[pos])
        if r > best_r:
            best_r = r
            best_node = node
    return best_node


def _compute_strahler(G: nx.Graph, root) -> Dict:
    """Compute Strahler order for every node via post-order BFS.

    Rules:
        Leaf -> 1
        All children have order n and ≥2 exist -> parent = n+1
        Children have different orders -> max(orders)
    """
    if root is None or root not in G:
        return {}

    tree = nx.bfs_tree(G, root)
    order: Dict = {}

    # post-order via reversed BFS layers
    bfs_layers = list(nx.bfs_layers(tree, root))
    for layer in reversed(bfs_layers):
        for node in layer:
            children = list(tree.successors(node))
            if not children:
                order[node] = 1
            else:
                child_orders = [order.get(c, 1) for c in children]
                max_o = max(child_orders)
                if child_orders.count(max_o) >= 2:
                    order[node] = max_o + 1
                else:
                    order[node] = max_o
    return order


def _compute_generation(G: nx.Graph, root) -> Dict:
    if root is None or root not in G:
        return {}
    tree = nx.bfs_tree(G, root)
    return nx.single_source_shortest_path_length(tree, root)


def _update_edge_radii(G: nx.Graph, edt: np.ndarray) -> None:
    """Update edge radius attributes from EDT values along path.

    Sets: mean_radius, radius_min_mm, radius_max_mm, diameter_mean_mm,
          and node radius from pos.
    """
    for u, v, data in G.edges(data=True):
        path = data.get("path")
        if path:
            radii = [float(edt[p]) for p in path]
            data["mean_radius"]     = float(np.mean(radii))
            data["radius_min_mm"]   = float(np.min(radii))
            data["radius_max_mm"]   = float(np.max(radii))
            data["diameter_mean_mm"] = 2.0 * float(np.mean(radii))
        if path and len(path) >= 2:  # tortuosity per edge
            p_start = np.array(path[0],  dtype=np.float64)
            p_end   = np.array(path[-1], dtype=np.float64)
            straight = float(np.linalg.norm(p_end - p_start))
            length   = data.get("length", data.get("length_mm", 0.0))
            data["tortuosity"] = (length / straight) if straight > 0 else 1.0
    for node, ndata in G.nodes(data=True):
        pos = ndata.get("pos")
        if pos is not None:
            ndata["radius"] = float(edt[pos])


def _paths_to_skeleton(G: nx.Graph, shape: tuple) -> np.ndarray:
    """Rasterise edge paths to binary skeleton array for fractal dim."""
    skel = np.zeros(shape, dtype=np.uint8)
    for u, v, data in G.edges(data=True):
        path = data.get("path", [G.nodes[u].get("pos"), G.nodes[v].get("pos")])
        for pos in path:
            if pos is not None:
                z, y, x = pos
                if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                    skel[z, y, x] = 1
    return skel


def _paths_to_vessel_volume(G: nx.Graph, edt: np.ndarray) -> np.ndarray:
    """Rasterise edge paths as filled cylinders using EDT radius at each point.

    Each skeleton point is expanded into a sphere of radius = EDT(point),
    producing a volumetric representation that respects vessel width.
    """
    shape = edt.shape
    Z, Y, X = shape
    vol = np.zeros(shape, dtype=np.uint8)

    for u, v, data in G.edges(data=True):
        path = data.get("path", [G.nodes[u].get("pos"), G.nodes[v].get("pos")])
        for pos in path:
            if pos is None:
                continue
            z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
            if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
                continue
            r = max(float(edt[z, y, x]), 0.5)
            ri = int(np.ceil(r))
            z0, z1 = max(z - ri, 0), min(z + ri + 1, Z)
            y0, y1 = max(y - ri, 0), min(y + ri + 1, Y)
            x0, x1 = max(x - ri, 0), min(x + ri + 1, X)
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            dist_sq = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
            vol[z0:z1, y0:y1, x0:x1] |= (dist_sq <= r * r).astype(np.uint8)

    return vol


def _fractal_dimension(skeleton: np.ndarray, scales: list) -> dict:
    """3D box-counting fractal dimension.

    Returns a dict with:
        d_f:        fractal dimension
        d_f_err:    uncertainty from the linear regression
        intercept:  log N at log eps = 0
        scales:     box sizes eps actually used (those that fit in the volume
                    and produced a non-zero count)
        counts:     non-empty box counts N(eps) at each used scale
    """
    Z, Y, X = skeleton.shape
    scales_used: list[int] = []
    counts: list[int] = []
    log_scales: list[float] = []
    log_counts: list[float] = []
    for s in scales:
        if s >= min(Z, Y, X) // 2:
            continue
        nz = math.ceil(Z / s)
        ny = math.ceil(Y / s)
        nx_ = math.ceil(X / s)
        count = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx_):
                    block = skeleton[
                        iz * s: (iz + 1) * s,
                        iy * s: (iy + 1) * s,
                        ix * s: (ix + 1) * s,
                    ]
                    if block.any():
                        count += 1
        if count > 0:
            scales_used.append(int(s))
            counts.append(int(count))
            log_scales.append(math.log(s))
            log_counts.append(math.log(count))

    result = {
        "d_f": float("nan"),
        "d_f_err": float("nan"),
        "intercept": float("nan"),
        "scales": scales_used,
        "counts": counts,
    }
    if len(log_scales) < 2:
        return result

    if len(log_scales) >= 3:
        coeffs, cov = np.polyfit(log_scales, log_counts, 1, cov=True)
        result["d_f_err"] = float(math.sqrt(cov[0, 0]))
    else:
        coeffs = np.polyfit(log_scales, log_counts, 1)

    result["d_f"] = float(-coeffs[0])
    result["intercept"] = float(coeffs[1])
    return result


def _diameter_index(G: nx.Graph, strahler: dict) -> float:
    """DI = mean_diameter_distal / mean_diameter_proximal.

    Proximal: Strahler >= max_order - 1
    Distal:   Strahler == 1
    """
    if not strahler:
        return float("nan")
    max_order = max(strahler.values())
    proximal_d, distal_d = [], []

    for u, v, data in G.edges(data=True):
        mean_d = data.get("diameter_mean_mm", 2.0 * data.get("mean_radius", 0.0))
        s_u = strahler.get(u, 1)
        s_v = strahler.get(v, 1)
        s_edge = min(s_u, s_v)  # edge order = min of endpoints
        if s_edge == 1:
            distal_d.append(mean_d)
        if s_edge >= max_order - 1:
            proximal_d.append(mean_d)

    if not proximal_d or not distal_d:
        return float("nan")
    return float(np.mean(distal_d) / np.mean(proximal_d))


def _murray_law(G: nx.Graph) -> Optional[float]:
    """Estimate Murray's Law exponent α via nonlinear optimisation.

    For each bifurcation node (degree >= 3) the mean edge radius
    along each incident branch is used instead of the point-wise
    EDT at the neighbour node.  This avoids the inflated radius
    values caused by the widened junction geometry in the EDT.

    The exponent α is found by minimising:
        min_α  Σ_j [ α·log(r_parent_j) − log(Σ_i r_child_ji^α) ]²
    """
    from scipy.optimize import minimize_scalar

    bifurcations: list[tuple[float, list[float]]] = []

    for node in G.nodes:
        neighbours = list(G.neighbors(node))
        if len(neighbours) < 3:
            continue
        # Use mean edge radius for each branch
        edge_radii: list[tuple[float, int]] = []
        for nb in neighbours:
            edata = G.edges[node, nb]
            mr = edata.get("mean_radius", 0.0)
            if mr > 0:
                edge_radii.append((mr, nb))

        if len(edge_radii) < 3:
            continue

        edge_radii.sort(reverse=True)
        r_parent = edge_radii[0][0]
        r_children = [r for r, _ in edge_radii[1:]]


        if r_parent <= 0 or any(r <= 0 for r in r_children):
            continue
        bifurcations.append((r_parent, r_children))

    if len(bifurcations) < 3:
        return None

    def _residual(alpha: float) -> float:
        total = 0.0
        for r_p, r_cs in bifurcations:
            lhs = alpha * math.log(r_p)
            rhs = math.log(sum(r ** alpha for r in r_cs))
            total += (lhs - rhs) ** 2
        return total

    result = minimize_scalar(_residual, bounds=(0.5, 8.0), method="bounded")
    return float(result.x) if result.success else None


def _horton_ratios(G: nx.Graph, strahler: dict) -> dict:
    """Compute Horton branching, length, and diameter ratios.

    A Strahler segment is a maximal connected chain of edges whose
    child-node Strahler order equals a common value n.  In the
    directed BFS tree rooted at the thickest node.
    """
    if not strahler:
        return {}

    root = max(strahler, key=lambda n: strahler[n])
    tree = nx.bfs_tree(G, root)

    # edge order = order of the child node (downstream end)
    edge_order: Dict[tuple, int] = {}
    for parent, child in tree.edges():
        edge_order[(parent, child)] = strahler.get(child, 1)

    children_map: Dict = {}
    for parent, child in tree.edges():
        children_map.setdefault(parent, []).append(child)

    parent_map: Dict = {}
    for parent, child in tree.edges():
        parent_map[child] = parent

    seg_lengths: Dict[int, list] = {}
    seg_diams:   Dict[int, list] = {}
    seg_counts:  Dict[int, int]  = {}

    visited_edges: set = set()

    def _edge_length(u: int, v: int) -> float:
        data = G.edges.get((u, v), G.edges.get((v, u), {}))
        return float(data.get("length_mm", data.get("length", 0.0)))

    def _edge_diam(u: int, v: int) -> float:
        data = G.edges.get((u, v), G.edges.get((v, u), {}))
        return float(data.get("diameter_mean_mm",
                              2.0 * data.get("mean_radius", 0.0)))

    def _trace_segment(start_node: int, order: int) -> None:
        """Walk downstream from start_node collecting edges of `order`."""
        total_len = 0.0
        weighted_diam = 0.0
        stack = [start_node]

        while stack:
            node = stack.pop()
            for child in children_map.get(node, []):
                ek = (node, child)
                if ek in visited_edges:
                    continue
                if edge_order.get(ek, -1) != order:
                    continue
                visited_edges.add(ek)
                el = _edge_length(node, child)
                ed = _edge_diam(node, child)
                total_len += el
                weighted_diam += ed * el
                stack.append(child)

        if total_len > 0:
            seg_lengths.setdefault(order, []).append(total_len)
            seg_diams.setdefault(order, []).append(weighted_diam / total_len)
            seg_counts[order] = seg_counts.get(order, 0) + 1

    for node in tree.nodes():
        par = parent_map.get(node)
        for child in children_map.get(node, []):
            ek = (node, child)
            if ek in visited_edges:
                continue
            child_order = edge_order.get(ek, 1)
            if par is None:
                is_start = True
            else:
                incoming_order = edge_order.get((par, node), -1)
                is_start = (incoming_order != child_order)
            if is_start:
                _trace_segment(node, child_order)

    max_order = max(strahler.values())

    rb_vals, rl_vals, rd_vals = [], [], []
    for o in range(1, max_order):
        n_o  = seg_counts.get(o,     0)
        n_o1 = seg_counts.get(o + 1, 0)
        l_o  = float(np.mean(seg_lengths[o]))     if seg_lengths.get(o)     else None
        l_o1 = float(np.mean(seg_lengths[o + 1])) if seg_lengths.get(o + 1) else None
        d_o  = float(np.mean(seg_diams[o]))        if seg_diams.get(o)       else None
        d_o1 = float(np.mean(seg_diams[o + 1]))    if seg_diams.get(o + 1)   else None

        if n_o > 0 and n_o1 > 0:
            rb_vals.append(n_o / n_o1)
        if l_o and l_o1 and l_o > 0:
            rl_vals.append(l_o1 / l_o)
        if d_o and d_o1 and d_o > 0:
            rd_vals.append(d_o1 / d_o)

    def _mean(v: list) -> float:
        return float(np.mean(v)) if v else float("nan")

    def _sd(v: list) -> float:
        return float(np.std(v, ddof=1)) if len(v) >= 2 else float("nan")

    # raw per-order lists allow pooling left+right before computing final mean±SD
    return {
        "R_B":      _mean(rb_vals),
        "R_L":      _mean(rl_vals),
        "R_D":      _mean(rd_vals),
        "R_B_sd":   _sd(rb_vals),
        "R_L_sd":   _sd(rl_vals),
        "R_D_sd":   _sd(rd_vals),
        "n_R_B":    len(rb_vals),
        "n_R_L":    len(rl_vals),
        "n_R_D":    len(rd_vals),
        "R_B_vals": [float(x) for x in rb_vals],
        "R_L_vals": [float(x) for x in rl_vals],
        "R_D_vals": [float(x) for x in rd_vals],
    }


def _tortuosity_stats(G: nx.Graph, spacing: np.ndarray) -> dict:
    values = []
    sp = spacing.astype(np.float64)
    for u, v, data in G.edges(data=True):
        path = data.get("path")
        if not path or len(path) < 2:
            continue
        p_start = np.array(path[0],  dtype=np.float64) * sp
        p_end   = np.array(path[-1], dtype=np.float64) * sp
        straight = float(np.linalg.norm(p_end - p_start))
        length   = float(data.get("length", data.get("length_mm", 0.0)))
        if straight > 0:
            values.append(length / straight)

    if not values:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
        "max":  float(np.max(values)),
    }


def _bifurcation_counts(G: nx.Graph) -> Dict[int, int]:
    """Count bifurcations by degree (3-way, 4-way, etc.).

    Returns:
        dict mapping degree -> count  (only degrees >= 3).
    """
    counts: Dict[int, int] = {}
    for node in G.nodes():
        deg = G.degree(node)
        if deg >= 3:
            counts[deg] = counts.get(deg, 0) + 1
    return counts


def _asymmetry_ratio(G: nx.Graph) -> dict:
    """Compute branching asymmetry ratio per Zamir (1999).

    For each bifurcation node (degree >= 3) the mean edge radius
    along each incident branch is used.  The largest-radius branch
    is taken as the parent, the two next-largest as daughters.
    delta = r_minor / r_major.

    Returns:
        dict with "mean", "std", "median".
    """
    deltas: list[float] = []
    for node in G.nodes():
        neighbours = list(G.neighbors(node))
        if len(neighbours) < 3:
            continue
        edge_radii = sorted(
            [G.edges[node, nb].get("mean_radius", 0.0) for nb in neighbours],
            reverse=True,
        )
        r_major = edge_radii[1]  # largest = parent, next two = daughters
        r_minor = edge_radii[2]
        if r_major > 0:
            deltas.append(r_minor / r_major)

    if not deltas:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean":   float(np.mean(deltas)),
        "std":    float(np.std(deltas)),
        "median": float(np.median(deltas)),
    }
