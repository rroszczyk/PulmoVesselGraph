"""Stage 05 - Graph construction from skeleton.

    1. Skeleton -> skan -> NetworkX graph (nodes with pos/pos_mm, edges with path/path_mm/length_mm)
    2. Cycle removal: MST per connected component (keep longest edges)
    3. Spur pruning: remove degree-1 nodes with short edges
    4. Small component removal
    5. Node type classification: endpoint / junction / path

Input:
    skeleton_left, skeleton_right : np.uint8 volumes from stage_04

Output (cache keys):
    graph_left  : NetworkX Graph
    graph_right : NetworkX Graph
"""
import logging
import time

import networkx as nx
import numpy as np

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def run(
    cache: CacheManager,
    skeleton_left: np.ndarray,
    skeleton_right: np.ndarray,
    spacing: np.ndarray,
    min_spur_length_mm: float = 2.0,
    min_component_nodes: int = 5,
    kimimaro_left: dict | None = None,
    kimimaro_right: dict | None = None,
    **_: object,
) -> dict:
    """
    When kimimaro_left / kimimaro_right dicts are provided (keys:
    vertices, edges, radii), the graph is built directly from the kimimaro output

    Args:
        cache:                CacheManager instance.
        skeleton_left:        Binary skeleton uint8 for left lung.
        skeleton_right:       Binary skeleton uint8 for right lung.
        spacing:              Voxel spacing [z,y,x] mm.
        min_spur_length_mm:   Prune degree-1 branches shorter than this [mm].
        min_component_nodes:  Remove connected components with fewer nodes.
        kimimaro_left:        Raw kimimaro data for left lung (optional).
        kimimaro_right:       Raw kimimaro data for right lung (optional).

    Returns:
        dict with graph_left, graph_right.
    """
    kimimaro_sides = {"left": kimimaro_left, "right": kimimaro_right}
    use_direct = any(v is not None for v in kimimaro_sides.values())

    params = {
        "min_spur_length_mm":    min_spur_length_mm,
        "min_component_nodes":   min_component_nodes,
        "skeleton_voxels_left":  int(skeleton_left.sum()),
        "skeleton_voxels_right": int(skeleton_right.sum()),
        "shape":                 list(skeleton_left.shape),
        "direct_kimimaro":       use_direct,
    }
    cached = cache.get("stage_05_graph_construction", params)
    if cached is not None:
        return cached

    t0 = time.time()
    result = {}

    for side, skeleton in (("left", skeleton_left), ("right", skeleton_right)):
        ts = time.time()
        kim = kimimaro_sides[side]

        if kim is not None and len(kim["vertices"]) > 0:
            logger.info("[%s] Building graph directly from kimimaro (%d vertices)",
                        side, len(kim["vertices"]))
            G = _build_graph_from_kimimaro(
                kim["vertices"], kim["edges"], kim["radii"], spacing, side,
            )
        else:
            n_vox = int(skeleton.sum())
            logger.info("[%s] Skeleton: %d voxels (skan fallback)", side, n_vox)
            if n_vox == 0:
                result[f"graph_{side}"] = nx.Graph()
                continue
            G = _build_graph(skeleton, spacing, side)

        tc = time.time()
        G = _remove_cycles(G)
        logger.info(
            "[%s] Acyclic graph: %d nodes, %d edges  (%.1fs)",
            side, G.number_of_nodes(), G.number_of_edges(), time.time() - tc,
        )

        td = time.time()
        G = _prune_and_simplify(G, min_spur_length_mm, min_component_nodes)
        logger.info(
            "[%s] Final graph: %d nodes, %d edges  (%.1fs)",
            side, G.number_of_nodes(), G.number_of_edges(), time.time() - td,
        )

        logger.info("[%s] Total: %.1fs", side, time.time() - ts)
        result[f"graph_{side}"] = G

    cache.put(
        "stage_05_graph_construction",
        params,
        result,
        numpy_keys=[],
    )
    logger.info("Stage 05 done  total=%.1fs", time.time() - t0)
    return result


def _build_graph_from_kimimaro(
    vertices: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    spacing: np.ndarray,
    side: str,
) -> nx.Graph:
    """Build compressed NetworkX graph directly from skeleton.

    Degree-2 chains are compressed into single edges with full path
    attributes (path, path_mm, length_mm, n_voxels), matching the
    output format of _build_graph. This avoids the
    lossy graph -> raster -> graph round-trip that quantises sub-voxel
    positions and can merge nearby branches.
    """
    t0 = time.perf_counter()
    sp = spacing.astype(np.float64)
    n_verts = len(vertices)

    if n_verts == 0:
        return nx.Graph()

    G_raw = nx.Graph()

    pos_vox = np.round(vertices / sp[np.newaxis, :]).astype(np.int64)
    for i in range(n_verts):
        G_raw.add_node(i,
                       pos=tuple(int(v) for v in pos_vox[i]),
                       pos_mm=tuple(float(v) for v in vertices[i]))

    for src, dst in edges:
        src, dst = int(src), int(dst)
        if src == dst or src >= n_verts or dst >= n_verts:
            continue
        d = float(np.linalg.norm(vertices[dst] - vertices[src]))
        G_raw.add_edge(src, dst, _len=d)

    logger.info("[%s] Kimimaro raw graph: V=%d  E=%d",
                side, G_raw.number_of_nodes(), G_raw.number_of_edges())

    # Significant nodes: degree != 2 (junctions, endpoints, isolates)
    sig = {n for n in G_raw.nodes() if G_raw.degree(n) != 2}
    if not sig and G_raw.number_of_nodes() > 0:
        sig = {next(iter(G_raw.nodes()))}  # pure cycle edge case

    G = nx.Graph()
    old_to_new: dict = {}
    nid = 0
    for n in sig:
        old_to_new[n] = nid
        G.add_node(nid,
                   pos=G_raw.nodes[n]["pos"],
                   pos_mm=G_raw.nodes[n]["pos_mm"])
        nid += 1

    visited: set = set()

    for start in sig:
        for nb in G_raw.neighbors(start):
            ek = (min(start, nb), max(start, nb))
            if ek in visited:
                continue

            chain = [start, nb]
            visited.add(ek)
            prev, cur = start, nb

            while cur not in sig:
                nbs = list(G_raw.neighbors(cur))
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                visited.add((min(cur, nxt), max(cur, nxt)))
                chain.append(nxt)
                prev, cur = cur, nxt

            new_src = old_to_new[chain[0]]
            new_dst = old_to_new[chain[-1]]

            path_vox = [G_raw.nodes[n]["pos"] for n in chain]
            path_mm  = [G_raw.nodes[n]["pos_mm"] for n in chain]
            length_mm = sum(
                G_raw[chain[i]][chain[i + 1]]["_len"]
                for i in range(len(chain) - 1)
            )

            if G.has_edge(new_src, new_dst):  # keep shorter if duplicate
                if length_mm >= G[new_src][new_dst].get("length_mm", 0):
                    continue

            G.add_edge(new_src, new_dst,
                       path=path_vox,
                       path_mm=path_mm,
                       length_mm=length_mm,
                       n_voxels=len(path_vox))

    for n in G.nodes():
        deg = G.degree(n)
        G.nodes[n]["type"] = (
            "endpoint" if deg <= 1 else ("junction" if deg >= 3 else "path")
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] Graph from kimimaro (direct): V=%d, E=%d  (%.1fs)",
        side, G.number_of_nodes(), G.number_of_edges(), elapsed,
    )
    return G


def _build_graph(
    skeleton: np.ndarray,
    spacing: np.ndarray,
    side: str,
) -> nx.Graph:
    """Convert 1-voxel skeleton to NetworkX graph using skan."""
    from skan import Skeleton as SkanSkeleton, summarize

    t0 = time.perf_counter()
    sp = spacing.astype(np.float64)

    skel_obj = SkanSkeleton(
        skeleton.astype(bool),
        spacing=tuple(float(s) for s in sp),
    )
    summary = summarize(skel_obj, find_main_branch=False, separator='-')
    nb = len(summary)
    logger.info("[%s] skan: %d segments (%.1fs)", side, nb, time.perf_counter() - t0)

    src_ids = summary['node-id-src'].values.astype(int)
    dst_ids = summary['node-id-dst'].values.astype(int)

    all_ids = np.concatenate([src_ids, dst_ids])
    max_id = int(all_ids.max()) + 1 if len(all_ids) > 0 else 0
    degree = np.bincount(all_ids, minlength=max_id)

    node_coords: dict = {}
    for i in range(nb):
        s, d = int(src_ids[i]), int(dst_ids[i])
        if s not in node_coords:
            node_coords[s] = skel_obj.path_coordinates(i)[0]
        if d not in node_coords:
            node_coords[d] = skel_obj.path_coordinates(i)[-1]

    G = nx.Graph()
    for nid, cv in node_coords.items():
        pos_mm = tuple(float(c) for c in cv)
        pos_vox = tuple(int(round(c / float(s))) for c, s in zip(cv, sp))
        deg = int(degree[nid]) if nid < len(degree) else 0
        G.add_node(
            nid,
            pos=pos_vox,
            pos_mm=pos_mm,
            type="endpoint" if deg == 1 else "junction",
        )

    for i in range(nb):
        s, d = int(src_ids[i]), int(dst_ids[i])
        if s == d or s not in G or d not in G:
            continue

        pm = skel_obj.path_coordinates(i)

        path_vox = [
            (int(round(c[0] / float(sp[0]))),
             int(round(c[1] / float(sp[1]))),
             int(round(c[2] / float(sp[2]))))
            for c in pm
        ]
        path_mm = [tuple(float(c) for c in p) for p in pm]
        length_mm = float(np.sum(np.linalg.norm(np.diff(pm, axis=0), axis=1)))

        if G.has_edge(s, d) and length_mm >= G[s][d].get("length_mm", 0):  # keep shorter if duplicate
            continue

        G.add_edge(
            s, d,
            path=path_vox,
            path_mm=path_mm,
            length_mm=length_mm,
            n_voxels=len(path_vox),
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] Graph: V=%d, E=%d (%.1fs)",
        side, G.number_of_nodes(), G.number_of_edges(), elapsed,
    )
    return G


def _remove_cycles(G: nx.Graph) -> nx.Graph:
    """Ensure G is acyclic by applying MST per connected component.

    Uses negative length as weight so MST keeps the longest edges.
    """
    ne = G.number_of_edges()
    nn = G.number_of_nodes()
    nc = nx.number_connected_components(G)
    n_cyc = ne - nn + nc

    if n_cyc == 0:
        logger.info("  No cycles - OK")
        return G

    logger.warning("  Cycles detected (%d) - applying MST removal per component.", n_cyc)

    for u, v, d in G.edges(data=True):
        d["_neg_len"] = -d.get("length_mm", 1.0)

    mst_edges = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_edges() <= sub.number_of_nodes() - 1:
            for eu, ev in sub.edges():
                mst_edges.add((min(eu, ev), max(eu, ev)))
        else:
            mst = nx.minimum_spanning_tree(sub, weight="_neg_len")
            for eu, ev in mst.edges():
                mst_edges.add((min(eu, ev), max(eu, ev)))

    to_rm = [
        (u, v) for u, v in list(G.edges())
        if (min(u, v), max(u, v)) not in mst_edges
    ]
    for u, v in to_rm:
        G.remove_edge(u, v)

    for u, v, d in G.edges(data=True):
        d.pop("_neg_len", None)
    iso = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(iso)

    n_cyc2 = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
    logger.info("  Removed %d edges, cycles remaining: %d", len(to_rm), n_cyc2)
    return G


def _prune_and_simplify(
    G: nx.Graph,
    min_spur_length_mm: float,
    min_component_nodes: int,
) -> nx.Graph:
    """Remove short spurs, small components, and classify node types."""
    removed = 0
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if n not in G:
                continue
            if G.degree(n) == 1:
                edges = list(G.edges(n, data=True))
                if edges and edges[0][2].get("length_mm", 999) < min_spur_length_mm:
                    G.remove_node(n)
                    removed += 1
                    changed = True

    small_removed = 0
    for comp in list(nx.connected_components(G)):
        if len(comp) < min_component_nodes:
            G.remove_nodes_from(comp)
            small_removed += 1

    for n in G.nodes():
        deg = G.degree(n)
        G.nodes[n]["type"] = (
            "endpoint" if deg <= 1 else ("junction" if deg >= 3 else "path")
        )

    if removed > 0 or small_removed > 0:
        logger.info("  Spurs: %d, small components: %d", removed, small_removed)

    return G
