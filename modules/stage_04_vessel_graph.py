"""Stage 04 - Skeleton extraction via TEASAR (kimimaro).

Pipeline:
    1. Growth mask: vesselness > threshold -> binary_closing() -> remove_small_objects()
    2. L/R lung separation on growth mask
    3. Per lung: kimimaro.skeletonize directly on growth mask -> skeleton

Output (cache keys):
    skeleton_left, skeleton_right, skeleton_combined : np.uint8
    edt_left, edt_right : np.float32
    left_mask, right_mask : np.uint8
    kimimaro_<side>_vertices, _edges, _radii : np.ndarray
"""
import logging
import time
from typing import List, Tuple

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label as nd_label
from skimage.morphology import ball, remove_small_objects

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def run(
    cache: CacheManager,
    vesselness_final: np.ndarray,
    spacing: np.ndarray,
    vesselness_threshold: float = 0.01,
    min_lung_voxels: int = 1000,
    min_skeleton_fragment: int = 5,
    teasar_scale: float = 1.5,
    teasar_const: float = 0.5,
    teasar_pdrf_scale: int = 5000,
    teasar_pdrf_exponent: int = 4,
    **_: object,
) -> dict:
    """Extract vascular skeletons for left and right lung via TEASAR (kimimaro).

    Returns:
        dict with skeleton_left, skeleton_right, skeleton_combined,
                  edt_left, edt_right, left_mask, right_mask,
                  kimimaro_<side>_vertices, _edges, _radii.
    """
    params = {
        "vesselness_threshold":  vesselness_threshold,
        "min_lung_voxels":       min_lung_voxels,
        "min_skeleton_fragment": min_skeleton_fragment,
        "teasar_scale":          teasar_scale,
        "teasar_const":          teasar_const,
        "teasar_pdrf_scale":     teasar_pdrf_scale,
        "teasar_pdrf_exponent":  teasar_pdrf_exponent,
        "shape":                 list(vesselness_final.shape),
    }
    cached = cache.get("stage_04_vessel_graph", params)
    if cached is not None:
        return cached

    t0 = time.time()
    shape = vesselness_final.shape

    mask = vesselness_final > vesselness_threshold
    mask = binary_closing(mask, structure=ball(1))
    mask = remove_small_objects(mask, min_size=50, connectivity=3)
    logger.info("Growth mask: %d voxels (threshold=%.3f)", int(mask.sum()), vesselness_threshold)

    lungs = _separate_lungs(mask, min_lung_voxels)
    if not lungs:
        logger.warning("No lung regions found - returning empty result.")
        empty_skel = np.zeros(shape, dtype=np.uint8)
        empty_edt = np.zeros(shape, dtype=np.float32)
        empty_mask = np.zeros(shape, dtype=np.uint8)
        return {
            "skeleton_left": empty_skel, "skeleton_right": empty_skel,
            "skeleton_combined": empty_skel,
            "edt_left": empty_edt, "edt_right": empty_edt,
            "left_mask": empty_mask, "right_mask": empty_mask,
        }

    result: dict = {}
    skeleton_combined = np.zeros(shape, dtype=np.uint8)

    for side, lung_mask_bool in lungs:
        ts = time.time()
        logger.info("Processing [%s] lung - %d voxels", side, int(lung_mask_bool.sum()))

        skeleton, edt, kimimaro_data = _process_lung_teasar(
            lung_mask_bool, shape, spacing,
            teasar_scale=teasar_scale,
            teasar_const=teasar_const,
            teasar_pdrf_scale=teasar_pdrf_scale,
            teasar_pdrf_exponent=teasar_pdrf_exponent,
            min_skeleton_fragment=min_skeleton_fragment,
            side_name=side,
        )
        result[f"kimimaro_{side}_vertices"] = kimimaro_data["vertices"]
        result[f"kimimaro_{side}_edges"]    = kimimaro_data["edges"]
        result[f"kimimaro_{side}_radii"]    = kimimaro_data["radii"]

        result[f"skeleton_{side}"] = skeleton
        result[f"edt_{side}"] = edt
        result[f"{side}_mask"] = lung_mask_bool.astype(np.uint8)
        skeleton_combined = np.maximum(skeleton_combined, skeleton)

        logger.info(
            "[%s] Done: %d skeleton voxels  (%.1fs)",
            side, int(skeleton.sum()), time.time() - ts,
        )

    result["skeleton_combined"] = skeleton_combined

    numpy_keys = [
        "skeleton_left", "skeleton_right", "skeleton_combined",
        "edt_left", "edt_right", "left_mask", "right_mask",
    ]
    for side in ("left", "right"):
        for suffix in ("vertices", "edges", "radii"):
            k = f"kimimaro_{side}_{suffix}"
            if k in result:
                numpy_keys.append(k)

    cache.put(
        "stage_04_vessel_graph",
        params,
        result,
        numpy_keys=numpy_keys,
    )
    logger.info("Stage 04 done  total=%.1fs", time.time() - t0)
    return result


def _process_lung_teasar(
    lung_mask: np.ndarray,
    shape: tuple,
    spacing: np.ndarray,
    *,
    teasar_scale: float,
    teasar_const: float,
    teasar_pdrf_scale: int,
    teasar_pdrf_exponent: int,
    min_skeleton_fragment: int,
    side_name: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract skeleton via TEASAR (kimimaro).

    Returns:
        (skeleton_volume, edt, kimimaro_data)
        kimimaro_data has keys "vertices" (N,3), "edges" (M,2), "radii" (N,).
    """
    import kimimaro

    t0 = time.time()

    # EDT for output (vessel radius estimation)
    edt = distance_transform_edt(lung_mask.astype(np.uint8)).astype(np.float32)
    logger.info("[%s] EDT max: %.1f", side_name, float(edt.max()))

    labels = lung_mask.astype(np.uint8)  # kimimaro expects labeled uint8, label=1

    sp = np.asarray(spacing, dtype=np.float64)
    anisotropy = (float(sp[0]), float(sp[1]), float(sp[2]))  # z, y, x (C-order)

    logger.info(
        "[%s] TEASAR: scale=%.2f const=%.1f pdrf_scale=%d pdrf_exp=%d aniso=%s",
        side_name, teasar_scale, teasar_const, teasar_pdrf_scale, teasar_pdrf_exponent,
        anisotropy,
    )

    skeletons = kimimaro.skeletonize(
        labels,
        teasar_params={
            "scale": teasar_scale,
            "const": teasar_const,
            "pdrf_scale": teasar_pdrf_scale,
            "pdrf_exponent": teasar_pdrf_exponent,
            "soma_acceptance_threshold": 0,
            "soma_detection_threshold": 0,
        },
        anisotropy=anisotropy,
        dust_threshold=min_skeleton_fragment,
        fix_branching=True,
        progress=False,
        in_place=True,
    )

    logger.info("[%s] TEASAR done (%.1fs), %d skeleton(s)", side_name, time.time() - t0, len(skeletons))

    all_verts: list = []
    all_edges: list = []
    all_radii: list = []
    offset = 0
    for seg_id, skel in skeletons.items():
        all_verts.append(skel.vertices)
        all_edges.append(skel.edges + offset)
        r = skel.radii if skel.radii is not None else np.zeros(len(skel.vertices))
        all_radii.append(r)
        offset += len(skel.vertices)

    if all_verts:
        kimimaro_data = {
            "vertices": np.concatenate(all_verts).astype(np.float32),
            "edges":    np.concatenate(all_edges).astype(np.int32),
            "radii":    np.concatenate(all_radii).astype(np.float32),
        }
    else:
        kimimaro_data = {
            "vertices": np.empty((0, 3), dtype=np.float32),
            "edges":    np.empty((0, 2), dtype=np.int32),
            "radii":    np.empty((0,),   dtype=np.float32),
        }

    logger.info(
        "[%s] Kimimaro raw: %d vertices, %d edges",
        side_name, len(kimimaro_data["vertices"]), len(kimimaro_data["edges"]),
    )

    # rasterized volume needed for NIfTI/napari output
    skeleton_vol = np.zeros(shape, dtype=np.uint8)
    for seg_id, skel in skeletons.items():
        verts = skel.vertices
        voxels_z = np.round(verts[:, 0] / float(sp[0])).astype(np.int64)
        voxels_y = np.round(verts[:, 1] / float(sp[1])).astype(np.int64)
        voxels_x = np.round(verts[:, 2] / float(sp[2])).astype(np.int64)

        for edge_src, edge_dst in skel.edges:
            src = np.array([voxels_z[edge_src], voxels_y[edge_src], voxels_x[edge_src]])
            dst = np.array([voxels_z[edge_dst], voxels_y[edge_dst], voxels_x[edge_dst]])
            n_steps = int(np.max(np.abs(dst - src))) + 1
            for t in np.linspace(0, 1, n_steps):
                pt = np.round(src + t * (dst - src)).astype(int)
                z, y, x = pt
                if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                    skeleton_vol[z, y, x] = 1

        logger.info(
            "[%s] Skeleton seg %d: %d vertices, %d edges",
            side_name, seg_id, len(skel.vertices), len(skel.edges),
        )

    skeleton_clean = remove_small_objects(
        skeleton_vol.astype(bool),
        min_size=min_skeleton_fragment,
        connectivity=3,
    ).astype(np.uint8)

    logger.info(
        "[%s] TEASAR skeleton: %d -> %d voxels (cleaned)  (%.1fs total)",
        side_name, int(skeleton_vol.sum()), int(skeleton_clean.sum()), time.time() - t0,
    )
    return skeleton_clean, edt, kimimaro_data


def _separate_lungs(
    mask: np.ndarray,
    min_lung_voxels: int,
) -> List[Tuple[str, np.ndarray]]:
    """Split growth mask into L/R lungs by centroid X position."""
    labeled, n_comp = nd_label(binary_closing(mask, structure=ball(2)))

    sizes = []
    for i in range(1, n_comp + 1):
        sz = int(np.sum(labeled == i))
        sizes.append((i, sz))
    sizes.sort(key=lambda x: -x[1])

    lungs: list = []
    for lid, sz in sizes:
        if sz < min_lung_voxels:
            continue
        lmask = (labeled == lid) & mask
        coords = np.argwhere(lmask)
        lungs.append({
            "mask": lmask,
            "size": int(np.sum(lmask)),
            "mean_x": float(coords[:, 2].mean()),
        })
        if len(lungs) >= 2:
            break

    lungs.sort(key=lambda l: -l["mean_x"])

    if len(lungs) >= 2:
        logger.info(
            "Lung separation: LEFT=%d vox (x̄=%.0f)  RIGHT=%d vox (x̄=%.0f)",
            lungs[0]["size"], lungs[0]["mean_x"],
            lungs[1]["size"], lungs[1]["mean_x"],
        )
        return [("left", lungs[0]["mask"]), ("right", lungs[1]["mask"])]

    if len(lungs) == 1:
        logger.warning("Single component - splitting by X midpoint.")
        m = lungs[0]["mask"]
        mid_x = m.shape[2] // 2
        left_m = m.copy()
        left_m[:, :, :mid_x] = False
        right_m = m.copy()
        right_m[:, :, mid_x:] = False
        return [("left", left_m), ("right", right_m)]

    return []


def graph_to_skeleton(G, shape: Tuple[int, int, int]) -> np.ndarray:
    """Rasterise graph edge paths to a binary skeleton volume."""
    skel = np.zeros(shape, dtype=np.uint8)
    for _u, _v, data in G.edges(data=True):
        for pos in data.get("path", []):
            z, y, x = pos
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                skel[z, y, x] = 1
    return skel
