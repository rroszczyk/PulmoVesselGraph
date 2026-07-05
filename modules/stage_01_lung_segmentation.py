"""Stage 01 - Lung segmentation from HU volume.

Pipeline (sequential):
    a) Body mask:     HU > -600 -> closing -> largest component
    b) Lung thresh:   -1000 ≤ HU ≤ -400 & body_mask -> remove small
    c) Airway break:  erode(ball(2)) -> top-2 components -> dilate(ball(2))
    d) Postprocess:   closing(ball(2)) -> fill holes
    e) Final:         top-2 components & body_mask
    f) [optional]     Morphological Chan-Vese refinement at reduced resolution
    g) Eroded mask:   distance_transform_edt ≥ lung_wall_erosion_mm
                      -> excludes pleural wall from vesselness computation

Input:
    hu_iso    : np.int16  (Z, Y, X)
    spacing   : np.float32 (3,) - voxel spacing [z, y, x] in mm

Output (cache keys):
    lung_mask        : np.uint8   - binary lung mask (full)
    lung_mask_eroded : np.uint8   - lung mask eroded by lung_wall_erosion_mm (for vesselness)
    stage_body       : np.uint8   - body mask
    stage_lungs0     : np.uint8   - after threshold
    stage_lungs1     : np.uint8   - after airway disconnection
    stage_lungs2     : np.uint8   - after postprocessing
"""
import logging
import time

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label,
    zoom as ndimage_zoom,
)
from skimage.morphology import ball, remove_small_objects

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def _largest_components(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """Return binary mask keeping only the *n* largest connected components."""
    labeled, num = label(mask)
    if num == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    top_labels = np.argsort(sizes)[::-1][:n]
    result = np.zeros_like(mask, dtype=np.uint8)
    for lbl in top_labels:
        result[labeled == lbl] = 1
    return result


def run(
    cache: CacheManager,
    hu_iso: np.ndarray,
    spacing: np.ndarray,
    body_thresh: float = -600.0,
    lung_low: float = -1000.0,
    lung_high: float = -400.0,
    airway_break_radius: int = 2,
    close_radius: int = 2,
    use_refine: bool = True,
    refine_target_spacing: float = 2.5,
    refine_ac_iters: int = 35,
    lung_wall_erosion_mm: float = 3.0,
    **_: object,
) -> dict:
    """Segment lungs from isotropic HU volume.

    Args:
        cache:                 CacheManager instance.
        hu_iso:                HU volume (Z, Y, X) int16.
        spacing:               Voxel spacing [z, y, x] in mm.
        body_thresh:           HU threshold for body mask [HU].
        lung_low:              Lower HU bound for lung region [HU].
        lung_high:             Upper HU bound for lung region [HU].
        airway_break_radius:   Ball radius for airway disconnection erosion.
        close_radius:          Ball radius for morphological closing.
        use_refine:            If True, apply Chan-Vese refinement.
        refine_target_spacing: Down-sample spacing for Chan-Vese [mm].
        refine_ac_iters:       Chan-Vese iterations.
        lung_wall_erosion_mm:  Erode lung mask by this distance [mm] to exclude
                               pleural wall from vesselness computation.

    Returns:
        dict with lung_mask, lung_mask_eroded, stage_body, stage_lungs0,
                  stage_lungs1, stage_lungs2.
    """
    params = {
        "body_thresh":           body_thresh,
        "lung_low":              lung_low,
        "lung_high":             lung_high,
        "airway_break_radius":   airway_break_radius,
        "close_radius":          close_radius,
        "use_refine":            use_refine,
        "refine_target_spacing": refine_target_spacing,
        "refine_ac_iters":       refine_ac_iters,
        "lung_wall_erosion_mm":  lung_wall_erosion_mm,
        "shape":                 list(hu_iso.shape),
    }
    cached = cache.get("stage_01_lung_segmentation", params)
    if cached is not None:
        return cached

    t0 = time.time()
    hu = hu_iso.astype(np.float32)

    ta = time.time()
    body_raw = (hu > body_thresh)
    body_closed = binary_closing(body_raw, structure=ball(3))
    stage_body = _largest_components(body_closed, n=1)
    stage_body = remove_small_objects(stage_body.astype(bool), min_size=50_000)
    stage_body = stage_body.astype(np.uint8)
    logger.info("Body mask done  (%.1fs)", time.time() - ta)

    tb = time.time()
    lungs0_raw  = ((hu >= lung_low) & (hu <= lung_high)) & (stage_body > 0)
    lungs0_raw  = remove_small_objects(lungs0_raw, min_size=5_000)
    stage_lungs0 = lungs0_raw.astype(np.uint8)
    logger.info("Lung threshold done  (%.1fs)", time.time() - tb)

    tc = time.time()
    eroded   = binary_erosion(lungs0_raw, structure=ball(airway_break_radius))
    top2     = _largest_components(eroded, n=2)
    lungs1   = binary_dilation(top2.astype(bool), structure=ball(airway_break_radius))
    stage_lungs1 = lungs1.astype(np.uint8)
    logger.info("Airway disconnection done  (%.1fs)", time.time() - tc)

    td = time.time()
    lungs2 = binary_closing(lungs1, structure=ball(close_radius))
    lungs2 = binary_fill_holes(lungs2)
    stage_lungs2 = lungs2.astype(np.uint8)
    logger.info("Postprocessing done  (%.1fs)", time.time() - td)

    te = time.time()
    lung_final = _largest_components(stage_lungs2.astype(bool), n=2)
    lung_final = (lung_final & (stage_body > 0)).astype(np.uint8)

    if use_refine:
        try:
            lung_final = _refine_chan_vese(
                hu_iso, lung_final, spacing,
                refine_target_spacing, refine_ac_iters,
            )
        except Exception as exc:
            logger.warning("Chan-Vese refinement failed (%s), using raw mask.", exc)

    logger.info(
        "Final lung mask voxels=%d  (%.1fs)",
        int(lung_final.sum()), time.time() - te,
    )

    tg = time.time()
    lung_mask_eroded = _erode_by_distance(lung_final, spacing, lung_wall_erosion_mm)
    logger.info(
        "Eroded mask (%.1fmm): voxels=%d -> %d  (%.1fs)",
        lung_wall_erosion_mm,
        int(lung_final.sum()), int(lung_mask_eroded.sum()),
        time.time() - tg,
    )

    result = {
        "lung_mask":        lung_final,
        "lung_mask_eroded": lung_mask_eroded,
        "stage_body":       stage_body,
        "stage_lungs0":     stage_lungs0,
        "stage_lungs1":     stage_lungs1,
        "stage_lungs2":     stage_lungs2,
    }
    cache.put(
        "stage_01_lung_segmentation",
        params,
        result,
        numpy_keys=list(result.keys()),
    )
    logger.info("Stage 01 done  total=%.1fs", time.time() - t0)
    return result


def _erode_by_distance(
    mask: np.ndarray,
    spacing: np.ndarray,
    erosion_mm: float,
) -> np.ndarray:
    """Erode binary mask by exact distance in mm using distance transform.

    Voxels whose distance to the mask boundary is < erosion_mm are removed.
    This gives precise mm-based erosion regardless of voxel size or isotropy.

    Args:
        mask:       Binary uint8 mask.
        spacing:    Voxel spacing [z, y, x] mm.
        erosion_mm: Minimum inward distance from wall to keep [mm].

    Returns:
        Eroded mask as uint8.
    """
    if erosion_mm <= 0:
        return mask.copy()
    # EDT: distance of each foreground voxel to nearest background voxel
    dist = distance_transform_edt(mask > 0, sampling=spacing.tolist())
    return (dist >= erosion_mm).astype(np.uint8)


def _refine_chan_vese(
    hu_iso: np.ndarray,
    lung_mask: np.ndarray,
    spacing: np.ndarray,
    target_spacing: float,
    ac_iters: int,
) -> np.ndarray:
    """Chan-Vese refinement at reduced resolution.

    Crops ROI with 10 mm margin, down-samples to target_spacing mm,
    runs morphological Chan-Vese, then up-samples and pastes back.

    Args:
        hu_iso:         Original HU volume.
        lung_mask:      Current binary lung mask.
        spacing:        Original voxel spacing [z, y, x] mm.
        target_spacing: Reduced resolution spacing mm.
        ac_iters:       Chan-Vese iterations.

    Returns:
        Refined lung_mask (uint8).
    """
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import morphological_chan_vese

    t0 = time.time()

    # Crop ROI with 10 mm margin
    margin_vox = [max(1, int(round(10.0 / s))) for s in spacing]
    coords = np.argwhere(lung_mask > 0)
    if coords.size == 0:
        return lung_mask

    lo = np.maximum(coords.min(axis=0) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=0) + np.array(margin_vox) + 1,
                    np.array(hu_iso.shape))

    roi_hu   = hu_iso[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    roi_mask = lung_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # Down-sample
    zoom_factor = (spacing / target_spacing).tolist()
    roi_small = ndimage_zoom(roi_hu, zoom_factor, order=1, prefilter=False)

    # Normalize to [0, 1]
    mn, mx = roi_small.min(), roi_small.max()
    if mx > mn:
        roi_norm = (roi_small - mn) / (mx - mn)
    else:
        return lung_mask

    # Gaussian blur sigma=1 mm in voxels of target spacing
    sigma_vox = max(0.5, 1.0 / target_spacing)
    roi_blur = gaussian_filter(roi_norm, sigma=sigma_vox)

    # Chan-Vese
    init_ls = ndimage_zoom(
        roi_mask.astype(np.float32), zoom_factor, order=0, prefilter=False
    )
    init_ls = (init_ls > 0.5).astype(np.float32) * 2 - 1  # +-1 level set

    cv_result = morphological_chan_vese(
        roi_blur,
        num_iter=ac_iters,
        init_level_set=init_ls,
        smoothing=2,
    )

    # Up-sample back to original resolution
    up_factor = [1.0 / z for z in zoom_factor]
    cv_up = ndimage_zoom(cv_result.astype(np.float32), up_factor, order=0, prefilter=False)

    # Clip to ROI shape and paste back
    target_shape = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    cv_up = cv_up[: target_shape[0], : target_shape[1], : target_shape[2]]
    # Pad if zoom rounding made it smaller
    pad_widths = [(0, max(0, t - c)) for t, c in zip(target_shape, cv_up.shape)]
    cv_up = np.pad(cv_up, pad_widths, mode="edge")

    refined = lung_mask.copy()
    refined[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (cv_up > 0).astype(np.uint8)

    logger.info("Chan-Vese refinement done  (%.1fs)", time.time() - t0)
    return refined
