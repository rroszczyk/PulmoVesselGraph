"""Stage 03 - Multi-scale Hessian vesselness (Frangi + Sato + Jerman hybrid).

Pipeline:

Key elemeents:
  - Each filter is cached independently (params = sigmas + shape).
    Changing weights only re-runs the fusion, not the expensive filters.
  - Normalization per filter: robust scaling using 99.9th percentile
    of non-zero values in eroded lung mask. This puts all filters
    on the same [0,1] scale before fusion so that no filter
    dominates due to its value range.

Output (cache keys):
    v_frangi_clean   : np.float32 [0,1]
    v_sato_clean     : np.float32 [0,1]
    v_jerman_clean   : np.float32 [0,1]
    vesselness       : np.float32 [0,1]  - weighted max fusion
    vesselness_final : np.float32 [0,1]  - percentile-normalised (pass to Stage 4)
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, sato

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def _run_frangi(hu_input: np.ndarray, sigmas: list) -> np.ndarray:
    return frangi(hu_input, sigmas=sigmas, alpha=0.5, beta=0.5, gamma=15,
                  black_ridges=False)


def _run_sato(hu_input: np.ndarray, sigmas: list) -> np.ndarray:
    return sato(hu_input, sigmas=sigmas, black_ridges=False)


def _run_jerman(hu_input: np.ndarray, sigmas: list,
                tau: float = 0.75) -> np.ndarray:
    """Jerman vesselness filter (Jerman et al. 2016)."""
    result = np.zeros_like(hu_input, dtype=np.float64)

    for sigma in sigmas:
        H    = hessian_matrix(hu_input, sigma=sigma, mode="reflect",
                              use_gaussian_derivatives=True)
        eigs = hessian_matrix_eigvals(H)

        eig_stack = np.stack(eigs, axis=0)
        sort_idx  = np.argsort(np.abs(eig_stack), axis=0)
        e2 = np.take_along_axis(eig_stack, sort_idx[1:2], axis=0)[0]
        e3 = np.take_along_axis(eig_stack, sort_idx[2:3], axis=0)[0]

        e3_neg = np.minimum(e3, 0)
        if not np.any(e3_neg < 0):
            continue
        e3_max = float(np.abs(np.percentile(e3_neg[e3_neg < 0], 2)))
        if e3_max < 1e-10:
            continue

        lam_rho = np.where(e3 <= tau * (-e3_max), e3, tau * (-e3_max))
        diff    = np.maximum(e2 - lam_rho, 0)
        abs_sum = np.maximum(np.abs(e2 + lam_rho), 1e-10)

        V          = (e2 ** 2) * diff * (3.0 / abs_sum) ** 3
        V[e2 >= 0] = 0
        V[e3 >= 0] = 0
        result = np.maximum(result, np.maximum(V, 0))

    return result


def _robust_scale(
    vesselness_map: np.ndarray,
    lung_mask: np.ndarray,
) -> np.ndarray:
    """Mask + robust scaling to [0, 1] using 99.9th percentile.

    Each filter is independently normalised so that Sato's native
    range (~0–300) and Frangi's (~0–0.9) end up on the same [0,1]
    scale before fusion.
    """
    v = np.where(lung_mask > 0, vesselness_map, 0.0)
    non_zero = v[v > 1e-5]
    if len(non_zero) > 0:
        vmax = float(np.percentile(non_zero, 99.9))
        if vmax > 0:
            v = v / vmax
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def _filter_cache_params(sigmas: list, shape: list) -> dict:
    """Params dict for individual filter cache - weights excluded."""
    return {"vessel_sigmas": sigmas, "shape": shape}


def _get_or_compute_filter(
    cache: CacheManager,
    cache_stage: str,
    filter_fn,
    filter_kwargs: dict,
    hu_input: np.ndarray,
    sigmas: list,
    active_mask: np.ndarray,
) -> np.ndarray:
    params = _filter_cache_params(sigmas, list(hu_input.shape))
    cached = cache.get(cache_stage, params)
    if cached is not None:
        return cached["map"]

    t0 = time.time()
    raw = filter_fn(hu_input, sigmas, **filter_kwargs)
    scaled = _robust_scale(raw, active_mask)
    logger.info(
        "Filter %s done  (%.1fs)  max=%.4f",
        cache_stage.split("_")[-1], time.time() - t0, float(scaled.max()),
    )

    cache.put(cache_stage, params, {"map": scaled}, numpy_keys=["map"])
    return scaled


def run(
    cache: CacheManager,
    smoothed_hu: np.ndarray,
    lung_mask: np.ndarray,
    lung_mask_eroded: np.ndarray,
    spacing: np.ndarray,
    vessel_sigmas: tuple = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                            4.5, 5.0, 5.5, 7.0, 8.0),
    frangi_weight: float = 1.0,
    sato_weight: float = 0.9,
    jerman_weight: float = 1.0,
    noise_percentile: float = 40.0,
    **_: object,
) -> dict:
    """Compute hybrid vesselness map: Frangi + Sato + Jerman (weighted max fusion).

    Two-tier caching:
      - Individual filter maps cached by (sigmas, shape).
        Changing weights does NOT recompute filters.
      - Final fusion cached by (sigmas, weights, noise_percentile, shape).

    Args:
        cache:             CacheManager instance.
        smoothed_hu:       Anisotropic-diffusion smoothed HU float32 (Z,Y,X).
        lung_mask:         Full binary lung mask uint8 (Z,Y,X).
        lung_mask_eroded:  Eroded lung mask uint8 (Z,Y,X).
        spacing:           Voxel spacing [z,y,x] mm.
        vessel_sigmas:     Scales for multi-scale Hessian analysis [mm].
        frangi_weight:     Weight w_F in weighted max fusion.
        sato_weight:       Weight w_S in weighted max fusion.
        jerman_weight:     Weight w_J in weighted max fusion.
        noise_percentile:  Zero out nonzero values below this percentile on fused map.

    Returns:
        dict with v_frangi_clean, v_sato_clean, v_jerman_clean,
                  vesselness, vesselness_final.
    """
    fusion_params = {
        "vessel_sigmas":    list(vessel_sigmas),
        "frangi_weight":    frangi_weight,
        "sato_weight":      sato_weight,
        "jerman_weight":    jerman_weight,
        "noise_percentile": noise_percentile,
        "shape":            list(smoothed_hu.shape),
    }
    cached = cache.get("stage_03_vesselness", fusion_params)
    if cached is not None:
        return cached

    t0 = time.time()
    sigmas = list(vessel_sigmas)

    # Raw smoothed HU
    hu_input = smoothed_hu.astype(np.float64)
    active_mask = (lung_mask_eroded > 0)

    # Each filter cached independently
    filter_specs = {
        "Frangi":  ("stage_03_filter_frangi",  _run_frangi,  {},              frangi_weight),
        "Sato":    ("stage_03_filter_sato",    _run_sato,    {},              sato_weight),
        "Jerman":  ("stage_03_filter_jerman",  _run_jerman,  {"tau": 0.75},   jerman_weight),
    }

    scaled_maps: dict = {}
    to_compute = {}
    for name, (stage_name, fn, kwargs, weight) in filter_specs.items():
        if weight <= 0:
            scaled_maps[name] = np.zeros(hu_input.shape, dtype=np.float32)
            logger.info("Filter %s skipped (weight=0)", name)
            continue
        fparams = _filter_cache_params(sigmas, list(hu_input.shape))
        hit = cache.get(stage_name, fparams)
        if hit is not None:
            scaled_maps[name] = hit["map"]
            logger.info("Filter %s from cache  max=%.4f", name,
                        float(hit["map"].max()))
        else:
            to_compute[name] = (stage_name, fn, kwargs)

    if to_compute:
        logger.info("Computing %d filter(s): %s",
                     len(to_compute), ", ".join(to_compute))
        with ThreadPoolExecutor(max_workers=min(2, len(to_compute))) as executor:
            futures = {
                executor.submit(fn, hu_input, sigmas, **kwargs): name
                for name, (_stage, fn, kwargs) in to_compute.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                stage_name = to_compute[name][0]
                t_f = time.time()
                try:
                    raw = future.result()
                    scaled = _robust_scale(raw, active_mask)
                    scaled_maps[name] = scaled

                    logger.info(
                        "Filter %s done  (%.1fs)  max=%.4f",
                        name, time.time() - t_f, float(scaled.max()),
                    )

                    fparams = _filter_cache_params(sigmas, list(hu_input.shape))
                    cache.put(stage_name, fparams, {"map": scaled},
                              numpy_keys=["map"])
                except Exception as exc:
                    logger.error("Filter %s FAILED: %s", name, exc)
                    scaled_maps[name] = np.zeros(hu_input.shape,
                                                  dtype=np.float32)

    v_frangi_clean = scaled_maps["Frangi"]
    v_sato_clean   = scaled_maps["Sato"]
    v_jerman_clean = scaled_maps["Jerman"]

    logger.info("Per-filter robust scaling done  (%.1fs)", time.time() - t0)

    v_hybrid = np.maximum.reduce([
        frangi_weight  * v_frangi_clean,
        sato_weight    * v_sato_clean,
        jerman_weight  * v_jerman_clean,
    ])
    v_hybrid = np.clip(v_hybrid, 0.0, 1.0).astype(np.float32)
    v_hybrid = np.where(active_mask, v_hybrid, 0.0).astype(np.float32)
    vesselness = v_hybrid
    logger.info(
        "Weighted max fusion done  max=%.4f  (%.1fs)",
        float(vesselness.max()), time.time() - t0,
    )

    # percentile-based noise floor + normalisation.
    t_clean = time.time()
    v = vesselness.copy()

    nonzero = v[v > 0]
    if len(nonzero) > 0:
        p_floor = float(np.percentile(nonzero, noise_percentile))
        v[v < p_floor] = 0.0
        logger.info("Noise floor at p%d = %.5f", noise_percentile, p_floor)

    nonzero = v[v > 0]
    vmax = float(np.percentile(nonzero, 99.5)) if len(nonzero) > 0 else 1.0
    vesselness_final = np.clip(v / max(vmax, 1e-9), 0.0, 1.0).astype(
        np.float32)

    logger.info("Percentile normalisation done  (%.1fs)", time.time() - t_clean)

    result = {
        "v_frangi_clean":   v_frangi_clean,
        "v_sato_clean":     v_sato_clean,
        "v_jerman_clean":   v_jerman_clean,
        "vesselness":       vesselness,
        "vesselness_final": vesselness_final,
    }
    cache.put(
        "stage_03_vesselness",
        fusion_params,
        result,
        numpy_keys=list(result.keys()),
    )
    logger.info("Stage 03 done  total=%.1fs", time.time() - t0)
    return result
