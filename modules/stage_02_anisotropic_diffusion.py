"""Stage 02 - Anisotropic diffusion (Perona-Malik 3D) on HU volume.

Option 1 (default): c = exp(-(∇I/κ)²)
Option 2:           c = 1 / (1 + (∇I/κ)²)  ;) for future

Gradients computed in 6 directions (+-z, +-y, +-x).

Input:
    hu_iso    : np.int16  (Z, Y, X)
    lung_mask : np.uint8  (Z, Y, X)
    spacing   : np.float32 (3,)

Output (cache keys):
    smoothed_hu : np.float32 (Z, Y, X) - diffused volume inside lung_mask
"""
import logging
import time

import numpy as np

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def run(
    cache: CacheManager,
    hu_iso: np.ndarray,
    lung_mask: np.ndarray,
    spacing: np.ndarray,
    diff_n_iter: int = 5,
    diff_kappa: float = 50.0,
    diff_gamma: float = 0.1,
    option: int = 1,
    **_: object,
) -> dict:
    """Perona-Malik anisotropic diffusion.

    Args:
        cache:       CacheManager instance.
        hu_iso:      HU volume (Z, Y, X) int16.
        lung_mask:   Binary lung mask (Z, Y, X) uint8.
        spacing:     Voxel spacing [z, y, x] mm.
        diff_n_iter: Number of diffusion iterations.
        diff_kappa:  Edge-sensitivity parameter κ [HU].
        diff_gamma:  Integration constant (step size, typically 0.1).
        option:      Conductance function: 1=Gaussian, 2=Lorentzian.

    Returns:
        dict with smoothed_hu (float32)
    """
    params = {
        "n_iter":  diff_n_iter,
        "kappa":   diff_kappa,
        "gamma":   diff_gamma,
        "option":  option,
        "shape":   list(hu_iso.shape),
    }
    cached = cache.get("stage_02_anisotropic_diffusion", params)
    if cached is not None:
        return cached

    t0 = time.time()
    img = hu_iso.astype(np.float32)
    mask = (lung_mask > 0)

    img_work = img.copy()

    for iteration in range(diff_n_iter):
        t_iter = time.time()

        dN = np.zeros_like(img_work)
        dS = np.zeros_like(img_work)
        dN[1:,  :, :]  = img_work[:-1, :, :] - img_work[1:,  :, :]
        dS[:-1, :, :]  = img_work[1:,  :, :] - img_work[:-1, :, :]

        dE = np.zeros_like(img_work)
        dW = np.zeros_like(img_work)
        dE[:, :-1, :]  = img_work[:, 1:,  :] - img_work[:, :-1, :]
        dW[:, 1:,  :]  = img_work[:, :-1, :] - img_work[:, 1:,  :]

        dF = np.zeros_like(img_work)
        dB = np.zeros_like(img_work)
        dF[:, :, :-1]  = img_work[:, :, 1:]  - img_work[:, :, :-1]
        dB[:, :, 1:]   = img_work[:, :, :-1] - img_work[:, :, 1:]

        if option == 1:
            def _c(d): return np.exp(-((d / diff_kappa) ** 2))
        else:
            def _c(d): return 1.0 / (1.0 + (d / diff_kappa) ** 2)

        cN = _c(dN)
        cS = _c(dS)
        cE = _c(dE)
        cW = _c(dW)
        cF = _c(dF)
        cB = _c(dB)

        update = diff_gamma * (
            cN * dN + cS * dS +
            cE * dE + cW * dW +
            cF * dF + cB * dB
        )
        img_work = np.where(mask, img_work + update, img_work)

        logger.info(
            "Diffusion iter %d/%d  (%.1fs)", iteration + 1, diff_n_iter, time.time() - t_iter
        )

    smoothed_hu = np.where(mask, img_work, 0.0).astype(np.float32)

    result = {"smoothed_hu": smoothed_hu}
    cache.put(
        "stage_02_anisotropic_diffusion",
        params,
        result,
        numpy_keys=["smoothed_hu"],
    )
    logger.info("Stage 02 done  total=%.1fs", time.time() - t0)
    return result
