"""Stage 00 - CT DICOM Series loading and isotropic resampling.

Input:  folder with .dcm files
Output: dict with keys:
    hu_iso      : np.int16  (Z, Y, X)  - HU volume resampled to iso_spacing
    spacing     : np.float32 (3,)      - original spacing [z, y, x] in mm
    spacing_iso : np.float32 (3,)      - isotropic spacing (1, 1, 1) mm
"""
import logging
import time
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
from scipy.ndimage import zoom

from modules.cache_manager import CacheManager

logger = logging.getLogger(__name__)


def _select_series(
    all_slices: list,
    dicom_series: str,
) -> list:
    """Group slices by SeriesInstanceUID and select one series.

    Args:
        all_slices:   List of pydicom Dataset objects with pixel_array.
        dicom_series: SeriesDescription string or SeriesInstanceUID substring.
                      If empty and only one series exists, that series is used.

    Returns:
        Filtered list of slices belonging to the selected series.

    Raises:
        ValueError: If dicom_series is empty with multiple series found,
                    or if the specified series is not found.
    """
    series_map: dict = {}
    for ds in all_slices:
        sid = getattr(ds, "SeriesInstanceUID", None)
        if sid is None:
            continue
        series_map.setdefault(str(sid), []).append(ds)

    if not series_map:
        logger.warning("No SeriesInstanceUID found - using all %d slices.", len(all_slices))
        return all_slices

    logger.info("Discovered %d DICOM series:", len(series_map))
    series_summary = []
    for sid, ds_list in series_map.items():
        sample = ds_list[0]
        desc = getattr(sample, "SeriesDescription", "N/A")
        thick = float(getattr(sample, "SliceThickness", 999))
        kernel = str(getattr(sample, "ConvolutionKernel", "unknown"))
        count = len(ds_list)
        logger.info(
            "  [%s] %d slices, %.1f mm, kernel=%s  (UID=...%s)",
            desc, count, thick, kernel, sid[-20:],
        )
        series_summary.append({
            "uid": sid, "desc": desc, "count": count,
            "thick": thick, "kernel": kernel,
        })

    if dicom_series:
        for info in series_summary:
            if dicom_series == info["desc"] or dicom_series in info["uid"]:
                selected = series_map[info["uid"]]
                logger.info(
                    "Selected series '%s' by match '%s': %d slices",
                    info["desc"], dicom_series, len(selected),
                )
                return selected
        available = "\n".join(
            f"  '{s['desc']}' - {s['count']} slices, {s['thick']:.1f} mm  (UID=...{s['uid'][-20:]})"
            for s in series_summary
        )
        raise ValueError(
            f"Series '{dicom_series}' not found.\nAvailable series:\n{available}"
        )

    if len(series_map) == 1:
        only_uid = next(iter(series_map))
        selected = series_map[only_uid]
        logger.info("Single series found, using it: %d slices", len(selected))
        return selected

    available = "\n".join(
        f"  '{s['desc']}' - {s['count']} slices, {s['thick']:.1f} mm  (UID=...{s['uid'][-20:]})"
        for s in series_summary
    )
    raise ValueError(
        f"Multiple DICOM series found. Specify dicom_series in config:\n{available}"
    )


def run(
    cache: CacheManager,
    dicom_dir: str,
    iso_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    dicom_series: str = "",
) -> dict:
    """Load DICOM series, convert to HU, resample to isotropic spacing.

    Args:
        cache:        CacheManager instance.
        dicom_dir:    Path to folder containing .dcm files.
        iso_spacing:  Target isotropic spacing in mm (z, y, x).
        dicom_series: SeriesDescription string
    Returns:
        dict with hu_iso, spacing, spacing_iso.
    """
    params = {
        "dicom_dir": str(dicom_dir),
        "iso_spacing": list(iso_spacing),
        "dicom_series": dicom_series,
    }
    cached = cache.get("stage_00_dicom_load", params)
    if cached is not None:
        return cached

    t0 = time.time()
    dicom_path = Path(dicom_dir)

    dcm_files = []
    if dicom_path.is_dir():
        for f in os.listdir(dicom_dir):
            if f.lower().endswith(".dcm") or "." not in f:
                full_path = dicom_path / f
                if full_path.is_file():
                    dcm_files.append(full_path)

    if not dcm_files:
        logger.info("Brak plików bezpośrednio w %s, szukam w podfolderach...", dicom_dir)
        for p in dicom_path.rglob("*"):
            if p.is_file() and (p.suffix.lower() == ".dcm" or "." not in p.name):
                dcm_files.append(p)

    dcm_files = sorted(dcm_files)

    if not dcm_files:
        raise FileNotFoundError(f"Brak plików DICOM w folderze: {dicom_dir}")
    logger.info("Found %d DICOM files in %s", len(dcm_files), dicom_dir)

    all_slices = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if hasattr(ds, "pixel_array"):
                all_slices.append(ds)
        except Exception:
            pass

    if not all_slices:
        raise RuntimeError("Nie udało się wczytać żadnych poprawnych warstw DICOM.")
    logger.info("Wczytano %d poprawnych warstw (%.1fs)", len(all_slices), time.time() - t0)

    slices = _select_series(all_slices, dicom_series)

    def _sort_key(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            try:
                return int(ds.InstanceNumber)
            except Exception:
                return 0.0

    slices.sort(key=_sort_key)

    t1 = time.time()
    hu_slices = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        hu_slices.append(arr)

    hu_volume = np.stack(hu_slices, axis=0).astype(np.int16)
    logger.info("HU volume shape=%s  (%.1fs)", hu_volume.shape, time.time() - t1)

    try:
        positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
        diffs = np.diff(positions)
        z_spacing = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 1.0
    except Exception:
        z_spacing = float(getattr(slices[0], "SliceThickness", 1.0))

    try:
        pix_spacing = slices[0].PixelSpacing
        y_spacing   = float(pix_spacing[0])
        x_spacing   = float(pix_spacing[1])
    except Exception:
        y_spacing = x_spacing = 1.0

    spacing_orig = np.array([z_spacing, y_spacing, x_spacing], dtype=np.float32)
    spacing_iso_ = np.array(iso_spacing, dtype=np.float32)
    logger.info("Original spacing [z,y,x] = %s mm", spacing_orig)

    t2 = time.time()
    zoom_factors = (spacing_orig / spacing_iso_).tolist()
    logger.info("Resampling zoom factors: %s", [f"{z:.3f}" for z in zoom_factors])

    hu_iso = zoom(
        hu_volume.astype(np.float32),
        zoom=zoom_factors,
        order=1,        # trilinear
        prefilter=False,
    ).astype(np.int16)
    logger.info(
        "Resampled to %s  iso_spacing=%s  (%.1fs)",
        hu_iso.shape, spacing_iso_, time.time() - t2,
    )

    result = {
        "hu_iso":      hu_iso,
        "spacing":     spacing_orig,
        "spacing_iso": spacing_iso_,
    }

    cache.put(
        "stage_00_dicom_load",
        params,
        result,
        numpy_keys=["hu_iso", "spacing", "spacing_iso"],
    )
    logger.info("Stage 00 done  total=%.1fs", time.time() - t0)
    return result
