import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from config.config import PipelineConfig
from modules.cache_manager import CacheManager
from modules.napari_viewer import PipelineViewer
from modules.pyvista_viewer import show_graph_vs_vesselness

import modules.stage_00_dicom_load as s0
import modules.stage_01_lung_segmentation as s1
import modules.stage_02_anisotropic_diffusion as s2
import modules.stage_03_vesselness as s3
import modules.stage_04_vessel_graph as s4
import modules.stage_05_graph_construction as s5
import modules.stage_06_morphometric_analysis as s6

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PipelineMain")


def save_nifti(data: np.ndarray, spacing: np.ndarray, out_path: Path) -> None:
    aff = np.diag([float(spacing[2]), float(spacing[1]), float(spacing[0]), 1.0])  # [z,y,x] -> NIfTI diagonal affine
    img = nib.Nifti1Image(data, aff)
    nib.save(img, str(out_path))
    logger.info("Zapisano NIfTI: %s", out_path)


def main(cfg: PipelineConfig | None = None):
    t_start = time.time()

    if cfg is None:
        cfg = PipelineConfig()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Rozpoczynam analizę drzewa naczyniowego płuc.")

    cache = CacheManager(cache_dir=cfg.cache_dir)
    cache.print_status()

    viewer = PipelineViewer(spacing=cfg.iso_spacing, three_d=cfg.napari_3d)

    try:
        res0 = s0.run(
            cache=cache,
            dicom_dir=cfg.dicom_dir,
            iso_spacing=cfg.iso_spacing,
            dicom_series=cfg.dicom_series,
        )
        sp = res0["spacing_iso"]

        save_nifti(res0["hu_iso"].astype(np.float32), sp, out / "s00_hu_iso.nii.gz")

        if cfg.show_napari and cfg.napari_after_stage in (-1, 0):
            viewer.stage0_raw_ct(res0["hu_iso"], sp)
            if cfg.napari_after_stage == 0:
                viewer.show()

        res1 = s1.run(
            cache=cache,
            hu_iso=res0["hu_iso"],
            spacing=sp,
            body_thresh=cfg.body_thresh,
            lung_low=cfg.lung_low,
            lung_high=cfg.lung_high,
            airway_break_radius=cfg.airway_break_radius,
            close_radius=cfg.close_radius,
            use_refine=cfg.use_refine,
            refine_target_spacing=cfg.refine_target_spacing,
            refine_ac_iters=cfg.refine_ac_iters
        )

        save_nifti(res1["lung_mask"], sp, out / "s01_lung_mask.nii.gz")
        save_nifti(res1["lung_mask_eroded"], sp, out / "s01_lung_mask_eroded.nii.gz")

        if cfg.show_napari and cfg.napari_after_stage in (-1, 1):
            viewer.stage1_segmentation(res0["hu_iso"], res1, sp)
            if cfg.napari_after_stage == 1:
                viewer.show()

        res2 = s2.run(
            cache=cache,
            hu_iso=res0["hu_iso"],
            lung_mask=res1["lung_mask"],
            spacing=sp,
            diff_n_iter=cfg.diff_n_iter,
            diff_kappa=cfg.diff_kappa,
            diff_gamma=cfg.diff_gamma,
        )

        save_nifti(res2["smoothed_hu"], sp, out / "s02_smoothed_hu.nii.gz")

        if cfg.show_napari and cfg.napari_after_stage in (-1, 2):
            viewer.stage2_diffusion(res0["hu_iso"], res2["smoothed_hu"], res1["lung_mask"], sp)
            if cfg.napari_after_stage == 2:
                viewer.show()

        res3 = s3.run(
            cache=cache,
            smoothed_hu=res2["smoothed_hu"],
            lung_mask=res1["lung_mask"],
            lung_mask_eroded=res1["lung_mask_eroded"],
            spacing=sp,
            vessel_sigmas=cfg.vessel_sigmas,
            frangi_weight=cfg.frangi_weight,
            sato_weight=cfg.sato_weight,
            jerman_weight=cfg.jerman_weight,
            noise_percentile=cfg.noise_percentile,
        )

        save_nifti(res3["vesselness_final"], sp, out / "s03_vesselness_final.nii.gz")
        save_nifti(res3["vesselness"], sp, out / "s03_vesselness_fused.nii.gz")
        save_nifti(res3["v_frangi_clean"], sp, out / "s03_map_frangi.nii.gz")
        save_nifti(res3["v_sato_clean"], sp, out / "s03_map_sato.nii.gz")
        save_nifti(res3["v_jerman_clean"], sp, out / "s03_map_jerman.nii.gz")

        vessel_mask = (res3["vesselness_final"] > 0).astype(np.uint8)
        save_nifti(vessel_mask, sp, out / "s03_vessel_mask.nii.gz")

        if cfg.show_napari and cfg.napari_after_stage in (-1, 3):
            viewer.stage3_vesselness(
                res0["hu_iso"], res3["vesselness_final"],
                res1["lung_mask"], sp,
            )
            if cfg.napari_after_stage == 3:
                viewer.show()

        res4 = s4.run(
            cache=cache,
            vesselness_final=res3["vesselness_final"],
            spacing=sp,
            vesselness_threshold=cfg.vesselness_threshold,
            min_lung_voxels=cfg.min_lung_voxels,
            min_skeleton_fragment=cfg.min_skeleton_fragment,
            teasar_scale=cfg.teasar_scale,
            teasar_const=cfg.teasar_const,
            teasar_pdrf_scale=cfg.teasar_pdrf_scale,
            teasar_pdrf_exponent=cfg.teasar_pdrf_exponent,
        )

        save_nifti(res4["skeleton_left"], sp, out / "s04_skeleton_left.nii.gz")
        save_nifti(res4["skeleton_right"], sp, out / "s04_skeleton_right.nii.gz")
        save_nifti(res4["skeleton_combined"], sp, out / "s04_skeleton_combined.nii.gz")
        save_nifti(res4["edt_left"], sp, out / "s04_edt_left.nii.gz")
        save_nifti(res4["edt_right"], sp, out / "s04_edt_right.nii.gz")
        save_nifti(res4["left_mask"], sp, out / "s04_left_mask.nii.gz")
        save_nifti(res4["right_mask"], sp, out / "s04_right_mask.nii.gz")

        if cfg.show_napari and cfg.napari_after_stage in (-1, 4):
            viewer.stage4_skeleton(
                res0["hu_iso"],
                res4["skeleton_left"], res4["skeleton_right"],
                res4["edt_left"], res4["edt_right"],
                res4["left_mask"], res4["right_mask"],
                sp,
            )
            if cfg.napari_after_stage == 4:
                viewer.show()

        kim_left = kim_right = None
        if "kimimaro_left_vertices" in res4:
            kim_left = {
                "vertices": res4["kimimaro_left_vertices"],
                "edges":    res4["kimimaro_left_edges"],
                "radii":    res4["kimimaro_left_radii"],
            }
        if "kimimaro_right_vertices" in res4:
            kim_right = {
                "vertices": res4["kimimaro_right_vertices"],
                "edges":    res4["kimimaro_right_edges"],
                "radii":    res4["kimimaro_right_radii"],
            }

        res5 = s5.run(
            cache=cache,
            skeleton_left=res4["skeleton_left"],
            skeleton_right=res4["skeleton_right"],
            spacing=sp,
            min_spur_length_mm=cfg.min_spur_length_mm,
            min_component_nodes=cfg.min_component_nodes,
            kimimaro_left=kim_left,
            kimimaro_right=kim_right,
        )

        if cfg.show_napari and cfg.napari_after_stage in (-1, 5):
            viewer.stage5_graph(
                res0["hu_iso"],
                res5["graph_left"], res5["graph_right"],
                sp,
            )
            if cfg.napari_after_stage == 5:
                viewer.show()

        res6 = s6.run(
            cache=cache,
            graph_left=res5["graph_left"],
            graph_right=res5["graph_right"],
            edt_left=res4["edt_left"],
            edt_right=res4["edt_right"],
            spacing=sp,
            output_dir=cfg.output_dir,
            fractal_scales=cfg.fractal_scales
        )

        # stage_06 cache may skip the mutation, so re-apply edge radii here
        from modules.stage_06_morphometric_analysis import _update_edge_radii
        _update_edge_radii(res5["graph_left"],  res4["edt_left"])
        _update_edge_radii(res5["graph_right"], res4["edt_right"])

        if cfg.show_pyvista:
            show_graph_vs_vesselness(
                graph_left=res5["graph_left"],
                graph_right=res5["graph_right"],
                vesselness=res3["vesselness_final"],
                spacing=sp,
            )

        if cfg.show_napari and cfg.napari_after_stage in (-1, 6):
            strahler_left = {
                n: d.get("strahler", 1) for n, d in res5["graph_left"].nodes(data=True)
            }
            strahler_right = {
                n: d.get("strahler", 1) for n, d in res5["graph_right"].nodes(data=True)
            }
            viewer.stage6_morphometry(
                res0["hu_iso"], res5["graph_left"], res5["graph_right"],
                strahler_left, strahler_right, sp,
            )
            if cfg.napari_after_stage == 6 or cfg.napari_after_stage == -1:
                viewer.show()

        logger.info(
            "Pipeline zakończony sukcesem! Całkowity czas: %.1fs",
            time.time() - t_start,
        )
        cache.print_status()

    except Exception as e:
        logger.error("Krytyczny błąd podczas wykonywania pipeline'u: %s", e, exc_info=True)


if __name__ == '__main__':
    main()
