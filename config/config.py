"""Pipeline configuration dataclass."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Paths
    dicom_dir    : str  = r"C:\Users\nemo\Downloads\Z76102888-20260111T115908Z-1-001\Z76102888"
    cache_dir    : str  = "cache"
    output_dir   : str  = "outputs"

    # Stage 0 - DICOM load
    iso_spacing  : tuple = (1.0, 1.0, 1.0)
    dicom_series : str   = "Z76102888_3"

    # Stage 1 - Lung segmentation
    body_thresh          : float = -600.0
    lung_low             : float = -1000.0
    lung_high            : float = -400.0
    airway_break_radius  : int   = 2
    close_radius         : int   = 2
    use_refine           : bool  = True
    refine_target_spacing: float = 2.5 #2.5 #1
    refine_ac_iters      : int   = 35 #35 #500
    lung_wall_erosion_mm : float = 3.0   # erode lung mask by N mm to exclude pleural wall

    # Stage 2 - Anisotropic diffusion
    diff_n_iter  : int   = 5
    diff_kappa   : float = 50.0
    diff_gamma   : float = 0.1

    # Stage 3 - Vesselness
    vessel_sigmas        : tuple = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                                    4.5, 5.0, 5.5, 7.0, 8.0)
    hu_clip_low          : float = -200.0
    hu_clip_high         : float =  300.0
    frangi_weight        : float = 1.0 #1.0
    sato_weight          : float = 1.2 #1.2
    meijering_weight     : float = 0.0
    jerman_weight        : float = 0
    noise_percentile     : float = 30.0    # zero out nonzero values below this percentile on fused map

    # Stage 4 - Skeleton extraction (TEASAR via kimimaro)
    vesselness_threshold         : float = 0.15
    min_lung_voxels              : int   = 1000
    min_skeleton_fragment        : int   = 5
    teasar_scale                 : float = 1.5    # invalidation radius = scale * DBF + const
    teasar_const                 : float = 0.5    # minimum invalidation radius (mm)
    teasar_pdrf_scale            : int   = 5000   # weight of DBF in path cost
    teasar_pdrf_exponent         : int   = 4      # exponent on DBF distance

    # Stage 5 - Graph construction
    min_spur_length_mm  : float = 2.0
    min_component_nodes : int   = 5

    # Stage 6 - Morphometry
    fractal_scales     : list  = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])

    # Napari viewer
    show_napari        : bool  = True
    napari_3d          : bool  = True
    napari_after_stage : int   = -1   # -1 = only at end, 0-6 = after that stage

    # PyVista 3D viewer
    show_pyvista       : bool  = True
