"""eyesim-py: Analysis of eye-movement data."""

from peyesim.fixations import (
    FixationGroup,
    fixation_group,
    concat_fixation_groups,
    coords,
    center,
    normalize,
    rescale,
    rep_fixations,
    sample_fixations,
)
from peyesim.eye_table import EyeTable, eye_table, as_eye_table, simulate_eye_table
from peyesim.saccades import Scanpath, scanpath, cart2pol, calcangle, add_scanpath
from peyesim.density import (
    eye_density,
    gen_density,
    get_density,
    density_matrix,
    density_by,
    suggest_sigma,
    EyeDensity,
    EyeDensityMultiscale,
)
from peyesim.similarity import (
    similarity,
    compute_similarity,
    template_similarity,
    fixation_similarity,
    scanpath_similarity,
    sample_density,
    sample_density_time,
)
from peyesim.overlap import fixation_overlap
from peyesim.multimatch import multi_match, install_multimatch
from peyesim.regression import template_regression, template_multireg, template_sample
from peyesim.latent_transforms import (
    latent_pca_transform,
    coral_transform,
    cca_transform,
)
from peyesim.repetitive_similarity import repetitive_similarity
from peyesim.visualization import anim_scanpath

__all__ = [
    "FixationGroup",
    "fixation_group",
    "concat_fixation_groups",
    "coords",
    "center",
    "normalize",
    "rescale",
    "rep_fixations",
    "sample_fixations",
    "EyeTable",
    "eye_table",
    "as_eye_table",
    "simulate_eye_table",
    "Scanpath",
    "scanpath",
    "cart2pol",
    "calcangle",
    "add_scanpath",
    "anim_scanpath",
    "eye_density",
    "gen_density",
    "get_density",
    "density_matrix",
    "density_by",
    "suggest_sigma",
    "EyeDensity",
    "EyeDensityMultiscale",
    "similarity",
    "compute_similarity",
    "template_similarity",
    "fixation_similarity",
    "scanpath_similarity",
    "sample_density",
    "sample_density_time",
    "fixation_overlap",
    "multi_match",
    "install_multimatch",
    "template_regression",
    "template_multireg",
    "template_sample",
    "latent_pca_transform",
    "coral_transform",
    "cca_transform",
    "repetitive_similarity",
]
