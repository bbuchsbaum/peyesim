"""Compare selected peyesim results against the local R eyesim package.

This is intentionally small and deterministic.  It is not a replacement for the
Python test suite; it is an oracle smoke test for core contracts whose semantics
come directly from ``~/code/eyesim``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from peyesim import (
    calcangle,
    cart2pol,
    center,
    fixation_entropy,
    fixation_group,
    fixation_overlap,
    normalize,
    rescale,
    sample_fixations,
    sample_density,
    scanpath,
    similarity,
    template_similarity,
    template_similarity_cv,
    template_multireg,
    template_regression,
    eye_density,
    latent_pca_transform,
    coral_transform,
    cca_transform,
    contract_transform,
    affine_transform,
    crqa,
)
from peyesim.density import EyeDensity, gen_density


REPO_ROOT = Path(__file__).resolve().parents[1]


R_CODE = r"""
suppressPackageStartupMessages(pkgload::load_all("eyesim", quiet = TRUE))
suppressPackageStartupMessages(library(jsonlite))

fg <- fixation_group(
  x = c(0, 1, 3, 4),
  y = c(0, 2, 2, 5),
  onset = c(0, 100, 200, 350),
  duration = c(100, 100, 150, 50)
)
fg_overlap_ref <- fixation_group(
  x = c(0, 10, 20),
  y = c(0, 0, 0),
  onset = c(0, 100, 200),
  duration = c(100, 100, 100)
)
fg_overlap_shift <- fixation_group(
  x = c(2, 15, 40),
  y = c(1, 0, 0),
  onset = c(0, 100, 200),
  duration = c(100, 100, 100)
)

make_entropy_density <- function(z) {
  structure(
    list(z = z, x = seq_len(nrow(z)), y = seq_len(ncol(z)), sigma = 1),
    class = c("eye_density", "density", "list")
  )
}

density_a <- gen_density(
  x = c(0, 1, 2),
  y = c(0, 1, 2),
  z = matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow = 3, byrow = TRUE)
)
density_b <- gen_density(
  x = c(0, 1, 2),
  y = c(0, 1, 2),
  z = matrix(c(9, 7, 6, 5, 4, 3, 2, 1, 8), nrow = 3, byrow = TRUE)
)
density_fg <- fixation_group(
  x = c(0, 1, 2),
  y = c(0, 1, 0),
  onset = c(0, 100, 200),
  duration = c(1, 2, 1)
)
density_unweighted <- eye_density(
  density_fg,
  sigma = 1.2,
  xbounds = c(0, 2),
  ybounds = c(0, 2),
  outdim = c(4, 4),
  normalize = FALSE,
  duration_weighted = FALSE
)
density_weighted <- eye_density(
  density_fg,
  sigma = 1.2,
  xbounds = c(0, 2),
  ybounds = c(0, 2),
  outdim = c(4, 4),
  normalize = FALSE,
  duration_weighted = TRUE
)
has_crqa <- requireNamespace("crqa", quietly = TRUE)
crqa_result <- if (has_crqa) {
  suppressMessages(crqa(
    data.frame(x = 1:5, y = 1:5),
    data.frame(x = 2:6, y = 3:7),
    radius = 60
  ))
} else {
  NULL
}
template_ref <- data.frame(id = c("A", "B"), stringsAsFactors = FALSE)
template_ref$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(4, 3, 2, 1), nrow = 2, byrow = TRUE))
)
template_src <- data.frame(id = c("A", "B"), stringsAsFactors = FALSE)
template_src$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 1, 2, 2), nrow = 2, byrow = TRUE))
)
template_result <- suppressMessages(template_similarity(
  template_ref,
  template_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "pearson",
  permutations = 0
))
template_perm_ref <- data.frame(id = c("A", "B", "C"), stringsAsFactors = FALSE)
template_perm_ref$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(4, 3, 2, 1), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 3, 2, 4), nrow = 2, byrow = TRUE))
)
template_perm_src <- data.frame(id = c("A", "B", "C"), stringsAsFactors = FALSE)
template_perm_src$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 1, 2, 2), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(4, 2, 3, 1), nrow = 2, byrow = TRUE))
)
template_perm_result <- suppressMessages(template_similarity(
  template_perm_ref,
  template_perm_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "pearson",
  permutations = 10
))
transform_ref <- data.frame(
  id = c("A", "B", "C", "D"),
  pid = "p1",
  stringsAsFactors = FALSE
)
transform_ref$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(0.2, 1.0, 0.5, 1.3), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1.5, 0.7, 0.2, 0.9), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(0.4, 0.1, 1.7, 1.1), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1.2, 1.6, 0.3, 0.5), nrow = 2, byrow = TRUE))
)
transform_src <- data.frame(
  id = c("A", "B", "C", "D"),
  pid = "p1",
  fold_group = c("G1", "G2", "G1", "G2"),
  stringsAsFactors = FALSE
)
transform_src$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(0.3, 1.2, 0.6, 1.1), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1.4, 0.5, 0.4, 1.0), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(0.2, 0.2, 1.9, 1.0), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(1.0, 1.7, 0.5, 0.4), nrow = 2, byrow = TRUE))
)
template_pca_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = latent_pca_transform,
  similarity_transform_args = list(comps = 3)
))
template_coral_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = coral_transform,
  similarity_transform_args = list(comps = 3, shrink = 1e-3)
))
template_coral_fit_by_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = coral_transform,
  similarity_transform_args = list(comps = 3, shrink = 1e-3, fit_by = "pid")
))
template_cca_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = cca_transform,
  similarity_transform_args = list(comps = 3, shrink = 1e-3)
))
template_cca_fit_by_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = cca_transform,
  similarity_transform_args = list(comps = 3, shrink = 1e-3, fit_by = "pid")
))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
template_cv_result <- suppressMessages(template_similarity_cv(
  transform_ref,
  transform_src,
  match_on = "id",
  split_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  n_folds = 2,
  seed = 3
))
template_cv_result_frame <- as.data.frame(template_cv_result[, c("id", "eye_sim", ".cv_fold")])
template_contract_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = contract_transform,
  similarity_transform_args = list(shrink = 1e-6)
))
template_contract_fit_by_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = contract_transform,
  similarity_transform_args = list(shrink = 1e-6, fit_by = "pid")
))
template_affine_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = affine_transform,
  similarity_transform_args = list(shrink = 1e-6)
))
template_affine_fit_by_result <- suppressMessages(template_similarity(
  transform_ref,
  transform_src,
  match_on = "id",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  similarity_transform = affine_transform,
  similarity_transform_args = list(shrink = 1e-6, fit_by = "pid")
))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
template_cv_contract_result <- suppressMessages(template_similarity_cv(
  transform_ref,
  transform_src,
  match_on = "id",
  split_on = "fold_group",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  n_folds = 2,
  seed = 1,
  similarity_transform = contract_transform,
  similarity_transform_args = list(shrink = 1e-6)
))
template_cv_contract_result_frame <- as.data.frame(template_cv_contract_result[, c("id", "eye_sim", ".cv_fold")])
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
template_cv_affine_result <- suppressMessages(template_similarity_cv(
  transform_ref,
  transform_src,
  match_on = "id",
  split_on = "fold_group",
  refvar = "density",
  sourcevar = "density",
  method = "cosine",
  permutations = 0,
  n_folds = 2,
  seed = 1,
  similarity_transform = affine_transform,
  similarity_transform_args = list(shrink = 1e-6)
))
template_cv_affine_result_frame <- as.data.frame(template_cv_affine_result[, c("id", "eye_sim", ".cv_fold")])
reg_baseline <- data.frame(scene = c("S1", "S2"), stringsAsFactors = FALSE)
reg_baseline$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(0, 1, 1, 0), nrow = 2, byrow = TRUE))
)
reg_ref <- data.frame(id = c("A", "B"), stringsAsFactors = FALSE)
reg_ref$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(0, 1, 2, 3), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(3, 2, 1, 0), nrow = 2, byrow = TRUE))
)
reg_source <- data.frame(id = c("A", "B"), scene = c("S1", "S2"), stringsAsFactors = FALSE)
reg_source$density <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 5), nrow = 2, byrow = TRUE)),
  gen_density(c(0, 1), c(0, 1), matrix(c(5, 3, 2, 1), nrow = 2, byrow = TRUE))
)
regression_result <- suppressMessages(template_regression(
  reg_ref,
  reg_source,
  match_on = "id",
  baseline_tab = reg_baseline,
  baseline_key = "scene",
  method = "lm"
))
regression_rlm_result <- suppressMessages(template_regression(
  reg_ref,
  reg_source,
  match_on = "id",
  baseline_tab = reg_baseline,
  baseline_key = "scene",
  method = "rlm"
))
has_ppcor <- requireNamespace("ppcor", quietly = TRUE)
regression_rank_result <- if (has_ppcor) {
  suppressMessages(template_regression(
    reg_ref,
    reg_source,
    match_on = "id",
    baseline_tab = reg_baseline,
    baseline_key = "scene",
    method = "rank"
  ))
} else {
  NULL
}
multireg_tab <- data.frame(row = 1L)
multireg_tab$response <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 2, 3, 4), nrow = 2, byrow = TRUE))
)
multireg_tab$cov1 <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(1, 0, 0, 1), nrow = 2, byrow = TRUE))
)
multireg_tab$cov2 <- list(
  gen_density(c(0, 1), c(0, 1), matrix(c(0, 1, 0, 1), nrow = 2, byrow = TRUE))
)
multireg_result <- suppressMessages(template_multireg(
  multireg_tab,
  response = "response",
  covars = c("cov1", "cov2"),
  method = "lm",
  intercept = TRUE
))
multireg_logistic_result <- suppressWarnings(suppressMessages(template_multireg(
  multireg_tab,
  response = "response",
  covars = c("cov1", "cov2"),
  method = "logistic",
  intercept = TRUE
)))
multireg_rlm_result <- suppressMessages(template_multireg(
  multireg_tab,
  response = "response",
  covars = c("cov1", "cov2"),
  method = "rlm",
  intercept = TRUE
))
has_nnls <- requireNamespace("nnls", quietly = TRUE)
multireg_nnls_result <- if (has_nnls) {
  suppressMessages(template_multireg(
    multireg_tab,
    response = "response",
    covars = c("cov1", "cov2"),
    method = "nnls",
    intercept = TRUE
  ))
} else {
  NULL
}

out <- list(
  coords = coords(fg),
  centered = as.data.frame(center(fg)),
  normalized = as.data.frame(normalize(fg, xbounds = c(0, 4), ybounds = c(0, 5))),
  rescaled = as.data.frame(rescale(fg, sx = 2, sy = 0.5)),
  sampled = as.data.frame(sample_fixations(fg, time = c(-10, 0, 50, 100, 250), fast = TRUE)),
  overlap_euclidean = fixation_overlap(
    fg_overlap_ref,
    fg_overlap_shift,
    dthresh = 6,
    time_samples = c(0, 100, 200),
    dist_method = "euclidean"
  ),
  overlap_manhattan = fixation_overlap(
    fg_overlap_ref,
    fg_overlap_shift,
    dthresh = 6,
    time_samples = c(0, 100, 200),
    dist_method = "manhattan"
  ),
  cart2pol = cart2pol(c(1, 2, 0), c(2, 0, -3)),
  calcangle = calcangle(c(1, 0), c(0, 1)),
  scanpath = as.data.frame(scanpath(fg)),
  entropy_point = fixation_entropy(
    make_entropy_density(matrix(c(1, 0, 0, 0), nrow = 2, byrow = TRUE)),
    normalize = FALSE
  ),
  entropy_uniform_norm = fixation_entropy(make_entropy_density(matrix(1, 2, 2)), normalize = TRUE),
  entropy_grid = fixation_entropy(
    fg,
    method = "grid",
    grid = c(2, 2),
    xbounds = c(0, 4),
    ybounds = c(0, 5),
    normalize = FALSE,
    base = 2
  ),
  sample_density_none = as.data.frame(sample_density(density_a, fg)),
  sample_density_sum = as.data.frame(sample_density(density_a, fg, normalize = "sum")),
  sample_density_zscore = as.data.frame(sample_density(density_a, fg, normalize = "zscore")),
  density_similarity = list(
    pearson = similarity(density_a, density_b, method = "pearson"),
    spearman = similarity(density_a, density_b, method = "spearman"),
    fisherz = similarity(density_a, density_b, method = "fisherz"),
    cosine = similarity(density_a, density_b, method = "cosine"),
    l1 = similarity(density_a, density_b, method = "l1")
  ),
  eye_density_unweighted = list(x = density_unweighted$x, y = density_unweighted$y, z = density_unweighted$z),
  eye_density_weighted = list(x = density_weighted$x, y = density_weighted$y, z = density_weighted$z),
  has_crqa = has_crqa,
  crqa = if (has_crqa) {
    list(
      RR = crqa_result$RR,
      DET = crqa_result$DET,
      NRLINE = crqa_result$NRLINE,
      maxL = crqa_result$maxL,
      L = crqa_result$L,
      ENTR = crqa_result$ENTR,
      rENTR = crqa_result$rENTR,
      LAM = crqa_result$LAM,
      TT = crqa_result$TT,
      max_vertlength = crqa_result$max_vertlength,
      RP = as.matrix(crqa_result$RP)
    )
  } else {
    NULL
  },
  template_similarity = as.data.frame(template_result[, c("id", "eye_sim")]),
  template_similarity_permuted = as.data.frame(
    template_perm_result[, c("id", "eye_sim", "perm_sim", "eye_sim_diff")]
  ),
  template_similarity_pca = as.data.frame(
    template_pca_result[, c("id", "eye_sim")]
  ),
  template_similarity_coral = as.data.frame(
    template_coral_result[, c("id", "eye_sim")]
  ),
  template_similarity_coral_fit_by = as.data.frame(
    template_coral_fit_by_result[, c("id", "eye_sim")]
  ),
  template_similarity_cca = as.data.frame(
    template_cca_result[, c("id", "eye_sim")]
  ),
  template_similarity_cca_fit_by = as.data.frame(
    template_cca_fit_by_result[, c("id", "eye_sim")]
  ),
  template_similarity_cv = template_cv_result_frame,
  template_similarity_contract = as.data.frame(
    template_contract_result[, c("id", "eye_sim")]
  ),
  template_similarity_contract_fit_by = as.data.frame(
    template_contract_fit_by_result[, c("id", "eye_sim")]
  ),
  template_similarity_affine = as.data.frame(
    template_affine_result[, c("id", "eye_sim")]
  ),
  template_similarity_affine_fit_by = as.data.frame(
    template_affine_fit_by_result[, c("id", "eye_sim")]
  ),
  template_similarity_cv_contract = template_cv_contract_result_frame,
  template_similarity_cv_affine = template_cv_affine_result_frame,
  template_regression = as.data.frame(
    regression_result[, c("id", "beta_baseline", "beta_source")]
  ),
  template_regression_rlm = as.data.frame(
    regression_rlm_result[, c("id", "beta_baseline", "beta_source")]
  ),
  has_ppcor = has_ppcor,
  template_regression_rank = if (has_ppcor) {
    as.data.frame(regression_rank_result[, c("id", "beta_baseline", "beta_source")])
  } else {
    NULL
  },
  template_multireg_lm = as.data.frame(
    multireg_result$multireg[[1]][, c("term", "estimate")]
  ),
  template_multireg_logistic = as.data.frame(
    multireg_logistic_result$multireg[[1]][, c("term", "estimate")]
  ),
  template_multireg_rlm = as.data.frame(
    multireg_rlm_result$multireg[[1]][, c("term", "estimate")]
  ),
  has_nnls = has_nnls,
  template_multireg_nnls = if (has_nnls) {
    as.data.frame(multireg_nnls_result$multireg[[1]][, c("term", "estimate")])
  } else {
    NULL
  }
)

cat(toJSON(out, dataframe = "columns", auto_unbox = TRUE, digits = 14, na = "null"))
"""


def _run_r_oracle() -> dict:
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript is not available")

    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            "cat(requireNamespace('pkgload', quietly=TRUE), "
            "requireNamespace('jsonlite', quietly=TRUE), sep='\\n')",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip().splitlines()[-2:] != ["TRUE", "TRUE"]:
        raise RuntimeError("R oracle requires R packages 'pkgload' and 'jsonlite'")

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(R_CODE)
        script = Path(handle.name)
    try:
        proc = subprocess.run(
            ["Rscript", str(script)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        script.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def _frame_columns(frame) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in frame.items()}


def _r_bool(value) -> bool:
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def _python_results() -> dict:
    fg = fixation_group(
        x=[0, 1, 3, 4],
        y=[0, 2, 2, 5],
        onset=[0, 100, 200, 350],
        duration=[100, 100, 150, 50],
    )
    fg_overlap_ref = fixation_group(
        x=[0, 10, 20],
        y=[0, 0, 0],
        onset=[0, 100, 200],
        duration=[100, 100, 100],
    )
    fg_overlap_shift = fixation_group(
        x=[2, 15, 40],
        y=[1, 0, 0],
        onset=[0, 100, 200],
        duration=[100, 100, 100],
    )
    point = EyeDensity(
        x=np.array([1, 2], dtype=float),
        y=np.array([1, 2], dtype=float),
        z=np.array([[1, 0], [0, 0]], dtype=float),
        sigma=1,
    )
    uniform = EyeDensity(
        x=np.array([1, 2], dtype=float),
        y=np.array([1, 2], dtype=float),
        z=np.ones((2, 2), dtype=float),
        sigma=1,
    )
    density_a = gen_density(
        x=np.array([0, 1, 2], dtype=float),
        y=np.array([0, 1, 2], dtype=float),
        z=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
    )
    density_b = gen_density(
        x=np.array([0, 1, 2], dtype=float),
        y=np.array([0, 1, 2], dtype=float),
        z=np.array([[9, 7, 6], [5, 4, 3], [2, 1, 8]], dtype=float),
    )
    density_fg = fixation_group(x=[0, 1, 2], y=[0, 1, 0], onset=[0, 100, 200], duration=[1, 2, 1])
    density_unweighted = eye_density(
        density_fg,
        sigma=1.2,
        xbounds=(0, 2),
        ybounds=(0, 2),
        outdim=(4, 4),
        normalize=False,
        duration_weighted=False,
    )
    density_weighted = eye_density(
        density_fg,
        sigma=1.2,
        xbounds=(0, 2),
        ybounds=(0, 2),
        outdim=(4, 4),
        normalize=False,
        duration_weighted=True,
    )
    template_ref = pd.DataFrame({
        "id": ["A", "B"],
        "density": [
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 2], [3, 4]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[4, 3], [2, 1]], dtype=float)),
        ],
    })
    template_src = pd.DataFrame({
        "id": ["A", "B"],
        "density": [
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 2], [3, 4]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 1], [2, 2]], dtype=float)),
        ],
    })
    template_result = template_similarity(
        template_ref,
        template_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="pearson",
        permutations=0,
    )
    template_perm_ref = pd.DataFrame({
        "id": ["A", "B", "C"],
        "density": [
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 2], [3, 4]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[4, 3], [2, 1]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 3], [2, 4]], dtype=float)),
        ],
    })
    template_perm_src = pd.DataFrame({
        "id": ["A", "B", "C"],
        "density": [
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 2], [3, 4]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[1, 1], [2, 2]], dtype=float)),
            gen_density(np.array([0, 1], dtype=float), np.array([0, 1], dtype=float), np.array([[4, 2], [3, 1]], dtype=float)),
        ],
    })
    template_perm_result = template_similarity(
        template_perm_ref,
        template_perm_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="pearson",
        permutations=10,
    )
    transform_ref = pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "pid": "p1",
        "density": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0.2, 1.0], [0.5, 1.3]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1.5, 0.7], [0.2, 0.9]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0.4, 0.1], [1.7, 1.1]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1.2, 1.6], [0.3, 0.5]], dtype=float),
            ),
        ],
    })
    transform_src = pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "pid": "p1",
        "fold_group": ["G1", "G2", "G1", "G2"],
        "density": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0.3, 1.2], [0.6, 1.1]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1.4, 0.5], [0.4, 1.0]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0.2, 0.2], [1.9, 1.0]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1.0, 1.7], [0.5, 0.4]], dtype=float),
            ),
        ],
    })
    template_pca_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=latent_pca_transform,
        similarity_transform_args={"comps": 3},
    )
    template_coral_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=coral_transform,
        similarity_transform_args={"comps": 3, "shrink": 1e-3},
    )
    template_coral_fit_by_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=coral_transform,
        similarity_transform_args={"comps": 3, "shrink": 1e-3, "fit_by": "pid"},
    )
    template_cca_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=cca_transform,
        similarity_transform_args={"comps": 3, "shrink": 1e-3},
    )
    template_cca_fit_by_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=cca_transform,
        similarity_transform_args={"comps": 3, "shrink": 1e-3, "fit_by": "pid"},
    )
    template_cv_result = template_similarity_cv(
        transform_ref,
        transform_src,
        match_on="id",
        split_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        n_folds=2,
        seed=3,
    )
    template_contract_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=contract_transform,
        similarity_transform_args={"shrink": 1e-6},
    )
    template_contract_fit_by_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=contract_transform,
        similarity_transform_args={"shrink": 1e-6, "fit_by": "pid"},
    )
    template_affine_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=affine_transform,
        similarity_transform_args={"shrink": 1e-6},
    )
    template_affine_fit_by_result = template_similarity(
        transform_ref,
        transform_src,
        match_on="id",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        similarity_transform=affine_transform,
        similarity_transform_args={"shrink": 1e-6, "fit_by": "pid"},
    )
    template_cv_contract_result = template_similarity_cv(
        transform_ref,
        transform_src,
        match_on="id",
        split_on="fold_group",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        n_folds=2,
        seed=1,
        similarity_transform=contract_transform,
        similarity_transform_args={"shrink": 1e-6},
    )
    template_cv_affine_result = template_similarity_cv(
        transform_ref,
        transform_src,
        match_on="id",
        split_on="fold_group",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
        n_folds=2,
        seed=1,
        similarity_transform=affine_transform,
        similarity_transform_args={"shrink": 1e-6},
    )
    reg_baseline = pd.DataFrame({
        "scene": ["S1", "S2"],
        "density": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1, 0], [0, 1]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0, 1], [1, 0]], dtype=float),
            ),
        ],
    })
    reg_ref = pd.DataFrame({
        "id": ["A", "B"],
        "density": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0, 1], [2, 3]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[3, 2], [1, 0]], dtype=float),
            ),
        ],
    })
    reg_source = pd.DataFrame({
        "id": ["A", "B"],
        "scene": ["S1", "S2"],
        "density": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1, 2], [3, 5]], dtype=float),
            ),
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[5, 3], [2, 1]], dtype=float),
            ),
        ],
    })
    regression_result = template_regression(
        reg_ref,
        reg_source,
        match_on="id",
        baseline_tab=reg_baseline,
        baseline_key="scene",
        method="lm",
    )
    regression_rlm_result = template_regression(
        reg_ref,
        reg_source,
        match_on="id",
        baseline_tab=reg_baseline,
        baseline_key="scene",
        method="rlm",
    )
    regression_rank_result = template_regression(
        reg_ref,
        reg_source,
        match_on="id",
        baseline_tab=reg_baseline,
        baseline_key="scene",
        method="rank",
    )
    multireg_tab = pd.DataFrame({
        "row": [1],
        "response": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1, 2], [3, 4]], dtype=float),
            )
        ],
        "cov1": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[1, 0], [0, 1]], dtype=float),
            )
        ],
        "cov2": [
            gen_density(
                np.array([0, 1], dtype=float),
                np.array([0, 1], dtype=float),
                np.array([[0, 1], [0, 1]], dtype=float),
            )
        ],
    })
    multireg_result = template_multireg(
        multireg_tab,
        response="response",
        covars=["cov1", "cov2"],
        method="lm",
        intercept=True,
    )
    multireg_logistic_result = template_multireg(
        multireg_tab,
        response="response",
        covars=["cov1", "cov2"],
        method="logistic",
        intercept=True,
    )
    multireg_rlm_result = template_multireg(
        multireg_tab,
        response="response",
        covars=["cov1", "cov2"],
        method="rlm",
        intercept=True,
    )
    multireg_nnls_result = template_multireg(
        multireg_tab,
        response="response",
        covars=["cov1", "cov2"],
        method="nnls",
        intercept=True,
    )

    return {
        "coords": np.asarray(fg.coords(), dtype=float),
        "centered": _frame_columns(center(fg).to_pandas(copy=False)),
        "normalized": _frame_columns(normalize(fg, xbounds=(0, 4), ybounds=(0, 5)).to_pandas(copy=False)),
        "rescaled": _frame_columns(rescale(fg, sx=2, sy=0.5).to_pandas(copy=False)),
        "sampled": _frame_columns(
            sample_fixations(fg, time=np.array([-10, 0, 50, 100, 250], dtype=float), fast=True)
            .to_pandas(copy=False)
        ),
        "overlap_euclidean": fixation_overlap(
            fg_overlap_ref,
            fg_overlap_shift,
            dthresh=6,
            time_samples=np.array([0, 100, 200], dtype=float),
            dist_method="euclidean",
        ),
        "overlap_manhattan": fixation_overlap(
            fg_overlap_ref,
            fg_overlap_shift,
            dthresh=6,
            time_samples=np.array([0, 100, 200], dtype=float),
            dist_method="manhattan",
        ),
        "cart2pol": cart2pol([1, 2, 0], [2, 0, -3]),
        "calcangle": calcangle([1, 0], [0, 1]),
        "scanpath": _frame_columns(scanpath(fg).to_pandas(copy=False)),
        "entropy_point": fixation_entropy(point, normalize=False),
        "entropy_uniform_norm": fixation_entropy(uniform, normalize=True),
        "entropy_grid": fixation_entropy(
            fg,
            method="grid",
            grid=(2, 2),
            xbounds=(0, 4),
            ybounds=(0, 5),
            normalize=False,
            base=2,
        ),
        "sample_density_none": _frame_columns(sample_density(density_a, fg).to_dict(orient="list")),
        "sample_density_sum": _frame_columns(sample_density(density_a, fg, normalize="sum").to_dict(orient="list")),
        "sample_density_zscore": _frame_columns(sample_density(density_a, fg, normalize="zscore").to_dict(orient="list")),
        "density_similarity": {
            method: similarity(density_a, density_b, method=method)
            for method in ("pearson", "spearman", "fisherz", "cosine", "l1")
        },
        "eye_density_unweighted": {
            "x": density_unweighted.x,
            "y": density_unweighted.y,
            "z": density_unweighted.z,
        },
        "eye_density_weighted": {
            "x": density_weighted.x,
            "y": density_weighted.y,
            "z": density_weighted.z,
        },
        "crqa": crqa(
            pd.DataFrame({"x": np.arange(1, 6, dtype=float), "y": np.arange(1, 6, dtype=float)}),
            pd.DataFrame({"x": np.arange(2, 7, dtype=float), "y": np.arange(3, 8, dtype=float)}),
            radius=60,
        ),
        "template_similarity": {
            "id": np.asarray(template_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_permuted": {
            "id": np.asarray(template_perm_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_perm_result["eye_sim"].to_numpy(dtype=float),
            "perm_sim": template_perm_result["perm_sim"].to_numpy(dtype=float),
            "eye_sim_diff": template_perm_result["eye_sim_diff"].to_numpy(dtype=float),
        },
        "template_similarity_pca": {
            "id": np.asarray(template_pca_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_pca_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_coral": {
            "id": np.asarray(template_coral_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_coral_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_coral_fit_by": {
            "id": np.asarray(template_coral_fit_by_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_coral_fit_by_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_cca": {
            "id": np.asarray(template_cca_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_cca_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_cca_fit_by": {
            "id": np.asarray(template_cca_fit_by_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_cca_fit_by_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_cv": {
            "id": np.asarray(template_cv_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_cv_result["eye_sim"].to_numpy(dtype=float),
            ".cv_fold": template_cv_result[".cv_fold"].to_numpy(dtype=float),
        },
        "template_similarity_contract": {
            "id": np.asarray(template_contract_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_contract_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_contract_fit_by": {
            "id": np.asarray(template_contract_fit_by_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_contract_fit_by_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_affine": {
            "id": np.asarray(template_affine_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_affine_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_affine_fit_by": {
            "id": np.asarray(template_affine_fit_by_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_affine_fit_by_result["eye_sim"].to_numpy(dtype=float),
        },
        "template_similarity_cv_contract": {
            "id": np.asarray(template_cv_contract_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_cv_contract_result["eye_sim"].to_numpy(dtype=float),
            ".cv_fold": template_cv_contract_result[".cv_fold"].to_numpy(dtype=float),
        },
        "template_similarity_cv_affine": {
            "id": np.asarray(template_cv_affine_result["id"].to_numpy(), dtype=str),
            "eye_sim": template_cv_affine_result["eye_sim"].to_numpy(dtype=float),
            ".cv_fold": template_cv_affine_result[".cv_fold"].to_numpy(dtype=float),
        },
        "template_regression": {
            "id": np.asarray(regression_result["id"].to_numpy(), dtype=str),
            "beta_baseline": regression_result["beta_baseline"].to_numpy(dtype=float),
            "beta_source": regression_result["beta_source"].to_numpy(dtype=float),
        },
        "template_regression_rlm": {
            "id": np.asarray(regression_rlm_result["id"].to_numpy(), dtype=str),
            "beta_baseline": regression_rlm_result["beta_baseline"].to_numpy(dtype=float),
            "beta_source": regression_rlm_result["beta_source"].to_numpy(dtype=float),
        },
        "template_regression_rank": {
            "id": np.asarray(regression_rank_result["id"].to_numpy(), dtype=str),
            "beta_baseline": regression_rank_result["beta_baseline"].to_numpy(dtype=float),
            "beta_source": regression_rank_result["beta_source"].to_numpy(dtype=float),
        },
        "template_multireg_lm": {
            "term": np.asarray(multireg_result["multireg"].iloc[0]["term"].to_numpy(), dtype=str),
            "estimate": multireg_result["multireg"].iloc[0]["estimate"].to_numpy(dtype=float),
        },
        "template_multireg_logistic": {
            "term": np.asarray(multireg_logistic_result["multireg"].iloc[0]["term"].to_numpy(), dtype=str),
            "estimate": multireg_logistic_result["multireg"].iloc[0]["estimate"].to_numpy(dtype=float),
        },
        "template_multireg_rlm": {
            "term": np.asarray(multireg_rlm_result["multireg"].iloc[0]["term"].to_numpy(), dtype=str),
            "estimate": multireg_rlm_result["multireg"].iloc[0]["estimate"].to_numpy(dtype=float),
        },
        "template_multireg_nnls": {
            "term": np.asarray(multireg_nnls_result["multireg"].iloc[0]["term"].to_numpy(), dtype=str),
            "estimate": multireg_nnls_result["multireg"].iloc[0]["estimate"].to_numpy(dtype=float),
        },
    }


def _assert_close(
    name: str,
    observed,
    expected,
    failures: list[str],
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> None:
    try:
        np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as exc:
        failures.append(f"{name}: {exc}")


def compare() -> list[str]:
    r = _run_r_oracle()
    py = _python_results()
    failures: list[str] = []

    _assert_close("coords", py["coords"], np.asarray(r["coords"], dtype=float), failures)
    _assert_close("cart2pol", py["cart2pol"], np.asarray(r["cart2pol"], dtype=float), failures)

    for key in (
        "centered",
        "normalized",
        "rescaled",
        "sampled",
        "scanpath",
        "sample_density_none",
        "sample_density_sum",
        "sample_density_zscore",
    ):
        r_cols = _frame_columns(r[key])
        for col, r_val in r_cols.items():
            if col in py[key]:
                _assert_close(f"{key}.{col}", py[key][col], r_val, failures)

    for key in ("calcangle", "entropy_point", "entropy_uniform_norm", "entropy_grid"):
        _assert_close(key, py[key], float(r[key]), failures)
    for key in ("overlap_euclidean", "overlap_manhattan"):
        for col in ("overlap", "perc"):
            _assert_close(f"{key}.{col}", py[key][col], float(r[key][col]), failures)
    for method, r_val in r["density_similarity"].items():
        _assert_close(f"density_similarity.{method}", py["density_similarity"][method], float(r_val), failures)
    for key in ("eye_density_unweighted", "eye_density_weighted"):
        _assert_close(f"{key}.x", py[key]["x"], np.asarray(r[key]["x"], dtype=float), failures)
        _assert_close(f"{key}.y", py[key]["y"], np.asarray(r[key]["y"], dtype=float), failures)
        _assert_close(f"{key}.z", py[key]["z"], np.asarray(r[key]["z"], dtype=float), failures,
                      rtol=1e-7, atol=1e-8)
    if _r_bool(r.get("has_crqa", False)):
        for key in ("RR", "DET", "NRLINE", "maxL", "L", "ENTR", "rENTR", "LAM", "TT", "max_vertlength"):
            _assert_close(f"crqa.{key}", py["crqa"][key], float(r["crqa"][key]), failures)
        try:
            np.testing.assert_array_equal(
                np.asarray(py["crqa"]["RP"], dtype=bool),
                np.asarray(r["crqa"]["RP"], dtype=bool),
            )
        except AssertionError as exc:
            failures.append(f"crqa.RP: {exc}")
    if list(py["template_similarity"]["id"]) != list(r["template_similarity"]["id"]):
        failures.append(
            "template_similarity.id: "
            f"{list(py['template_similarity']['id'])} != {list(r['template_similarity']['id'])}"
        )
    _assert_close(
        "template_similarity.eye_sim",
        py["template_similarity"]["eye_sim"],
        np.asarray(r["template_similarity"]["eye_sim"], dtype=float),
        failures,
    )
    if list(py["template_similarity_permuted"]["id"]) != list(r["template_similarity_permuted"]["id"]):
        failures.append(
            "template_similarity_permuted.id: "
            f"{list(py['template_similarity_permuted']['id'])} != "
            f"{list(r['template_similarity_permuted']['id'])}"
        )
    for col in ("eye_sim", "perm_sim", "eye_sim_diff"):
        _assert_close(
            f"template_similarity_permuted.{col}",
            py["template_similarity_permuted"][col],
            np.asarray(r["template_similarity_permuted"][col], dtype=float),
            failures,
        )
    for key in (
        "template_similarity_pca",
        "template_similarity_coral",
        "template_similarity_coral_fit_by",
        "template_similarity_cca",
        "template_similarity_cca_fit_by",
    ):
        if list(py[key]["id"]) != list(r[key]["id"]):
            failures.append(f"{key}.id: {list(py[key]['id'])} != {list(r[key]['id'])}")
        _assert_close(
            f"{key}.eye_sim",
            py[key]["eye_sim"],
            np.asarray(r[key]["eye_sim"], dtype=float),
            failures,
        )
    if list(py["template_similarity_cv"]["id"]) != list(r["template_similarity_cv"]["id"]):
        failures.append(
            "template_similarity_cv.id: "
            f"{list(py['template_similarity_cv']['id'])} != "
            f"{list(r['template_similarity_cv']['id'])}"
        )
    _assert_close(
        "template_similarity_cv.eye_sim",
        py["template_similarity_cv"]["eye_sim"],
        np.asarray(r["template_similarity_cv"]["eye_sim"], dtype=float),
        failures,
    )
    _assert_close(
        "template_similarity_cv..cv_fold",
        py["template_similarity_cv"][".cv_fold"],
        np.asarray(r["template_similarity_cv"][".cv_fold"], dtype=float),
        failures,
    )
    for key in (
        "template_similarity_contract",
        "template_similarity_contract_fit_by",
        "template_similarity_affine",
        "template_similarity_affine_fit_by",
    ):
        if list(py[key]["id"]) != list(r[key]["id"]):
            failures.append(f"{key}.id: {list(py[key]['id'])} != {list(r[key]['id'])}")
        _assert_close(
            f"{key}.eye_sim",
            py[key]["eye_sim"],
            np.asarray(r[key]["eye_sim"], dtype=float),
            failures,
            rtol=1e-7,
            atol=1e-8,
        )
    for key in ("template_similarity_cv_contract", "template_similarity_cv_affine"):
        if list(py[key]["id"]) != list(r[key]["id"]):
            failures.append(f"{key}.id: {list(py[key]['id'])} != {list(r[key]['id'])}")
        for col in ("eye_sim", ".cv_fold"):
            _assert_close(
                f"{key}.{col}",
                py[key][col],
                np.asarray(r[key][col], dtype=float),
                failures,
                rtol=1e-7,
                atol=1e-8,
            )
    if list(py["template_regression"]["id"]) != list(r["template_regression"]["id"]):
        failures.append(
            "template_regression.id: "
            f"{list(py['template_regression']['id'])} != {list(r['template_regression']['id'])}"
        )
    for col in ("beta_baseline", "beta_source"):
        _assert_close(
            f"template_regression.{col}",
            py["template_regression"][col],
            np.asarray(r["template_regression"][col], dtype=float),
            failures,
        )
    if list(py["template_regression_rlm"]["id"]) != list(r["template_regression_rlm"]["id"]):
        failures.append(
            "template_regression_rlm.id: "
            f"{list(py['template_regression_rlm']['id'])} != {list(r['template_regression_rlm']['id'])}"
        )
    for col in ("beta_baseline", "beta_source"):
        _assert_close(
            f"template_regression_rlm.{col}",
            py["template_regression_rlm"][col],
            np.asarray(r["template_regression_rlm"][col], dtype=float),
            failures,
            rtol=1e-4,
            atol=1e-4,
        )
    if _r_bool(r.get("has_ppcor", False)):
        if list(py["template_regression_rank"]["id"]) != list(r["template_regression_rank"]["id"]):
            failures.append(
                "template_regression_rank.id: "
                f"{list(py['template_regression_rank']['id'])} != {list(r['template_regression_rank']['id'])}"
            )
        for col in ("beta_baseline", "beta_source"):
            _assert_close(
                f"template_regression_rank.{col}",
                py["template_regression_rank"][col],
                np.asarray(r["template_regression_rank"][col], dtype=float),
                failures,
            )
    if list(py["template_multireg_lm"]["term"]) != list(r["template_multireg_lm"]["term"]):
        failures.append(
            "template_multireg_lm.term: "
            f"{list(py['template_multireg_lm']['term'])} != {list(r['template_multireg_lm']['term'])}"
        )
    _assert_close(
        "template_multireg_lm.estimate",
        py["template_multireg_lm"]["estimate"],
        np.asarray(r["template_multireg_lm"]["estimate"], dtype=float),
        failures,
    )
    if list(py["template_multireg_logistic"]["term"]) != list(r["template_multireg_logistic"]["term"]):
        failures.append(
            "template_multireg_logistic.term: "
            f"{list(py['template_multireg_logistic']['term'])} != "
            f"{list(r['template_multireg_logistic']['term'])}"
        )
    _assert_close(
        "template_multireg_logistic.estimate",
        py["template_multireg_logistic"]["estimate"],
        np.asarray(r["template_multireg_logistic"]["estimate"], dtype=float),
        failures,
        rtol=1e-6,
        atol=1e-7,
    )
    if list(py["template_multireg_rlm"]["term"]) != list(r["template_multireg_rlm"]["term"]):
        failures.append(
            "template_multireg_rlm.term: "
            f"{list(py['template_multireg_rlm']['term'])} != "
            f"{list(r['template_multireg_rlm']['term'])}"
        )
    _assert_close(
        "template_multireg_rlm.estimate",
        py["template_multireg_rlm"]["estimate"],
        np.asarray(r["template_multireg_rlm"]["estimate"], dtype=float),
        failures,
        rtol=1e-4,
        atol=1e-4,
    )
    if _r_bool(r.get("has_nnls", False)):
        if list(py["template_multireg_nnls"]["term"]) != list(r["template_multireg_nnls"]["term"]):
            failures.append(
                "template_multireg_nnls.term: "
                f"{list(py['template_multireg_nnls']['term'])} != "
                f"{list(r['template_multireg_nnls']['term'])}"
            )
        _assert_close(
            "template_multireg_nnls.estimate",
            py["template_multireg_nnls"]["estimate"],
            np.asarray(r["template_multireg_nnls"]["estimate"], dtype=float),
            failures,
        )

    return failures


def main() -> int:
    try:
        failures = compare()
    except RuntimeError as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return 77

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("R oracle parity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
