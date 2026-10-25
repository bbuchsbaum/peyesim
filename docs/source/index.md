---
title: "peyesim"
description: "Fixation densities, template similarity and scanpath comparison for eye-movement data."
---

peyesim is a Python port of the R package
[eyesim](https://github.com/bbuchsbaum/eyesim). It compares fixation patterns
between viewings, for example between encoding and retrieval in a memory
experiment, using fixation density maps, template similarity with permutation
baselines, MultiMatch scanpath comparison and cross-recurrence analysis.

```bash
pip install "peyesim @ git+https://github.com/bbuchsbaum/peyesim"
# with matplotlib for plotting:
pip install "peyesim[viz] @ git+https://github.com/bbuchsbaum/peyesim"
```

```python
import numpy as np
from peyesim import fixation_group, eye_density, similarity

rng = np.random.default_rng(1)

def viewing():
    return fixation_group(
        x=rng.uniform(0, 100, 25),
        y=rng.uniform(0, 100, 25),
        onset=np.cumsum(rng.uniform(0, 100, 25)),
        duration=rng.uniform(50, 300, 25),
    )

encode = eye_density(viewing(), sigma=20, xbounds=(0, 100), ybounds=(0, 100))
retrieve = eye_density(viewing(), sigma=20, xbounds=(0, 100), ybounds=(0, 100))
similarity(encode, retrieve, method="pearson")
```

## What it covers

| Task | Functions |
| --- | --- |
| Fixation tables and groups | `fixation_group()`, `eye_table()`, `sample_fixations()` |
| Density maps | `eye_density()`, `density_by()`, `suggest_sigma()` |
| Similarity between maps | `similarity()` with Pearson, Spearman, Fisher z, cosine, L1, Jaccard, distance correlation and EMD |
| Encoding–retrieval comparison | `template_similarity()`, `template_similarity_cv()`, `sample_density_time()` |
| Scanpaths | `scanpath()`, `multi_match()`, `scanpath_similarity()`, `crqa()` |
| Revisitation | `fixation_revisits()`, `revisit_summary()`, `visual_angle_to_px()` |
| Domain adaptation | `latent_pca_transform()`, `coral_transform()`, `cca_transform()`, `contract_transform()`, `affine_transform()` |
| Plots | `plot_fixation_group()`, `plot_eye_density()`, `anim_scanpath()` |

## Relation to R eyesim

peyesim follows eyesim 0.3.0, except for the experimental GazeWeave models,
which are not ported. When R and the eyesim source are available, the test
suite compares results against R directly. The remaining known differences are
listed in
[PARITY.md](https://github.com/bbuchsbaum/peyesim/blob/main/PARITY.md).

## Where to go next

- [Get started](/get-started): the three object types and the main functions.
- [Tutorials](/tutorials): one worked analysis per workflow.
- [API reference](/reference): every public function and class.
