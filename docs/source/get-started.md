---
title: "Get started"
description: "From a list of fixations to a similarity score."
---

## Install

peyesim is not on PyPI yet; install it from GitHub (Python 3.9 or later):

```bash
pip install "peyesim @ git+https://github.com/bbuchsbaum/peyesim"
```

The `viz` extra adds matplotlib for the plotting functions:
`pip install "peyesim[viz] @ git+https://github.com/bbuchsbaum/peyesim"`.
The bundled example datasets load without optional dependencies.

## Fixations, densities, similarity

An analysis usually has three steps. A `FixationGroup` holds the fixations of
one viewing: x and y position, onset and duration.

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

encode, retrieve = viewing(), viewing()
encode
```

`eye_density()` turns the fixations into a smoothed density map on a fixed
grid. `sigma` is the standard deviation of the Gaussian kernel, in the units
of the coordinates.

```python
ed_encode = eye_density(encode, sigma=20, xbounds=(0, 100), ybounds=(0, 100))
ed_retrieve = eye_density(retrieve, sigma=20, xbounds=(0, 100), ybounds=(0, 100))
ed_encode.z.shape
```

`similarity()` compares two maps cell by cell. Both maps must be on the same
grid, so give them the same bounds and `outdim`.

```python
{m: round(similarity(ed_encode, ed_retrieve, method=m), 3)
 for m in ["pearson", "spearman", "cosine", "l1"]}
```

## Objects and functions

| Object | Contents | Constructor |
| --- | --- | --- |
| `FixationGroup` | fixations of one viewing: x, y, onset, duration | `fixation_group()` |
| `EyeTable` | one row per viewing, with a fixation-group column | `eye_table()` |
| `EyeDensity` | a 2-D density map on a grid | `eye_density()` |
| `Scanpath` | saccades between consecutive fixations | `scanpath()` |

The functions used most often:

- `template_similarity()`: similarity of each retrieval map to its matched
  encoding map, with a permutation baseline from non-matching maps.
- `multi_match()`: MultiMatch scanpath comparison in six dimensions.
- `repetitive_similarity()`: similarity within and between conditions.
- `fixation_revisits()`: returns to locations already fixated in the same
  viewing.

## Next steps

- [Comparing eye-movement patterns](/tutorials/overview)
- [Comparing scanpaths with MultiMatch](/tutorials/multimatch)
- [Similarity across repeated viewings](/tutorials/repetitive-similarity)
- [Latent transforms for template similarity](/tutorials/latent-transforms)
- [Revisitation](/tutorials/revisitation)
