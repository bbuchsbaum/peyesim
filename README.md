# peyesim

[Documentation](https://bbuchsbaum.github.io/peyesim/) ·
[API reference](https://bbuchsbaum.github.io/peyesim/reference/) ·
[Source](https://github.com/bbuchsbaum/peyesim)

Python port of the R package [eyesim](https://github.com/bbuchsbaum/eyesim)
for analyzing eye-movement data.

peyesim compares fixation patterns between viewings, for example between
encoding and retrieval in a memory experiment. It builds fixation density
maps, scores template similarity against permutation baselines, compares
scanpaths with MultiMatch and cross-recurrence analysis, and scores
revisitation within a viewing.

## Installation

peyesim is not on PyPI yet. Install it from GitHub (Python 3.9 or later):

```bash
pip install "peyesim @ git+https://github.com/bbuchsbaum/peyesim"
# with matplotlib for the plotting functions:
pip install "peyesim[viz] @ git+https://github.com/bbuchsbaum/peyesim"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/bbuchsbaum/peyesim
cd peyesim
pip install -e ".[dev,viz]"
python -m pytest
```

Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, POT and
networkx; matplotlib is optional (`viz`).

## Quick start

Compare an encoding viewing with a retrieval that returns to the same
locations, and with an unrelated viewing:

```python
import numpy as np
from peyesim import fixation_group, eye_density, similarity

rng = np.random.default_rng(1)

def viewing(x, y):
    n = len(x)
    return fixation_group(x=x, y=y, onset=np.arange(n) * 250.0, duration=np.full(n, 200.0))

# Encoding: 20 fixations on a 100 x 100 display
ex, ey = rng.uniform(0, 100, 20), rng.uniform(0, 100, 20)
encoding = viewing(ex, ey)
# Retrieval that returns to the encoded locations, with some spatial noise
recall = viewing(ex + rng.normal(0, 5, 20), ey + rng.normal(0, 5, 20))
# An unrelated viewing
other = viewing(rng.uniform(0, 100, 20), rng.uniform(0, 100, 20))

def density(fg):
    # sigma is the Gaussian kernel SD, in display units
    return eye_density(fg, sigma=10, xbounds=(0, 100), ybounds=(0, 100))

similarity(density(encoding), density(recall), method="pearson")  # 0.89
similarity(density(encoding), density(other), method="pearson")   # 0.31
```

Maps compared with `similarity()` must share a grid, so give them the same
bounds and `outdim`.

## Documentation

The documentation site is at
[bbuchsbaum.github.io/peyesim](https://bbuchsbaum.github.io/peyesim/). Every
tutorial is executed when the site is built.

- [Get started](https://bbuchsbaum.github.io/peyesim/get-started/): fixations, density maps and a first similarity score
- [Comparing eye-movement patterns](https://bbuchsbaum.github.io/peyesim/tutorials/overview/): template similarity and multiscale analysis
- [Comparing scanpaths with MultiMatch](https://bbuchsbaum.github.io/peyesim/tutorials/multimatch/): scanpath comparison in six dimensions
- [Similarity across repeated viewings](https://bbuchsbaum.github.io/peyesim/tutorials/repetitive-similarity/): within- versus between-stimulus similarity
- [Latent transforms for template similarity](https://bbuchsbaum.github.io/peyesim/tutorials/latent-transforms/): PCA, CORAL, CCA and geometric alignment
- [Revisitation](https://bbuchsbaum.github.io/peyesim/tutorials/revisitation/): returns to previously fixated locations, with a spatial null
- [API reference](https://bbuchsbaum.github.io/peyesim/reference/): every public function and class

The site's source is in [`docs/`](docs/README.md).

## Features

- **Density maps:** Gaussian kernel density estimation of fixations, optionally duration-weighted, at one or several bandwidths.
- **Similarity:** Pearson, Spearman, Fisher z, cosine, L1, extended Jaccard, distance correlation and Earth Mover's Distance between maps.
- **Template similarity:** each retrieval map against its matched encoding map, with a baseline from non-matching maps, and a cross-validated variant for fitted transforms.
- **Scanpaths:** MultiMatch in six dimensions (vector, direction, length, position, duration, EMD position) and cross-recurrence quantification (CRQA).
- **Repeated viewings:** similarity within versus between conditions.
- **Revisitation:** returns to previously fixated locations, with subject- or item-matched spatial null models, and conversion between visual angle and pixels.
- **Latent transforms:** PCA, CORAL, CCA, contraction and affine alignment of maps before comparison, for cross-device or cross-participant data.
- **Plots:** fixation and density plots and scanpath animation with matplotlib, in the colours of the R package.
- **Datasets:** the `wynn_study`, `wynn_test` and `wynn_study_image` recognition-memory datasets from eyesim.

## Relation to R eyesim

peyesim follows eyesim 0.3.0, except for the experimental GazeWeave models,
which are not ported. The test suite can compare results against R directly:
with R, the R packages `pkgload` and `jsonlite`, and an eyesim source checkout,
the `tests/test_parity_*.py` modules run the same computations in both
languages. The checkout is found at `eyesim/` (a symlink in this repository)
or at `$EYESIM_SOURCE_DIR`; without it those tests are skipped.
[PARITY.md](PARITY.md) lists the remaining known differences.

## License

MIT; see [LICENSE](LICENSE).
