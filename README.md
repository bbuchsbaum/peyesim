# peyesim

Python port of the R [eyesim](https://github.com/bbuchsbaum/eyesim) package for analyzing eye-movement data.

`peyesim` provides tools for comparing fixation patterns across experimental conditions — measuring eye-movement reinstatement, scanpath similarity, and density-based overlap. It is designed for memory researchers studying how gaze patterns change between encoding and retrieval.

## Installation

```bash
pip install -e .
```

### Dependencies

numpy, pandas, scipy, scikit-learn, pot, networkx

## Quick start

```python
import numpy as np
from peyesim import fixation_group, eye_density, similarity

# Create two fixation patterns
fg1 = fixation_group(
    x=np.random.uniform(0, 100, 25),
    y=np.random.uniform(0, 100, 25),
    onset=np.cumsum(np.random.uniform(0, 100, 25)),
    duration=np.random.uniform(50, 300, 25),
)
fg2 = fixation_group(
    x=np.random.uniform(0, 100, 25),
    y=np.random.uniform(0, 100, 25),
    onset=np.cumsum(np.random.uniform(0, 100, 25)),
    duration=np.random.uniform(50, 300, 25),
)

# Convert to density maps and compare
ed1 = eye_density(fg1, sigma=50, xbounds=(0, 100), ybounds=(0, 100))
ed2 = eye_density(fg2, sigma=50, xbounds=(0, 100), ybounds=(0, 100))
similarity(ed1, ed2, method="pearson")
```

## Documentation

Full tutorials and API reference pages live in the Quarto documentation:

- [Get started](docs/get-started.qmd) — fixations, density maps, and a first similarity score
- [Comparing Eye-Movement Patterns](docs/tutorials/overview.qmd) — template similarity and multiscale analysis
- [Comparing Scanpaths with MultiMatch](docs/tutorials/multimatch.qmd) — scanpath comparison across six dimensions
- [Measuring Similarity Across Repeated Viewings](docs/tutorials/repetitive-similarity.qmd) — within- vs. cross-stimulus similarity
- [Latent Transforms for Template Similarity](docs/tutorials/latent-transforms.qmd) — PCA, CORAL, CCA, and geometric alignment
- [API reference](docs/reference/index.qmd) — public functions and data structures

## Key features

- **Fixation density maps** — kernel density estimation with configurable bandwidth and multiscale support
- **Template similarity** — compare encoding vs. retrieval gaze with permutation-based baselines
- **MultiMatch** — scanpath comparison across vector, direction, length, position, duration, and EMD dimensions
- **CRQA** — cross-recurrence summaries for multidimensional fixation trajectories
- **Repetitive similarity** — within- vs. cross-stimulus consistency
- **Latent transforms** — PCA, CORAL, and CCA for cross-device/cross-participant domain adaptation
- **Similarity methods** — Pearson, Spearman, Fisher z, cosine, L1, Jaccard, distance covariance, EMD
- **Visualization** — static fixation/density plots and scanpath animation with optional matplotlib support

## Lineage

This is a Python port of the R [eyesim](https://github.com/bbuchsbaum/eyesim) package. The R package source is included in the `eyesim/` directory for reference.

## License

See [LICENSE](LICENSE) for details.
