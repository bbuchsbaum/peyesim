# peyesim

[Documentation](https://bbuchsbaum.github.io/peyesim/) ·
[API reference](https://bbuchsbaum.github.io/peyesim/reference/) ·
[Source](https://github.com/bbuchsbaum/peyesim)

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

The rendered documentation site is available at
[bbuchsbaum.github.io/peyesim](https://bbuchsbaum.github.io/peyesim/).

- [Get started](https://bbuchsbaum.github.io/peyesim/get-started.html) — fixations, density maps, and a first similarity score
- [Comparing Eye-Movement Patterns](https://bbuchsbaum.github.io/peyesim/tutorials/overview.html) — template similarity and multiscale analysis
- [Comparing Scanpaths with MultiMatch](https://bbuchsbaum.github.io/peyesim/tutorials/multimatch.html) — scanpath comparison across six dimensions
- [Measuring Similarity Across Repeated Viewings](https://bbuchsbaum.github.io/peyesim/tutorials/repetitive-similarity.html) — within- vs. cross-stimulus similarity
- [Latent Transforms for Template Similarity](https://bbuchsbaum.github.io/peyesim/tutorials/latent-transforms.html) — PCA, CORAL, CCA, and geometric alignment
- [API reference](https://bbuchsbaum.github.io/peyesim/reference/) — public functions and data structures

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
