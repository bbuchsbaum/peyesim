# peyesim

[Documentation](https://bbuchsbaum.github.io/peyesim/) ·
[API reference](https://bbuchsbaum.github.io/peyesim/reference) ·
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

- [Get started](https://bbuchsbaum.github.io/peyesim/get-started) — fixations, density maps, and a first similarity score
- [Comparing Eye-Movement Patterns](https://bbuchsbaum.github.io/peyesim/tutorials/overview) — template similarity and multiscale analysis
- [Comparing Scanpaths with MultiMatch](https://bbuchsbaum.github.io/peyesim/tutorials/multimatch) — scanpath comparison across six dimensions
- [Measuring Similarity Across Repeated Viewings](https://bbuchsbaum.github.io/peyesim/tutorials/repetitive-similarity) — within- vs. cross-stimulus similarity
- [Latent Transforms for Template Similarity](https://bbuchsbaum.github.io/peyesim/tutorials/latent-transforms) — PCA, CORAL, CCA, and geometric alignment
- [Revisitation](https://bbuchsbaum.github.io/peyesim/tutorials/revisitation) — returns to previously fixated locations, with a spatial null
- [API reference](https://bbuchsbaum.github.io/peyesim/reference) — public functions and data structures

## Key features

- **Fixation density maps** — kernel density estimation with configurable bandwidth and multiscale support
- **Template similarity** — compare encoding vs. retrieval gaze with permutation-based baselines
- **MultiMatch** — scanpath comparison across vector, direction, length, position, duration, and EMD dimensions
- **CRQA** — cross-recurrence summaries for multidimensional fixation trajectories
- **Repetitive similarity** — within- vs. cross-stimulus consistency
- **Revisitation** — returns to previously fixated locations, with subject- or item-matched spatial nulls, and visual-angle ↔ pixel conversion
- **Latent transforms** — PCA, CORAL, and CCA for cross-device/cross-participant domain adaptation
- **Similarity methods** — Pearson, Spearman, Fisher z, cosine, L1, Jaccard, distance covariance, EMD
- **Visualization** — static fixation/density plots and scanpath animation (matplotlib), sharing eyesim's colour roles and density/time ramps

## Lineage

This is a Python port of the R [eyesim](https://github.com/bbuchsbaum/eyesim) package (synced to eyesim 0.3.0, excluding the experimental GazeWeave models). The `eyesim/` entry points at the R source; with R and the source available, `tests/test_parity_*.py` check results against R live. See [PARITY.md](PARITY.md).

## License

See [LICENSE](LICENSE) for details.
