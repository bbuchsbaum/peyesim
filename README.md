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

Full tutorials are available as Jupyter notebooks (with rendered output viewable on GitHub):

- [Comparing Eye-Movement Patterns](notebooks/01_overview.ipynb) — core workflow: fixations, density maps, template similarity, multiscale analysis
- [Comparing Scanpaths with MultiMatch](notebooks/02_multimatch.ipynb) — scanpath comparison across six dimensions
- [Measuring Similarity Across Repeated Viewings](notebooks/03_repetitive_similarity.ipynb) — within- vs. cross-stimulus similarity
- [Latent Transforms for Template Similarity](notebooks/04_latent_transforms.ipynb) — PCA, CORAL, and CCA domain adaptation

## Key features

- **Fixation density maps** — kernel density estimation with configurable bandwidth and multiscale support
- **Template similarity** — compare encoding vs. retrieval gaze with permutation-based baselines
- **MultiMatch** — scanpath comparison across vector, direction, length, position, duration, and EMD dimensions
- **Repetitive similarity** — within- vs. cross-stimulus consistency
- **Latent transforms** — PCA, CORAL, and CCA for cross-device/cross-participant domain adaptation
- **Similarity methods** — Pearson, Spearman, Fisher z, cosine, L1, Jaccard, distance covariance, EMD

## Lineage

This is a Python port of the R [eyesim](https://github.com/bbuchsbaum/eyesim) package. The R package source is included in the `eyesim/` directory for reference.

## License

See [LICENSE](eyesim/LICENSE) for details.
