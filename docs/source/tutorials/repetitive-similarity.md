---
title: "Measuring Similarity Across Repeated Viewings"
description: "Within- vs. cross-stimulus similarity for repeated viewings."
---


When participants view the same image multiple times — say, during encoding
and retrieval — do they look at the same locations each time? And is that
consistency specific to the repeated stimulus, or just a general gaze pattern?

`repetitive_similarity()` compares each density map with every other map that
shares its value of a condition variable (`repsim`) and with every map that
does not (`othersim`). With the stimulus as the condition variable, `repsim` is
the similarity between viewings of the same image and `othersim` the similarity
to viewings of other images. Unlike `template_similarity()`, which needs an
explicit encoding–retrieval pairing, it uses every pair of maps.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from peyesim import (
    fixation_group, eye_table, density_by,
)
from peyesim.repetitive_similarity import repetitive_similarity
```

## Setting up the data

Let's simulate a small dataset: 2 participants view 3 images during encoding
and retrieval.

```python
rng = np.random.default_rng(42)

def gen_fixations(imname, phase, participant, rng):
    nfix = rng.integers(3, 11)
    return pd.DataFrame({
        "x": rng.uniform(0, 100, nfix),
        "y": rng.uniform(0, 100, nfix),
        "onset": np.cumsum(rng.uniform(20, 80, nfix)),
        "duration": rng.uniform(80, 300, nfix),
        "image": imname,
        "phase": phase,
        "participant": participant,
    })

rows = []
for s in ["s1", "s2"]:
    for p in ["encoding", "retrieval"]:
        for img in ["img1", "img2", "img3"]:
            rows.append(gen_fixations(img, p, s, rng))

df = pd.concat(rows, ignore_index=True)

eyetab = eye_table(
    "x", "y", "duration", "onset",
    groupvar=["participant", "phase", "image"],
    data=df,
)
```

Compute density maps for each participant-phase-image combination:

```python
eyedens = density_by(
    eyetab,
    groups=["phase", "image", "participant"],
    sigma=15,
    xbounds=(0, 100),
    ybounds=(0, 100),
)
eyedens.drop(columns=["density", "fixgroup"], errors="ignore").head()
```

All maps must share one grid (the same `xbounds`, `ybounds` and `outdim`);
`repetitive_similarity()` refuses the whole call otherwise. Passing explicit
bounds, as here, guarantees it.

## Computing repetitive similarity

`repetitive_similarity()` takes the density table and the condition variable
that defines which maps count as repeats. To ask about the stimulus, use the
image:

```python
rep_sim = repetitive_similarity(
    eyedens,
    condition_var="image",
    method="pearson",
)
rep_sim.drop(columns=["density", "fixgroup"], errors="ignore")
```

The result adds two columns:

- **repsim**: the mean similarity of each map to all other maps of the *same*
  image, across phases and participants.
- **othersim**: the mean similarity of each map to all maps of *other* images.

The condition variable decides what "same" means. With `condition_var="phase"`,
`repsim` would instead compare maps from the same phase, across different
images.

## Interpreting the results

Each image has four maps (two participants × two phases), so summarise per
image before plotting:

```python
per_image = rep_sim.groupby("image")[["repsim", "othersim"]].mean()

fig, ax = plt.subplots(figsize=(6, 3.5))
per_image.plot.bar(ax=ax, color=["steelblue", "coral"], rot=0)
ax.set_xlabel("Image")
ax.set_ylabel("Mean Pearson similarity")
ax.legend(["Same image (repsim)", "Other images (othersim)"])
plt.tight_layout()
```

Neither value is near zero, even though these fixations are random: smoothed
density maps of fixations spread over the same display correlate positively
whatever their content. That is why `othersim` is needed as a baseline. The
difference `repsim - othersim` is the stimulus-specific part, and for random
data it should hover around zero:

```python
diff = rep_sim["repsim"] - rep_sim["othersim"]
diff.describe()[["mean", "std", "min", "max"]].round(3)
```

In a real experiment, `repsim` above `othersim` indicates that viewers return
to image-specific locations: stimulus-specific eye-movement reinstatement
beyond general gaze tendencies.

## What's next?

- `pairwise=True` adds a `pairwise_repsim` column holding each map's
  similarities to the other maps of its own condition; `repsim` and `othersim`
  are still returned. For multiscale maps, `multiscale_aggregation` controls how
  the scales are combined.
- [Comparing eye-movement patterns](/tutorials/overview) covers the full `template_similarity()` workflow
  with permutation-based baselines
- [Latent Transforms](/tutorials/latent-transforms) describes domain-adaptation methods that
  can improve similarity estimates across devices or participants
