---
title: "Comparing Eye-Movement Patterns"
description: "Core workflow: fixations, density maps, template similarity, and multiscale analysis."
---


When you study an image, you move your eyes to different locations in a
particular sequence. If you later see that same image during a memory test, do
you look at the same places? This kind of "eye-movement reinstatement"
tracks memory retrieval, and `peyesim` gives you the tools to measure it.

This tutorial walks you through the core workflow: representing fixations,
computing density maps, and measuring similarity between fixation patterns
across experimental conditions.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from peyesim import (
    fixation_group, eye_density, similarity,
    eye_table, density_by, template_similarity,
    sample_density_time,
)
```

## How do you represent fixations?

A `fixation_group` holds a set of eye fixations — each defined by an x/y
screen position, an onset time (when the fixation began), and a duration (how
long the eye stayed there). You can create one directly:

```python
fg = fixation_group(
    x=[-100, 0, 100],
    y=[0, 100, 0],
    onset=[0, 10, 60],
    duration=[10, 50, 100],
)
fg.to_pandas()
```

Here is a more realistic group with 25 randomly placed fixations:

```python
rng = np.random.default_rng(42)
fg = fixation_group(
    x=rng.uniform(0, 100, 25),
    y=rng.uniform(0, 100, 25),
    onset=np.cumsum(rng.uniform(0, 100, 25)),
    duration=rng.uniform(50, 300, 25),
)
fg.to_pandas().head()
```

## How do you visualize fixation density?

Individual points can be hard to interpret. Density maps show you where
fixations cluster. Convert a fixation group into an `EyeDensity` and
plot it as a heatmap:

`sigma` is the standard deviation of the Gaussian kernel, in the units of the
coordinates. On this 100 × 100 display, values from about 5 to 25 keep the
structure of the pattern; much larger values spread every fixation over the
whole display.

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, bw, title in zip(axes, [5, 15, 25], ["Narrow (5)", "Medium (15)", "Wide (25)"]):
    d = eye_density(fg, sigma=bw, xbounds=(0, 100), ybounds=(0, 100))
    ax.imshow(d.z.T, origin="lower", extent=[d.x.min(), d.x.max(), d.y.min(), d.y.max()], cmap="hot")
    ax.set_title(title)
plt.suptitle("Effect of bandwidth on density maps")
plt.tight_layout()
plt.show()
```

## How do you compare two fixation patterns?

To quantify how similar two fixation patterns are, convert them to
`EyeDensity` maps and call `similarity()`. Here we create two patterns that
share roughly half their fixation locations:

```python
rng = np.random.default_rng(123)
x_shared = rng.uniform(0, 100, 12)
y_shared = rng.uniform(0, 100, 12)

fg1 = fixation_group(
    x=np.concatenate([x_shared, rng.uniform(0, 100, 13)]),
    y=np.concatenate([y_shared, rng.uniform(0, 100, 13)]),
    onset=np.cumsum(rng.uniform(0, 100, 25)),
    duration=rng.uniform(50, 300, 25),
)

fg2 = fixation_group(
    x=np.concatenate([x_shared, rng.uniform(0, 100, 13)]),
    y=np.concatenate([y_shared, rng.uniform(0, 100, 13)]),
    onset=np.cumsum(rng.uniform(0, 100, 25)),
    duration=rng.uniform(50, 300, 25),
)
```

```python
ed1 = eye_density(fg1, sigma=10, xbounds=(0, 100), ybounds=(0, 100))
ed2 = eye_density(fg2, sigma=10, xbounds=(0, 100), ybounds=(0, 100))

print("Pearson similarity:", similarity(ed1, ed2, method="pearson"))
```

Large bandwidths make any two maps look alike, so choose `sigma` with care;
`suggest_sigma()` gives a data-driven starting value. `similarity()` compares
maps cell by cell and refuses maps on different grids, so pass the same
`xbounds`, `ybounds` and `outdim` to every map you compare.

Several similarity methods are available:

```python
methods = ["pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov"]
results = {m: similarity(ed1, ed2, method=m) for m in methods}
pd.DataFrame({"method": methods, "similarity": [round(results[m], 4) for m in methods]})
```

## How do you analyze a full experiment?

In a typical memory study, participants view images during encoding and again
during retrieval. You want to compare fixation patterns between these phases
for the same image and control for non-specific similarity.

Let's simulate a small experiment: 3 participants, 20 images, encoding +
retrieval.

```python
rng = np.random.default_rng(99)

def gen_fixations(imname, phase, participant, rng):
    nfix = rng.integers(3, 11)
    return pd.DataFrame({
        "x": rng.uniform(0, 100, nfix),
        "y": rng.uniform(0, 100, nfix),
        "onset": np.cumsum(rng.uniform(50, 200, nfix)),
        "duration": rng.uniform(100, 400, nfix),
        "image": imname,
        "phase": phase,
        "participant": participant,
    })

rows = []
for s in ["s1", "s2", "s3"]:
    for p in ["encoding", "retrieval"]:
        for img in [f"img{i}" for i in range(1, 21)]:
            rows.append(gen_fixations(img, p, s, rng))

df = pd.concat(rows, ignore_index=True)
print(f"Rows: {len(df)} | Participants: {df['participant'].nunique()} | Images: {df['image'].nunique()}")
df.head()
```

Wrap the raw data in an `EyeTable`, which groups fixations by the variables
that define your experimental design:

```python
eyetab = eye_table(
    "x", "y", "duration", "onset",
    groupvar=["participant", "phase", "image"],
    data=df,
)
print(eyetab)
```

### Computing density maps by group

`density_by()` computes a density map for every combination of your grouping
variables:

```python
eyedens = density_by(
    eyetab,
    groups=["phase", "image", "participant"],
    sigma=15,
    xbounds=(0, 100),
    ybounds=(0, 100),
)
# One key per participant and image, so each retrieval map is matched with
# the same participant's encoding map of that image.
eyedens["match_key"] = eyedens["participant"].astype(str) + "_" + eyedens["image"].astype(str)
eyedens.drop(columns=["density", "fixgroup"], errors="ignore").head()
```

### Template similarity analysis

`template_similarity()` compares each retrieval density map with the same
participant's encoding map of that image. With `permute_on="participant"`, the
baseline averages that participant's encoding maps of the other images, and the
result reports the corrected difference:

```python
enc_dens = eyedens[eyedens["phase"] == "encoding"].reset_index(drop=True)
ret_dens = eyedens[eyedens["phase"] == "retrieval"].reset_index(drop=True)

simres = template_similarity(
    enc_dens, ret_dens,
    match_on="match_key",
    permute_on="participant",
    method="fisherz",
    permutations=50,
)
simres.drop(columns=["density", "fixgroup"], errors="ignore").head()
```

The result includes four key columns:

- `eye_sim` — raw similarity between matched pairs
- `perm_sim` — average similarity across non-matching pairs (baseline)
- `eye_sim_diff` — the corrected score (raw minus baseline)
- `n_perm` — how many distinct non-matching templates entered the baseline
  (0 when none could be computed)

Since our data is random, there should be no true reinstatement:

```python
from scipy import stats

t_result = stats.ttest_1samp(simres["eye_sim_diff"].dropna(), 0)
print(f"t = {t_result.statistic:.3f}, p = {t_result.pvalue:.3f}")
```

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label in zip(axes,
    ["eye_sim", "perm_sim", "eye_sim_diff"],
    ["Raw", "Permuted", "Corrected"]):
    ax.hist(simres[col].dropna(), bins=20, color="steelblue", alpha=0.7)
    ax.set_xlabel("Similarity (Fisher z)")
    ax.set_ylabel("Count")
    ax.set_title(label)
plt.tight_layout()
plt.show()
```

As expected, the corrected similarity is centered near zero.

**A note on units.** With `method="fisherz"`, all similarity values are
Fisher z scores (atanh of Pearson r). Convert back to correlations with
`np.tanh()` if desired. With `method="pearson"` or `"spearman"`, values are
correlations directly.

**A note on matching.** `match_on` must identify one reference map per source
row. Matching on `image` alone would pair every participant's retrieval map
with the *first* encoding map of that image, here participant s1's.

**A note on permutations.** The `permutations` argument is an upper bound. If
fewer non-matching candidates are available, `peyesim` uses all of them
exhaustively.

## What about multiscale analysis?

Choosing a single bandwidth is somewhat arbitrary. You can compute density at
multiple scales by passing a list of sigma values:

```python
eyedens_multi = density_by(
    eyetab,
    groups=["phase", "image", "participant"],
    sigma=[10, 15, 25],
    xbounds=(0, 100),
    ybounds=(0, 100),
)
eyedens_multi["match_key"] = (
    eyedens_multi["participant"].astype(str) + "_" + eyedens_multi["image"].astype(str)
)

enc_multi = eyedens_multi[eyedens_multi["phase"] == "encoding"].reset_index(drop=True)
ret_multi = eyedens_multi[eyedens_multi["phase"] == "retrieval"].reset_index(drop=True)

simres_multi = template_similarity(
    enc_multi, ret_multi,
    match_on="match_key",
    permute_on="participant",
    method="fisherz",
    permutations=50,
)

print(f"Single-scale mean: {simres['eye_sim_diff'].mean():.4f}")
print(f"Multiscale mean:   {simres_multi['eye_sim_diff'].mean():.4f}")
```

With the default `multiscale_aggregation="mean"`, each comparison is the
average over the scales, which makes the result depend less on any single
bandwidth. Pass `multiscale_aggregation="none"` to keep one value per scale.

## How does similarity evolve over time?

The previous analyses summarize each trial with a single density map. Sometimes
you want to know when retrieval fixations land on encoding-hot regions. Use
`sample_density_time()` to sample the encoding template at each retrieval
fixation over a sequence of time points:

```python
enc_temporal = enc_dens.copy()
ret_temporal = eyetab[eyetab["phase"] == "retrieval"].reset_index(drop=True).copy()

ret_temporal["match_key"] = (
    ret_temporal["participant"].astype(str) + "_" + ret_temporal["image"].astype(str)
)

temporal = sample_density_time(
    template_tab=enc_temporal,
    source_tab=ret_temporal,
    match_on="match_key",
    times=np.arange(0, 500, 50),
    time_bins=np.array([0, 250, 500]),
    permutations=10,
    permute_on="participant",
)

temporal[
    ["participant", "image", "bin_1", "perm_bin_1", "diff_bin_1",
     "bin_2", "perm_bin_2", "diff_bin_2"]
].head()
```

The `sampled` column contains the full time course for each retrieval trial.
The `bin_*` columns summarize early and late windows, while `diff_bin_*`
subtracts a participant-stratified permutation baseline. Bins are half-open,
[lower, upper), so a time equal to the last break (500) would fall in no bin.
Times before a trial's first fixation have no gaze position and give NaN. With
real data, a positive `diff_bin_*` shows when retrieval gaze returns to regions
fixated during encoding; these data are random, so the differences are near
zero.

```python
time_rows = []
for _, row in temporal.head(6).iterrows():
    sampled = row["sampled"].copy()
    sampled["trial"] = f"{row['participant']} / {row['image']}"
    time_rows.append(sampled)

time_series = pd.concat(time_rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 4))
for trial, group in time_series.groupby("trial"):
    ax.plot(group["time"], group["z"], marker="o", linewidth=1.4, label=trial)

ax.set_xlabel("Retrieval time")
ax.set_ylabel("Encoding density at fixation")
ax.set_title("Temporal sampling of encoding templates")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
```

## What's next?

- [MultiMatch](/tutorials/multimatch) — compare scanpath structure on MultiMatch's five dimensions plus an order-free EMD position measure
- [Repetitive Similarity](/tutorials/repetitive-similarity) — analyze repeated viewing of the same stimulus
- [Latent Transforms](/tutorials/latent-transforms) — domain adaptation for cross-device or cross-participant comparisons
- [Revisitation](/tutorials/revisitation) — returns to previously fixated locations within a viewing
- Real data: `load_dataset("wynn_study")` and `load_dataset("wynn_test")` hold a recognition-memory experiment
