---
title: "Comparing Scanpaths with MultiMatch"
description: "Compare scanpath structure across vector, direction, length, position, duration, and EMD dimensions."
---


A single similarity number can hide important differences between scanpaths.
Two people might fixate the same locations but in a completely different order,
or follow the same trajectory but at different speeds. MultiMatch
(Dewhurst et al., 2012) addresses this by evaluating scanpath similarity
across five distinct dimensions:

| Dimension | What it captures |
|:----------|:-----------------|
| **vector** | Overall shape similarity of saccade vectors |
| **direction** | Angular similarity of saccade directions |
| **length** | Similarity of saccade amplitudes |
| **position** | Distance between the start points of aligned saccades (the last fixation is not included) |
| **duration** | Relative difference `abs(d1 - d2) / max(d1, d2)` of aligned fixation durations |

The `peyesim` implementation adds a sixth measure, **position_emd**: the
duration-weighted earth mover's distance between all fixations (including the
last), as `1 - EMD / screen diagonal`. It ignores the order of the fixations.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from peyesim import fixation_group, scanpath, multi_match
```

## How do you compare two scanpaths?

Start by creating two `fixation_group` objects, convert them to `Scanpath`
objects, and call `multi_match()`:

```python
rng = np.random.default_rng(1)

def simulate_linear(n, rng):
    return fixation_group(
        x=np.cumsum(rng.normal(50, 10, n)),
        y=np.cumsum(rng.normal(50, 10, n)),
        duration=rng.poisson(200, n),
        onset=np.arange(1, n + 1, dtype=float),
    )

fg1 = simulate_linear(10, rng)
fg2 = simulate_linear(12, rng)
```

```python
sp1 = scanpath(fg1)
sp2 = scanpath(fg2)
multi_match(sp1, sp2, screensize=(500, 500))
```

All metrics are high, consistent with the known similarity of the two
scanpaths.

## What happens when scanpaths differ structurally?

Let's compare a linear scanpath against a zigzag pattern:

```python
fg_zigzag = fixation_group(
    x=np.cumsum(np.full(10, 50.0)),
    # alternate shallow and steep downward saccades: neither leg points the
    # way the linear path goes (up and to the right)
    y=400 + np.cumsum(np.tile([-10.0, -50.0], 5)),
    duration=rng.poisson(200, 10),
    onset=np.arange(1, 11, dtype=float),
)

sp_linear = scanpath(simulate_linear(10, rng))
sp_zigzag = scanpath(fg_zigzag)
multi_match(sp_linear, sp_zigzag, screensize=(500, 500))
```

**Direction** falls well below its value for random pairs (about 0.72, shown
below): neither zigzag leg points the way the linear path goes, so no alignment
can pair them up. (A zigzag that alternates between up-right and down-right
legs would not show this reliably, because the alignment can pair the up-right
legs with the linear path.)

## What do the metrics look like for random scanpaths?

To build intuition about each metric's range and distribution, let's simulate
500 pairs of random scanpaths and plot the results:

```python
rng = np.random.default_rng(42)
records = []
fgs_list = []

for _ in range(500):
    fg_a = fixation_group(
        rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
        np.round(rng.uniform(0, 10, 10)).astype(float) + 1,
        np.arange(1, 11, dtype=float),
    )
    fg_b = fixation_group(
        rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
        np.round(rng.uniform(0, 10, 10)).astype(float) + 1,
        np.arange(1, 11, dtype=float),
    )
    fgs_list.append((fg_a, fg_b))
    records.append(multi_match(scanpath(fg_a), scanpath(fg_b), (500, 500)))

ssdf = pd.DataFrame(records)
```

```python
#| fig-width: 8
#| fig-height: 8
#| out-width: "100%"
metrics = ["mm_vector", "mm_direction", "mm_length",
           "mm_position", "mm_duration", "mm_position_emd"]
labels = ["Vector", "Direction", "Length", "Position", "Duration", "EMD"]
colors = ["blue", "green", "red", "purple", "orange", "cyan"]

fig, axes = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)
for ax, m, lbl, c in zip(axes.ravel(), metrics, labels, colors):
    ax.hist(ssdf[m].dropna(), bins=40, color=c, alpha=0.7)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"{lbl} Similarity")
plt.show()
```

None of the metrics sits near zero for unrelated scanpaths. Their medians over
the random pairs are the baseline that a single similarity value has to be
read against:

```python
ssdf[metrics].median().round(2).sort_values()
```

Direction is a useful example. Two independent saccade directions differ by
an angle that is uniform on [0, π], which would put direction similarity near
0.5. But MultiMatch first aligns the two scanpaths, pairing each saccade with
the one in the other scanpath that makes the vector difference smallest, and
that alignment favours saccades pointing the same way. Comparing saccade *i*
with saccade *i*, without the alignment, does give about 0.5:

```python
def unaligned_direction(fg_a, fg_b):
    # saccade directions; the last row of a scanpath has no outgoing saccade
    theta_a = scanpath(fg_a)["theta"].to_numpy()[:-1]
    theta_b = scanpath(fg_b)["theta"].to_numpy()[:-1]
    # absolute angle between saccade i and saccade i, in [0, pi]
    diff = np.abs(np.angle(np.exp(1j * (theta_a - theta_b))))
    return 1 - np.median(diff) / np.pi

pd.Series({
    "aligned (mm_direction)": ssdf["mm_direction"].median(),
    "unaligned": np.median([unaligned_direction(a, b) for a, b in fgs_list]),
}).round(2)
```

## What do extreme matches look like?

The histograms are useful, but the fastest way to understand each metric is to
inspect the scanpath pairs at the extremes. Here we find, for each metric, the
highest-scoring and lowest-scoring pair from the random simulation above:

```python
metric_names = ["vector", "direction", "length", "position", "duration", "position_emd"]

extremes = {
    metric: {
        "high": ssdf[f"mm_{metric}"].idxmax(),
        "low": ssdf[f"mm_{metric}"].idxmin(),
    }
    for metric in metric_names
}
```

```python
#| fig-width: 8
#| fig-height: 11
#| out-width: "100%"
def plot_fixation_path(ax, fg, title, score):
    tab = fg.to_pandas(copy=True)
    ax.plot(tab["x"], tab["y"], color="#4f46e5", linewidth=1.4, marker="o", markersize=3.5)
    ax.scatter(tab["x"].iloc[0], tab["y"].iloc[0], color="#16a34a", s=36, zorder=3)
    ax.scatter(tab["x"].iloc[-1], tab["y"].iloc[-1], color="#dc2626", s=36, zorder=3)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nscore = {score:.2f}", fontsize=9)
    ax.tick_params(labelsize=7)


def plot_extreme_grid(which):
    fig, axes = plt.subplots(len(metric_names), 2, figsize=(8, 11), constrained_layout=True)

    for row, metric in enumerate(metric_names):
        idx = extremes[metric][which]
        fg_a, fg_b = fgs_list[idx]
        score = ssdf.loc[idx, f"mm_{metric}"]

        plot_fixation_path(axes[row, 0], fg_a, f"{metric}: path A", score)
        plot_fixation_path(axes[row, 1], fg_b, f"{metric}: path B", score)

    fig.suptitle(f"{which.title()}-scoring scanpath pairs by metric", fontsize=12)
    return fig


plot_extreme_grid("high")
plt.show()
```

The highest-scoring pairs are not visibly similar scanpaths: with ten random
fixations each, a pair can score highly on one dimension (for example, similar
saccade lengths) while differing everywhere else. This is why the six measures
are reported separately rather than averaged.

```python
#| fig-width: 8
#| fig-height: 11
#| out-width: "100%"
plot_extreme_grid("low")
plt.show()
```

The lowest-scoring pairs are not visibly more different than the others either:
with ten random fixations, a low score on one dimension says little about the
rest. (These plots do not show fixation durations.)

## How do transformations affect the metrics?

Understanding how each metric responds to spatial and temporal transformations
helps you choose the right one for your research question.

### Identity (perfect match)

```python
rng = np.random.default_rng(7)
fg = fixation_group(
    rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
    np.round(rng.uniform(0, 10, 10)).astype(float) + 1,
    np.arange(1, 11, dtype=float),
)
print("Identity:")
multi_match(scanpath(fg), scanpath(fg), (500, 500))
```

All metrics are 1.0 — a scanpath is perfectly similar to itself.

### Scaling

Scaling the coordinates by 0.5 preserves **direction** (relative
angles are unchanged) but reduces **position** (absolute locations differ):

```python
fg_scaled = fg.copy()
fg_scaled["x"] = fg_scaled["x"] * 0.5
fg_scaled["y"] = fg_scaled["y"] * 0.5

print("Scaling (0.5x):")
multi_match(scanpath(fg), scanpath(fg_scaled), (500, 500))
```

As expected, **direction** stays at 1.0, but **position** drops.

### Shuffling fixation order

Keeping the same fixation locations but scrambling their order (and their
durations) separates the order-insensitive EMD from the sequence-based
metrics:

```python
rng2 = np.random.default_rng(3)
fg_shuffled = fg.copy()
order = rng2.permutation(len(fg))
fg_shuffled["x"] = fg["x"].values[order]
fg_shuffled["y"] = fg["y"].values[order]
fg_shuffled["duration"] = fg["duration"].values[order]

print("Order-shuffled:")
multi_match(scanpath(fg), scanpath(fg_shuffled), (500, 500))
```

Only **position_emd** is unchanged: it compares the two sets of fixations
regardless of order, and the set is the same. **Position** drops as well as
**direction**, because MultiMatch's position compares the start points of
aligned saccades in sequence, and reordering moves them. **Duration** drops
because the durations were shuffled with the locations.

## References

Dewhurst, R., Nyström, M., Jarodzka, H., Foulsham, T., Johansson, R., &
Holmqvist, K. (2012). It depends on how you look at it: Scanpath comparison in
multiple dimensions with MultiMatch, a vector-based approach. *Behavior
Research Methods*, 44, 1079–1100.
