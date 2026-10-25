---
title: "Revisitation"
description: "Returns to previously fixated locations within one viewing, compared against a spatial null."
---

Similarity compares one viewing with another. Revisitation looks inside a
single viewing: how often does gaze return to a location it has already
fixated? Revisits during study have been linked to the hippocampus and to later
memory (Voss et al., 2011; Kragel et al., 2021). `fixation_revisits()` scores
them fixation by fixation.

## Scoring one viewing

A fixation is a revisit when it lands within `radius` of an earlier fixation
at least `min_lag` fixations back (default 2, so the fixation immediately
before never counts). By default gaze must also have left the location first:
the preceding fixation has to be at least `radius` away. A fixation that could
have been a revisit under these rules is an *opportunity*.

In this sequence the fourth fixation returns to the first location:

```python
import numpy as np
import pandas as pd
from peyesim import fixation_group, fixation_revisits, revisit_summary

fg = fixation_group(x=[100, 400, 400, 105], y=[100, 100, 400, 98],
                    onset=[0, 250, 500, 750], duration=[200] * 4)
rv = fixation_revisits(fg, radius=50)
rv
```

`lag` counts fixations back to the matched fixation, and `distance` is the
distance to the nearest eligible earlier fixation. `revisit_summary()` gives
one row per viewing:

```python
revisit_summary(rv)
```

`radius` is in the units of the fixation coordinates. To set it in degrees of
visual angle, convert with `visual_angle_to_px()` for your display:

```python
from peyesim import visual_angle_to_px

# 52 cm wide, 1920 px display viewed from 60 cm
visual_angle_to_px(2, distance=60, screen_size=52, screen_res=1920)
```

## Comparing with a spatial null

Raw revisit counts partly restate how many fixations a viewer made and how
tightly they clustered: more fixations in a smaller area give more returns by
chance. A spatial null estimates how many revisits to expect from where people
look. For each opportunity, it takes the fraction of reference locations that
would also have counted as a revisit.

The `EyeTable` method scores every row and builds the reference:

- `null="item"` uses fixations from other viewers of the same image
  (`item_var`). With `subject_var` given, all of the viewer's own rows are
  left out. This keeps where the image draws gaze, but not the viewer's own
  bias.
- `null="subject"` uses the same viewer's other rows. This keeps the viewer's
  spatial bias, but not the image content.

Simulate four viewers who each see the same six images, with random fixations:

```python
from peyesim import simulate_eye_table

np.random.seed(1)
et = simulate_eye_table(n_fixations=24 * 20, n_groups=24, clip_bounds=(0, 800, 0, 600))
et["subject"] = np.repeat([f"s{i}" for i in range(1, 5)], 6)
et["image"] = np.tile(list("abcdef"), 4)

scored = fixation_revisits(et, radius=60, null="item", item_var="image",
                           subject_var="subject")
cols = ["subject", "image", "n_fix", "opportunities", "revisits",
        "expected_revisits", "excess_revisits"]
scored.to_pandas()[cols].head(8)
```

`excess_revisits` is the observed count minus the expected count, and
`excess_rate` is the same difference per opportunity. Rows with
no opportunities give NaN, as do rows whose image no other viewer saw. The simulated fixations are random, so the excess
should be centred on zero:

```python
from scipy import stats

stats.ttest_1samp(scored["excess_rate"].dropna(), 0)
```

```python
import matplotlib.pyplot as plt

tab = scored.to_pandas()
fig, ax = plt.subplots(figsize=(4.5, 4.2))
lim = max(tab["expected_revisits"].max(), tab["revisits"].max()) + 1
ax.plot([0, lim], [0, lim], linestyle="--", color="grey", linewidth=1)
ax.scatter(tab["expected_revisits"], tab["revisits"], alpha=0.7)
ax.set_xlabel("Expected revisits (item null)")
ax.set_ylabel("Observed revisits")
ax.set_aspect("equal")
fig.tight_layout()
```

Score one phase of an experiment at a time; otherwise a subject null would
include the same image from the other phase. Pass `keep_fixations=True` to
keep each row's fixation-level table in a `revisit_fixations` column, and
`max_lag` to limit how far back a return can reach. Re-scoring a scored table
(for example at another radius) replaces the columns the earlier call wrote.

## References

Voss, J. L., Warren, D. E., Gonsalves, B. D., Federmeier, K. D., Tranel, D.,
& Cohen, N. J. (2011). Spontaneous revisitation during visual exploration as a
link among strategic behavior, learning, and the hippocampus. *PNAS*, 108(31),
E402–E409.

Kragel, J. E., Schuele, S., VanHaerents, S., Rosenow, J. M., & Voss, J. L.
(2021). Rapid coordination of effective learning by the human hippocampus.
*Science Advances*, 7(25), eabf7144.
