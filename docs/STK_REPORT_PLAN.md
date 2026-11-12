# STK report writing plan

Use the [21-track study protocol](STK_ALL_TRACKS_STUDY.md) for the expanded report. The outline below describes the earlier Lighthouse pilot.

**Working title:** DAgger and symmetric replay for solo SuperTuxKart time trials

Use [the existing study draft](STK_STUDY.md) as the shared manuscript and [the experimental protocol](STK_EXPERIMENT_PROTOCOL.md) as the source for method settings and analysis decisions. The current runs support the report. A fresh-seed repeat is a separate experiment if the team chooses to run it.

## Divide the writing

| Owner | First draft | Review responsibility |
| --- | --- | --- |
| Teammate A | Introduction, related work, simulator, expert dataset, and DAgger | Recompute the main lap-time table from the saved results and check that the text uses the right budget and sample count. |
| Teammate B | SAC and the replay ablation, evaluation design, statistics, and results figures | Check every environment setting and hyperparameter in the Methods section against the code or run metadata. |
| Both | Discussion, limitations, and conclusion | Agree on the claims and write the abstract together after the figures and results are final. |

Start by assigning names to these roles and agreeing on the primary comparison. Then draft Methods and Results in parallel, trade reviews, and merge the sections into one manuscript.

## Report outline

| Section | What to write | Evidence |
| --- | --- | --- |
| Abstract | State the question, solo Lighthouse setup, two online budgets, and how the gap changed. Give the effect sizes in seconds. | Results table and paired intervals. |
| Introduction and related work | Explain why a driver that imitates an expert can encounter unfamiliar states. Contrast DAgger’s live expert queries with the fixed offline data used in symmetric replay. Explain why the comparison matches online interaction budgets. | [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html) · [RLPD paper](https://proceedings.mlr.press/v202/ball23a.html) |
| Experimental methods | Describe the one-kart, one-lap setup, shared features and actions, expert collection, common BC initialization, three training conditions, seeds, budgets, and timeout rule. | [Protocol and hyperparameters](STK_EXPERIMENT_PROTOCOL.md) |
| Results | Lead with DAgger versus symmetric SAC at 16,384 and 65,536 steps. Show individual seed differences and the interval for their mean. Use BC and online-only SAC to explain what changed relative to initialization and what adding replay contributed within SAC. | [16,384-step results](../outputs/stk-study-20261108/summary.json) · [65,536-step results](../outputs/stk-study-20261108-65k/summary.json) |
| Recovery and replay | Report the matched grid of shifted and misaligned starts. Use seed 3 to illustrate one observed source of time loss. | [Recovery results](../outputs/stk-study-20261108-65k/recovery/summary.json) · [Seed 3 telemetry](../outputs/stk-study-20261108-65k/recovery/replays/seed-03-telemetry.png) |
| Discussion and conclusion | Explain what the data support for this course, dataset, and training setup. Discuss the strong BC baseline, the role of expert access, and uncertainty at the larger budget. | The paired intervals, completion counts, and replay. |
| Reproducibility appendix | Include the environment versions, exact seeds, hyperparameters, expert revision, dataset checksum, commands, and location of the per-race outputs. | [Reproduction steps](STK_EXPERIMENT_PROTOCOL.md#reproduce-and-audit) |

## Results to write around

| Test | BC | DAgger | Symmetric SAC | Online-only SAC |
| --- | ---: | ---: | ---: | ---: |
| 16,384 online steps | 38.31 s | 38.42 s | 40.51 s | 45.16 s |
| 65,536 online steps | 38.31 s | 38.28 s | 38.69 s | 40.89 s |
| Recovery starts at 65,536 steps | 38.32 s | 38.36 s | 38.78 s | — |

The mean gap, defined as symmetric SAC minus DAgger, was **+2.09 s** at 16,384 steps (95% paired interval +1.54 to +2.75) and **+0.41 s** at 65,536 steps (−0.43 to +1.73). The longer-budget interval supports differences in either direction. Symmetric replay improved the matched SAC control at both budgets. The recovery grid produced **160/160 finishes per evaluated method**, so use it to discuss lap time and robustness over the tested poses.

Use one main table for completion and mean lap time, then put the paired effects for both budgets in the main figure. The replay plot is a secondary figure. The largest seed 3 delays occurred near 400–500 m and 600–700 m when symmetric SAC went onto the grass. Present this as a case study of that policy.

## Before handing it in

- [ ] Each number in the Results section can be traced to a saved summary or raw lap record.
- [ ] The analysis counts eight training seeds per budget. The laps within a seed are repeated evaluations of the same policy.
- [ ] The comparison is named **DAgger versus SAC with symmetric sampling**. The SAC implementation uses RLPD’s sampling rule, with standard SB3 critics and updates.
- [ ] The discussion explains the common warm start and the difference between a fixed offline dataset and live expert queries.
- [ ] Figures show units and sample counts. All cited figures and data links are accessible to the report reader. Run outputs are ignored by Git and must be shared separately if cited.
- [ ] ML Arena, if included, is a short appendix about the separate multiplayer deployment. The experiment’s conclusions come from the solo time trials.
