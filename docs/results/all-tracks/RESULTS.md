# DAgger and symmetric replay in solo SuperTuxKart

The study covers 21 tracks, 2 demonstration sources, and 10 training seeds per condition at 16,384 online steps. Each policy ran on 10 matched starts. The overall symmetric SAC minus DAgger gap was **+120.47% [+112.71, +128.83]** of Team L’Eclair’s solo capped lap time (mean and 95% paired bootstrap interval). Positive differences favor DAgger. Each track and source receives equal weight.

## Completion, controller comparisons, and variability

| Demonstrations | Learner | Finishes | Mean vs Team 3 native | Tracks faster than best native / restricted controller | Median within-track SD | Across-track SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Team 3 | Behavior cloning | 1937/2100 | +20.18% | 1 / 3 | 3.49% | 33.84% |
| Team 3 | DAgger | 2035/2100 | +8.13% | 3 / 2 | 1.97% | 18.80% |
| Team 3 | Symmetric SAC | 1109/2100 | +139.89% | 0 / 0 | 54.36% | 128.25% |
| Team 3 | Online-only SAC | 931/2100 | +171.02% | 0 / 0 | 83.30% | 153.47% |
| Team 4 | Behavior cloning | 1937/2100 | +21.83% | 0 / 0 | 1.72% | 45.94% |
| Team 4 | DAgger | 1970/2100 | +17.32% | 0 / 0 | 1.13% | 44.39% |
| Team 4 | Symmetric SAC | 1228/2100 | +126.50% | 0 / 0 | 28.01% | 143.36% |
| Team 4 | Online-only SAC | 870/2100 | +176.87% | 0 / 0 | 82.02% | 152.42% |

Controller wins use lower mean capped time on the fixed starts. The two SD columns describe variation across training seeds within a track and variation of normalized means across tracks. Read both alongside completion, since a timeout contributes the full capped time.

## Effect of the demonstration source

| Learner | Team 4 minus Team 3 |
| --- | ---: |
| Behavior cloning | +1.65% [-6.12, +10.26] |
| DAgger | +9.19% [+2.53, +15.00] |
| Symmetric SAC | -13.39% [-27.43, +1.22] |
| Online-only SAC | +5.85% [-5.07, +16.40] |

Switching from Team 3 to Team 4 changed the symmetric SAC minus DAgger gap by -22.58% [-36.97, -6.81]. The symmetric SAC minus online-only SAC contrast was -40.75% [-53.84, -28.81]. Both comparisons use the same normalization and paired training seeds.

## Track-level results

| Track | Symmetric SAC minus DAgger |
| --- | ---: |
| `abyss` | +8.40% [-5.26, +34.33] |
| `black_forest` | +62.25% [+55.13, +68.08] |
| `candela_city` | +36.34% [+19.59, +56.18] |
| `cocoa_temple` | +42.88% [+8.24, +85.53] |
| `cornfield_crossing` | +104.47% [+76.46, +127.47] |
| `fortmagma` | +104.41% [+96.42, +110.23] |
| `gran_paradiso_island` | +19.98% [+2.69, +40.35] |
| `hacienda` | +190.75% [+125.29, +249.65] |
| `lighthouse` | +14.17% [+2.44, +36.29] |
| `mines` | +222.48% [+206.90, +235.74] |
| `minigolf` | +349.18% [+323.07, +372.96] |
| `olivermath` | +385.00% [+269.73, +494.27] |
| `ravenbridge_mansion` | +110.96% [+66.85, +154.48] |
| `sandtrack` | -3.30% [-4.59, -1.86] |
| `scotland` | +34.17% [+7.62, +67.64] |
| `snowmountain` | +22.66% [-22.31, +65.47] |
| `snowtuxpeak` | +257.75% [+220.05, +291.95] |
| `stk_enterprise` | +171.69% [+152.66, +185.34] |
| `volcano_island` | +1.70% [-0.60, +5.12] |
| `xr591` | +10.14% [-14.79, +37.97] |
| `zengarden` | +383.71% [+350.48, +413.79] |

The across-track SD of the mean method gaps was 130.01% of the reference time. The track-level intervals are exploratory.

[Paired track effects](track-effects.png) · [Training-seed variance](training-variance.png) · [Per-track learner metrics](track-methods.csv) · [Solo controller baselines](controller-baselines.csv) · [Learner vs controller comparisons](learner-vs-controller.csv)
