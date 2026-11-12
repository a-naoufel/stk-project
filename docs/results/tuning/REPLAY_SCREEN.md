# Development replay screen

The comparison uses 4 tracks, both experts and 2 paired training seeds.
Replay ratios are offline/online. Lower capped time is better.

| SAC replay split | Finished laps | Capped mean | Mean normalized time | Updates per run |
| --- | ---: | ---: | ---: | ---: |
| 0/100 | 92/160 | 116.58 s | 189.60% | 4,096 |
| 25/75 | 118/160 | 94.63 s | 155.61% | 4,096 |
| 50/50 | 106/160 | 104.41 s | 169.03% | 4,096 |
| 75/25 | 106/160 | 102.89 s | 166.73% | 4,096 |
| 100/0 | 30/160 | 170.30 s | 288.83% | 4,096 |

Each difference is in percentage points of the per-track reference time. Negative favors the first condition.

| Paired comparison | Mean difference | 95% bootstrap interval |
| --- | ---: | ---: |
| 0/100 vs 50/50 | +20.57 | — |
| 25/75 vs 50/50 | -13.42 | — |
| 75/25 vs 50/50 | -2.30 | — |
| 100/0 vs 50/50 | +119.80 | — |

This development screen uses the original SAC settings and two training seeds. Results are descriptive.
