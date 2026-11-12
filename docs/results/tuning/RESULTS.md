# Held-out Optuna and replay results

The comparison uses 17 tracks, both experts and 10 paired training seeds.
Replay ratios are offline/online. Lower capped time is better.

| SAC replay split | Finished laps | Capped mean | Mean normalized time | Updates per run |
| --- | ---: | ---: | ---: | ---: |
| 0/100 | 2572/3400 | 96.89 s | 178.07% | 8,192 |
| 25/75 | 2776/3400 | 88.11 s | 162.03% | 8,192 |
| 50/50 | 2493/3400 | 101.42 s | 199.11% | 8,192 |
| 75/25 | 2299/3400 | 109.48 s | 215.83% | 8,192 |
| 100/0 | 493/3400 | 181.49 s | 347.90% | 8,192 |

| Other training condition | Finished laps | Capped mean | Mean normalized time | Updates per run |
| --- | ---: | ---: | ---: | ---: |
| Tuned DAgger | 3332/3400 | 62.39 s | 109.53% | 6,144 |
| Original DAgger | 3301/3400 | 63.82 s | 112.10% | 3,072 |
| Original SAC 50/50 | 1693/3400 | 132.16 s | 253.83% | 4,096 |

Each difference is in percentage points of the per-track reference time. Negative favors the first condition.

| Paired comparison | Mean difference | 95% bootstrap interval |
| --- | ---: | ---: |
| 0/100 vs 50/50 | -21.04 | [-30.65, -11.82] |
| 25/75 vs 50/50 | -37.08 | [-47.37, -25.91] |
| 75/25 vs 50/50 | +16.72 | [+6.63, +26.82] |
| 100/0 vs 50/50 | +148.79 | [+136.98, +160.37] |
| Tuned 50/50 vs tuned DAgger | +89.58 | [+79.86, +100.00] |
| Tuned DAgger vs original | -2.57 | [-7.68, +1.60] |
| Tuned 50/50 vs original | -54.72 | [-72.66, -37.66] |
| 25/75 vs 0/100 | -16.04 | [-26.60, -4.42] |
| 25/75 vs tuned DAgger | +52.50 | [+43.47, +62.47] |

The intervals resample paired seed blocks. The ratio contrasts and comparisons involving the best observed ratio are exploratory.
All ratios use the same SAC settings selected at 50/50. At 100/0 the online rollouts preserve the update cadence and supply zero gradient samples.
