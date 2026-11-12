# Controller baselines across 21 SuperTuxKart tracks

All five controllers ran one solo lap on 10 matched starts per track. Each controller used the same Tux kart, a 200 s cap, and two action configurations. Native driving enables brake, drift, nitro, and rescue. Restricted driving executes throttle and steering, the learners' action space. Both configurations disable firing and hide held power-ups. The historical five-kart profile used shuffled starting grids. [Per-track solo results](controller-baselines.csv) · [Historical and solo results](historical-controller-context.csv)

| Controller | Historical five-kart finishes | Historical capped mean | Solo native finishes | Solo native capped mean | Solo restricted finishes | Solo restricted capped mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tasty Crousteam | 208/210 | 59.65 s | 205/210 | 60.48 s | 198/210 | 65.16 s |
| DemoPilote | 209/210 | 62.02 s | 209/210 | 59.51 s | 190/210 | 71.62 s |
| Team L'Eclair | 209/210 | 60.29 s | 210/210 | 57.77 s | 195/210 | 66.79 s |
| The Winners | 210/210 | 58.36 s | 210/210 | 56.85 s | 194/210 | 67.94 s |
| Donkey Bombs | 203/210 | 69.11 s | 207/210 | 63.53 s | 189/210 | 74.99 s |

The Winners had the lowest capped mean in both the historical and native solo benchmarks. In the restricted solo runs, Tasty Crousteam had the lowest overall mean and Team L’Eclair had the lowest capped mean on 11 tracks. Both expert sources finished every native solo lap. In the restricted condition, both lost completions and their relative ranking changed.

On Fort Magma, The Winners finished **10/10 at 59.63 s** with native driving and **0/10 at the 200 s cap** when restricted. Team L’Eclair finished **10/10 at 66.51 s** with native driving and **8/10 with an 82.18 s capped mean** when restricted. This difference carries over to the fixed demonstrations. None of The Winners' five collection episodes on Fort Magma finished, and none of Team L’Eclair’s five collection episodes on STK Enterprise finished. The datasets retain those transitions under the same 8,192-transition rule used on every track. Report these two cells when discussing how the expert source affects both learners. [Fort Magma laps](reference/fortmagma/controllers.json) · [STK Enterprise laps](reference/stk_enterprise/controllers.json)

The median of the 21 per-track capped-time standard deviations in native solo driving was 1.48 s for Tasty Crousteam, 0.93 s for DemoPilote, 2.25 s for Team L’Eclair, 0.36 s for The Winners, and 1.65 s for Donkey Bombs. These values measure sensitivity to the fixed starts. The learners’ within-track standard deviations also include the effect of training seed, which the learner analysis reports separately. Historical standard deviations were computed over finishes in the five-kart runs. [Historical controller aggregates](historical-controller-context.csv)
