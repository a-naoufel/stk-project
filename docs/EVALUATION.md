# STK evaluation — 7 November 2026

**Seed 8 is the strongest of the 14 PPO models, but the heuristic completes more races on this suite.** The built-in AI is the strongest reference. The PPO models have learned useful driving, but reliability remains the main limitation.

## Method

- Benchmark: all 14 final checkpoints plus the heuristic and built-in AI, on all 21 available race tracks and seeds 100–104: **1,680 races**.
- Validation: the three leading PPO checkpoints (seeds 8, 15 and 12) plus both baselines, on the same tracks with fresh seeds 200–204: **525 races**.
- Total: **2,205 complete evaluation records**, with one lap, four karts, difficulty 2 and a 1,500-step limit. A timeout counts as a failure.
- Learned actions use deterministic CPU inference. Rank models by completion rate, then mean steps with failures capped at 1,500, then mean final position.
- The top three were fixed before validation began. The selection record proves unfinished benchmark runs could not reach their completion totals. Training used random tracks from this same track pool; the validation holds out seeds, not track identities.
- Each checkpoint trained independently for 1,000,448 steps. The backup includes its inference weights, resumable PPO model, monitor CSV and training configuration.

## Fresh-seed validation

| Agent | Finished | Completion | Mean steps including timeouts | Mean final position |
| --- | ---: | ---: | ---: | ---: |
| ai | 105/105 | 100.0% | 568.0 | 2.61 |
| heuristic | 100/105 | 95.2% | 819.4 | 3.99 |
| ppo-seed-08 | 91/105 | 86.7% | 727.7 | 3.70 |
| ppo-seed-15 | 81/105 | 77.1% | 831.6 | 3.66 |
| ppo-seed-12 | 76/105 | 72.4% | 864.3 | 3.68 |

![Fresh-seed validation](../outputs/ppti-evaluation/validation-20261107/comparison.png)

Seed 8 has fewer mean capped steps than the heuristic (727.7 versus 819.4), but its completion rate is 8.6 percentage points lower. This is a tradeoff between driving speed and reliability. Mean final position includes unfinished races and should not be read as finishing position on completed races only.

A paired bootstrap over whole tracks gives a 95% interval of -26.7 to +9.5 percentage points for seed 8 minus the heuristic. That interval includes zero: the observed gap is clear for this fixed suite, but it does not establish a statistically certain gap on arbitrary tracks. Intervals use 5,000 resamples, preserve matched seeds, and are recorded in the JSON summaries.

## Track failures

Seed 8 has 180 completions out of 210 races across both phases. Its 30 failures are concentrated as follows:

| Track | Seed 8 finished | Heuristic finished |
| --- | ---: | ---: |
| cocoa_temple | 9/10 | 10/10 |
| fortmagma | 0/10 | 10/10 |
| snowtuxpeak | 0/10 | 10/10 |
| stk_enterprise | 2/10 | 10/10 |
| zengarden | 9/10 | 10/10 |

The heuristic fails all ten Black Forest races across both phases; seed 8 completes all ten. Every PPO model fails all benchmark races on Fort Magma and Snow Tux Peak, and the three leading models also fail all fresh-seed races on those tracks. These repeatable track failures are better targets for the next experiments than increasing the training budget without inspecting behavior. The results alone do not identify whether observation design, control restrictions, exploration, or recovery behavior is the main cause.

![Benchmark track completion](../outputs/ppti-evaluation/benchmark-20261107/track_completion.png)

## All trained models — benchmark

| Training seed | Finished / 105 | Completion | Mean capped steps |
| --- | ---: | ---: | ---: |
| 8 | 89 | 84.8% | 747.5 |
| 15 | 81 | 77.1% | 827.1 |
| 12 | 76 | 72.4% | 862.7 |
| 0 | 75 | 71.4% | 878.0 |
| 14 | 73 | 69.5% | 890.8 |
| 6 | 70 | 66.7% | 918.6 |
| 7 | 70 | 66.7% | 922.7 |
| 4 | 69 | 65.7% | 933.1 |
| 5 | 64 | 61.0% | 973.6 |
| 10 | 59 | 56.2% | 1033.3 |
| 13 | 54 | 51.4% | 1058.6 |
| 11 | 51 | 48.6% | 1129.4 |
| 3 | 49 | 46.7% | 1096.1 |
| 9 | 35 | 33.3% | 1216.4 |

## Runtime and submission compatibility

- Seed 8 averaged **0.137 ms** per action in validation, with a measured maximum of **1.156 ms**.
- The instructor’s flat-upload loader successfully loaded the seed-8 weights and completed its 100-step Lighthouse check: mean/max action time 0.2/0.7 ms and reported peak memory 437 MB. This check used a temporary folder and did not upload anything.
- The built-in AI uses the full game action set. The learned policies and heuristic control acceleration and steering only. Built-in AI computation occurs inside simulator steps, so its inference latency is not directly comparable and is omitted.
- These races use three built-in opponents. They do not measure ML-Arena Elo or head-to-head competition between the 14 learned models. No deployed policy was changed.

## Saved artifacts

- [Seed 8 inference weights](../outputs/ppti-evaluation/benchmark-20261107/models/seed-08/pystk_actor.pth)
- [Seed 8 resumable PPO model](../outputs/ppti-evaluation/benchmark-20261107/models/seed-08/ppo.zip)
- [Full benchmark report](../outputs/ppti-evaluation/benchmark-20261107/report.md) and [leaderboard CSV](../outputs/ppti-evaluation/benchmark-20261107/leaderboard.csv)
- [Fresh-seed report](../outputs/ppti-evaluation/validation-20261107/report.md) and [per-race CSV](../outputs/ppti-evaluation/validation-20261107/races.csv)
- Both phase folders also contain fixed protocols, JSON summaries, raw race records, checkpoint hashes, exact runtime versions and standalone PNG/SVG comparison figures.

Seed 8 checkpoint SHA-256:

`ab8fe24b5d44b0b954dbbbc3a86898d889eb2f878e2ccdf7389ef9afa6071ed5`

## Reproduce

Run from the local project directory; SSH uses the existing PPTI gateway configuration.

```bash
python3 scripts/evaluate_ppti.py \
  --manifest outputs/ppti-training/ppo-20261106-220242/manifest.json \
  --phase benchmark-20261107 --jobs 14

python3 scripts/evaluate_ppti.py \
  --manifest outputs/ppti-training/ppo-20261106-220242/manifest.json \
  --phase validation-20261107 \
  --hosts ppti-14-407-09,ppti-14-407-16,ppti-14-407-13 \
  --seeds 200,201,202,203,204 --jobs 3

python3 scripts/analyze_evaluation.py outputs/ppti-evaluation/benchmark-20261107
python3 scripts/analyze_evaluation.py outputs/ppti-evaluation/validation-20261107
```

The analysis commands use the local Python installation with NumPy and Matplotlib. The remote simulator and policy run inside the locked PPTI environment.
