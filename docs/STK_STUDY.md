# DAgger and symmetric sampling in SuperTuxKart

**DAgger was 2.09 seconds faster on solo Lighthouse at 16,384 online steps.** It led in all eight paired seeds. At 65,536 steps, the mean gap fell to 0.41 seconds and each method led in four seeds. Both methods finished every test lap at both budgets.

## Task and expert data

One Tux kart drives one lap of Around the Lighthouse in SuperTuxKart's `TIME_TRIAL` mode. A neutral command advances the starting countdown to GO before the learner takes control. Every trial uses the same kart, course direction, difficulty 2, and 0.1-second simulation step. All controllers use continuous throttle and steering. Brake, drift, nitro, rescue commands, and item firing are disabled.

The expert is Team 3's `Agent3` from commit `ddf2cb6573850f890bba26d4e5f82888d9dac82a`. The adapter clears item observations and projects the expert action onto the shared throttle and steering controls. Its action is recomputed for the current state whenever DAgger queries it. The code checks hashes of the five expert source files it loads. [Approved Team 3 source at the pinned revision](https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team3)

The fixed dataset consists of 12,224 transitions from 32 expert laps. All 32 finished, averaging 38.173 s. Demonstrations use race seeds 12300–12331 and a NumPy RNG seeded with 12300 to draw lateral grid offsets uniformly from −1.5 m to +1.5 m. Offsets are applied at GO with the original heading and zero speed. Observation traces from all 32 laps have distinct hashes. The dataset stores observations, actions, next observations, rewards and terminal flags. [Expert lap records](../outputs/stk-study-20261108/expert/expert.json)

## Matched training protocol

Every learner uses the same 32 bounded inputs: five upcoming path endpoints, their widths and validity masks, a vector to the track centre, kart-local velocity, and energy. The actor has two ReLU layers with 64 units each and a tanh-squashed output. The two normalized action components map to throttle in [0, 1] and steering in [−1, 1].

Each paired training seed starts from the same behavior-cloned actor. BC uses 2,048 Adam updates at learning rate 0.001, batch size 128, and mean squared error on the deterministic squashed action. It samples uniformly with replacement from the fixed expert dataset. The SAC actor's initial log standard deviation has zero weights and bias −2. Each of the three methods then receives exactly 16,384 online environment steps. The training environment samples lateral offsets uniformly from −1.5 m to +1.5 m using a separate RNG seeded by the learner seed. A pair shares the same start sequence by episode index.

| Method | Online learning | Training hyperparameters |
| --- | --- | --- |
| DAgger | Queries Team 3 at every visited state. Acts with a mixture of the expert and current deterministic learner. Appends the expert labels to the initial demonstrations and fits on the accumulated data. | 8 rounds of 2,048 online steps. Expert probability 0.5 in the first round and halved each round, ending at 0.00390625. After each round, 384 supervised updates with the BC optimizer, batch size and loss. |
| SAC with symmetric sampling | Learns from Bellman targets using the fixed expert data and its own stochastic rollouts. | Each batch has 64 offline and 64 online transitions, sampled with replacement. |
| Online-only SAC | Uses the same SAC training with batches drawn from online experience. | Each batch has 128 online transitions. The fixed data enters through the common BC initialization. |

Both SAC variants use Stable-Baselines3 SAC 2.9.0 with twin ReLU critics of widths 64 and 64, Adam, learning rate 0.0001, discount 0.995, target coefficient 0.005, fixed entropy coefficient 0.02, one-step targets, and a target update after each gradient update. They sample the actor from the first step and run one gradient update per four online steps, for 4,096 updates. Online replay capacity is 16,384 transitions. Each symmetric run drew 262,144 samples from each source. Each DAgger run made 16,384 live expert queries, with expert actions executed on 2,003–2,091 steps across seeds.

SAC's reward gives 0.1 per metre of verified new on-track progress, a cost of 0.02 per racing step, another 0.08 off track, and a bonus of 10 for finishing. Progress is bounded by physical motion and corrects for the wrap at the start line. The timeout is treated as a truncation for SAC bootstrapping. DAgger learns from the action labels alone. [Reward implementation](../stk_actor/learn.py)

DAgger addresses the change in state distribution when a learner drives itself by asking the expert for actions in the states the learner reaches. Symmetric sampling instead gives the offline dataset a constant share of each RL batch as online experience grows. This experiment uses the 50/50 sampling component of RLPD with standard SB3 SAC. The original RLPD work also studies LayerNorm, larger critic ensembles, and higher update-to-data ratios. [Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html), [Ball et al., 2023](https://proceedings.mlr.press/v202/ball23a.html)

## Results at 16,384 online steps

The protocol fixes eight training seeds (0–7), the final checkpoint, and eleven evaluation starts. All methods use the same starts, pairing race seeds 50000–50010 with lateral offsets of −2, −1.75, −1.25, −0.75, −0.25, 0, 0.25, 0.75, 1.25, 1.75, and 2 m. Four starts lie outside the training offset range. Evaluation uses deterministic actions and the simulator's finish time. Each episode allows 1,500 control steps. A timeout would be scored as 150 seconds and counted as a non-finish. No evaluation results select checkpoints. [Frozen protocol](STK_STUDY_PROTOCOL.json)

| Policy | Laps completed | Mean capped lap time |
| --- | ---: | ---: |
| Team 3 expert | 11/11 | 38.17 s |
| Shared BC initialization | 88/88 | 38.31 s |
| **DAgger** | **88/88** | **38.42 s** |
| SAC with symmetric sampling | 88/88 | 40.51 s |
| Online-only SAC after BC | 88/88 | 45.16 s |

![Paired lap times and per-seed differences](../outputs/stk-study-20261108/seed-comparison.png)

| Training seed | BC (s) | DAgger (s) | Symmetric SAC (s) | Online-only SAC (s) | Symmetric SAC minus DAgger (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38.265 | 38.407 | 40.518 | 41.772 | 2.111 |
| 1 | 38.198 | 38.386 | 42.359 | 57.949 | 3.974 |
| 2 | 38.573 | 38.548 | 40.467 | 45.309 | 1.919 |
| 3 | 38.289 | 38.499 | 39.757 | 45.006 | 1.258 |
| 4 | 38.281 | 38.213 | 39.620 | 42.066 | 1.407 |
| 5 | 38.436 | 38.439 | 39.657 | 42.433 | 1.218 |
| 6 | 38.281 | 38.214 | 40.099 | 41.231 | 1.885 |
| 7 | 38.174 | 38.666 | 41.578 | 45.522 | 2.912 |

The unit of inference is the training seed. First average the eleven laps within each seed and method. Then subtract the paired DAgger mean from the symmetric SAC mean. The mean difference is +2.086 s, with an approximate percentile bootstrap interval of +1.539 to +2.746 s from 20,000 resamples of the eight paired seeds (bootstrap seed 17). All eight differences are positive. Under the sign test's null of equally likely directions for independent pairs, the two-sided probability is 2/256 = 0.0078125. Repeated starts from a single learner contribute to its mean and do not act as additional independent training runs. [Machine-readable results](../outputs/stk-study-20261108/summary.json)

Symmetric sampling improved SAC by 4.65 s on average and was faster than the online-only control for each seed. The shared BC initialization still averaged 0.11 s faster than final DAgger. Under this protocol, the advantage of DAgger over symmetric SAC therefore coincides with a strong imitation starting point and slower SAC policies after online training.

These estimates are conditional on Lighthouse, the fixed expert dataset, the BC initialization, the two-control interface, and the specified online budget. DAgger also has live expert access, whereas both SAC variants have access to rewards during online learning. The comparison evaluates the two training procedures with those information requirements. The small start shifts provide a test of local recovery around the grid. Performance on different tracks, broader disturbances, and other RLPD configurations remains unmeasured.

## Follow-up at 65,536 online steps

All 24 PPTI jobs completed. The follow-up reused the dataset, eight seeds, initial BC weights, and eleven evaluation starts. DAgger kept eight rounds and 384 supervised updates per round, with each rollout extended from 2,048 to 8,192 steps. Each SAC variant kept one gradient update per four steps and ran 16,384 updates. The runs started from BC again. [Follow-up protocol](../outputs/stk-study-20261108-65k/protocol.json)

| Method | Mean lap time at 16,384 steps | Mean lap time at 65,536 steps | Completed at 65,536 steps |
| --- | ---: | ---: | ---: |
| Shared BC initialization | 38.31 s | 38.31 s | 88/88 |
| DAgger | 38.42 s | 38.28 s | 88/88 |
| SAC with symmetric sampling | 40.51 s | 38.69 s | 88/88 |
| Online-only SAC | 45.16 s | 40.89 s | 88/88 |

At the longer budget, symmetric SAC minus DAgger averaged **+0.41 s**. The approximate paired 95% bootstrap interval was **−0.43 to +1.73 s**, and the exact paired sign test gave **p = 1.00**. Each method led in four pairs. The median gap was +0.05 s. Seed 3 contributed a +4.72 s mean gap and was slower with symmetric SAC on all eleven starts. The comparison at this budget does not establish a consistent advantage for either method. Symmetric sampling still reduced SAC's mean lap time by 2.20 s relative to the online-only control. [Follow-up per-seed results](../outputs/stk-study-20261108-65k/summary.json) · [Paired plot](../outputs/stk-study-20261108-65k/seed-comparison.png)

## Recovery from shifted and misaligned starts

The saved 65,536-step policies were evaluated on 20 new Lighthouse starts, crossing lateral shifts of −2.5, −1.25, 0, +1.25 and +2.5 m with heading errors of −60°, −30°, +30° and +60°. The test uses race seeds 60000–60019. The pose changes are applied at GO and all starts passed the on-track check after the first step. BC was reconstructed from the fixed expert data and its actor hash was matched to each original training seed. All methods use deterministic actions and the same 150-second cap. [Recovery protocol](../outputs/stk-study-20261108-65k/recovery/protocol.json)

| Method | Finished | Mean lap time | Mean lap time at 60° error |
| --- | ---: | ---: | ---: |
| BC | 160/160 | 38.32 s | 38.34 s |
| DAgger | 160/160 | 38.36 s | 38.38 s |
| Symmetric SAC | 160/160 | 38.78 s | 38.83 s |

All three recovered and finished in every tested condition. The paired symmetric SAC minus DAgger gap averaged +0.42 s, with a 95% bootstrap interval of −0.44 to +1.74 s over the eight training seeds. Each method led in four seeds. The grid tests recovery from an initial pose error on one track. [Per-seed and per-heading results](../outputs/stk-study-20261108-65k/recovery/summary.json)

The nominal seed 3 replay used race seed 50005 with zero heading or lateral offset. DAgger finished in 38.20 s and symmetric SAC in 42.94 s. The gap was about 0.1 s at 300 m and grew by about 1.3 s from 400 to 500 m and another 1.4 s from 600 to 700 m. Symmetric SAC ran onto the grass repeatedly and logged 92 off-track steps, compared with four for DAgger. In the 600–700 m section it averaged 56 km/h versus 83 km/h. The video and headless rollouts produced identical per-step traces and finish times for these two policies. [Side-by-side video](../outputs/stk-study-20261108-65k/recovery/replays/seed-03-comparison.mp4) · [Speed and time gap plot](../outputs/stk-study-20261108-65k/recovery/replays/seed-03-telemetry.png) · [Telemetry summary](../outputs/stk-study-20261108-65k/recovery/replays/diagnostics.json)

The evaluator accepts the 65,536-step campaign directory, `--method` (`bc`, `dagger` or `symmetric`), `--seed` (0–7) and `--output`. Save the output for each pair to `recovery/laps/<method>-seed-<two-digit seed>.json` under the campaign, then run `python -m scripts.analyze_recovery <campaign>`. The replay adds `--replay` and uses `--render` under Xvfb to capture the single kart camera. [Recovery evaluator](../scripts/evaluate_recovery.py) · [Analysis source](../scripts/analyze_recovery.py)

## Reproduction

The experiment ran on PPTI hosts 09–12, with two seeds per host, all methods for a given seed on the same host, and one CPU thread per learner. The common dataset was generated on host 09 and verified by checksum before training on each host. Python 3.12.14, NumPy 2.2.6, PySuperTuxKart2 0.7.4, pystk2-gymnasium 0.10.0, PyTorch 2.12.0+cpu, and SB3 2.9.0 were recorded for every run. [Host and artifact provenance](../outputs/stk-study-20261108/execution.json)

Run the following from the project root in the project's Python environment. On PPTI the same commands can use `bash scripts/run_ppti.sh python` in place of `python`.

```bash
git clone https://github.com/osigaud/pystk_project.git .cache/time-trial/pystk_project
git -C .cache/time-trial/pystk_project checkout --detach ddf2cb6573850f890bba26d4e5f82888d9dac82a

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
study=outputs/stk-study-reproduction
mkdir -p "$study"
cp docs/STK_STUDY_PROTOCOL.json "$study/protocol.json"
python -m scripts.stk_study expert --output "$study/expert"

for seed in {0..7}; do
    for method in dagger symmetric sac; do
        python -m scripts.stk_study "$method" --seed "$seed" \
          --steps 16384 --bc-updates 2048 --dataset "$study/expert/expert.npz" \
          --output "$study/$method/$(printf 'seed-%02d' "$seed")"
    done
done

python3 -m scripts.analyze_stk_study "$study"
```

The analysis uses NumPy and Matplotlib. It checks dataset hashes, matched BC weights and evaluation starts, online budgets, and exact replay counts. The archived dataset SHA-256 is `68a61d2914c744f3dc1b6f15a2a59368ac66e9422f99f9c73a434f3a76f32bc1`. The completed campaign contains the dataset, 24 final policy ZIPs, training logs, and all 264 final lap records. [Training source](../scripts/stk_study.py), [environment source](../scripts/study_env.py), [analysis source](../scripts/analyze_stk_study.py), [campaign files](../outputs/stk-study-20261108/)
