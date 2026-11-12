# Experimental protocol for solo SuperTuxKart

This document records the settings used for the completed study and defines a separate, optional repeat on fresh training seeds. The run records and the completed manuscript provide the evidence for the report. [Study draft](STK_STUDY.md) · [Report writing plan](STK_REPORT_PLAN.md)

## Question and comparisons

At a fixed number of online simulator interactions, how does DAgger compare with SAC using 50/50 offline and online replay when a single kart tries to finish one Lighthouse lap as fast as possible?

| Condition | Starts from shared BC weights | Online expert labels | Online training |
| --- | --- | --- | --- |
| Behavior cloning (BC) | Yes | None | Evaluate the initialization before online training. |
| DAgger | Yes | One per visited state | Mix expert and learner actions, add expert labels for those states to the training set, and refit after each round. |
| Symmetric SAC | Yes | None | Train standard SB3 SAC with half of each batch from the frozen expert dataset and half from online replay. |
| Online-only SAC | Yes | None | Train the same SAC using only online replay for its updates. |

The DAgger versus symmetric SAC comparison holds online interactions fixed. DAgger has access to live expert queries, and SAC learns from the simulator reward. The symmetric versus online-only SAC comparison isolates replay sampling within the shared SAC implementation. BC measures the quality of the starting policy. The expert is a reference controller. [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html) · [RLPD paper](https://proceedings.mlr.press/v202/ball23a.html)

## Simulator and expert data

- Use PySuperTuxKart2 0.7.4, pystk2-gymnasium 0.10.0, SB3 2.9.0, PyTorch 2.12.0+cpu, NumPy 2.2.6, and Python 3.12.14 for the recorded configuration. Run one CPU thread per learner. [Environment code](../scripts/study_env.py)
- Use one Tux kart, one Lighthouse lap, `TIME_TRIAL` mode, difficulty 2, a 0.1 s simulation step, and a 1,500-control-step limit. Start control at GO. Apply the start offset there with zero speed. Keep brake, drift, nitro, manual rescue, and fire disabled.
- Use the same 32 features and continuous throttle and steering controls for all learners. The features describe the next five path endpoints, widths, validity masks, track center vector, kart-local velocity, and energy. Use two 64-unit ReLU layers and a tanh action output. [Feature and action code](../stk_actor/actors.py)
- Use Team 3’s `Agent3` at revision `ddf2cb6573850f890bba26d4e5f82888d9dac82a`. Clear its item observations and pass through its throttle and steering. Use that adapted expert for both demonstration collection and DAgger labels. [Pinned expert](https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team3)
- Freeze the 32-lap dataset generated with race seeds 12300–12331. A NumPy RNG seeded with 12300 draws lateral offsets uniformly from −1.5 to +1.5 m. It contains 12,224 transitions and all demonstrations finished. Verify the SHA-256 `68a61d2914c744f3dc1b6f15a2a59368ac66e9422f99f9c73a434f3a76f32bc1` before using the archived dataset. [Dataset manifest](../outputs/stk-study-20261108/expert/expert.json)

## Training settings

Use eight training seeds, 0–7, at each of the two completed budgets. Initialize all methods within a seed from identical BC actor weights and verify the actor hash. BC uses 2,048 Adam updates at learning rate 0.001 and batch size 128. Draw examples uniformly with replacement from the expert data and fit mean squared error on the deterministic action. The initial SAC log standard deviation has zero weights and bias −2. Training resets sample offsets uniformly from −1.5 to +1.5 m using a generator seeded by the learner seed. DAgger and SAC share the reset distribution, with different first-reset seed handling. Use identical race seeds and poses across methods at evaluation. [Trainer](../scripts/stk_study.py)

| Setting | 16,384 online steps | 65,536 online steps |
| --- | ---: | ---: |
| DAgger rounds | 8 | 8 |
| DAgger online steps per round | 2,048 | 8,192 |
| DAgger supervised updates per round | 384 | 384 |
| DAgger live expert queries | 16,384 | 65,536 |
| SAC gradient updates | 4,096 | 16,384 |
| SAC online replay capacity | 16,384 | 65,536 |
| Symmetric SAC draws from each replay source | 262,144 | 1,048,576 |

In DAgger round `r` (1–8), execute the expert’s action with probability `0.5**r` per step and the current deterministic actor otherwise. Query the expert at every step. Retain the initial demonstrations and all collected labels. Use the BC optimizer, loss, and batch size for supervised updates.

For both SAC variants, use twin 64-by-64 ReLU critics, Adam at learning rate 0.0001, discount 0.995, target coefficient 0.005, fixed entropy coefficient 0.02, one-step targets, and a target update after each gradient update. Sample the stochastic actor from the first step and take one gradient update per four environment steps. The batch size is 128. Symmetric SAC draws 64 transitions independently with replacement from each replay source. Online-only SAC draws all 128 from its online buffer.

SAC uses `progress-v1`. It rewards 0.1 per metre of verified new on-track progress, charges 0.02 per racing step and another 0.08 for off-track steps, and gives 10 at the finish in the solo setup. The timeout is a truncation for SAC bootstrapping. Restart both budget conditions from the common BC initialization. The larger budget changes rollout length for DAgger and the number of updates for SAC. [Reward code](../stk_actor/learn.py)

## Evaluation and analysis

1. Evaluate the final checkpoint at each budget using deterministic actions. The 11 primary starts pair race seeds 50000–50010 with lateral offsets −2, −1.75, −1.25, −0.75, −0.25, 0, +0.25, +0.75, +1.25, +1.75 and +2 m, in that order. Use the normal heading. Score a finish by the simulator’s finish time and assign 150 s to each timeout. Keep completion as a separate outcome. [Original run configuration](STK_STUDY_PROTOCOL.json)
2. For each seed and method, average capped time over the same 11 starts. Subtract DAgger’s mean from symmetric SAC’s mean within the seed. Positive values favor DAgger. Report the individual seed differences, their mean, and an approximate 95% percentile interval from 20,000 paired bootstrap resamples of training seeds, with RNG seed 17. Report the two-sided exact sign test separately as evidence about the direction of the differences. The independent unit is the training seed. [Analysis code](../scripts/analyze_stk_study.py)
3. Report completion, mean time among finishers, live expert queries, and each SAC replay count. Present comparisons with BC and the online-only SAC control as secondary. Treat changes across budgets descriptively and show both intervals. A wide interval around zero at the larger budget leaves the size and direction of the difference uncertain.
4. For the recovery follow-up, use the saved 65,536-step BC, DAgger and symmetric SAC policies. Cross offsets `[-2.5, -1.25, 0, 1.25, 2.5]` m with headings `[-60, -30, 30, 60]` degrees in offset-major order and assign seeds 60000–60019. Verify the requested poses and that each start is on track after the first step. Use the same capped metric and paired analysis. Treat the nominal seed 3 replay as a case study. [Recovery protocol](../outputs/stk-study-20261108-65k/recovery/protocol.json) · [Recovery evaluator](../scripts/evaluate_recovery.py)

Keep all valid trials, including timeouts. Rerun an interrupted or corrupted job with the same seed and configuration and record the reason. Report missing pairs explicitly. Evaluation results are never used to choose checkpoints for the comparison.

## Optional fresh-seed repeat

If the team wants stronger evidence at 65,536 steps, repeat that budget with **16 new training seeds, 100–115**, and keep the dataset, hyperparameters, actor initialization procedure, three online conditions, and 11-start evaluation suite fixed. This is 48 training jobs. The new seeds test repeatability on the fixed suite. The suite has already been used in this study.

Before starting, agree to include all planned pairs and report the mean paired gap and 95% interval as the primary result. Keep the original eight seeds in a separate table. Compare each method with BC and online-only SAC as secondary results. The number of seeds is a compute budget choice. Report the interval even if it spans zero.

## Reproduce and audit

Use the project's locked environment, or the configured PPTI Python runner if it matches the recorded versions. Clone and check out the expert revision as shown in the [study draft](STK_STUDY.md#reproduction). The commands below reuse the archived expert data and run the optional repeat. The same commands with seeds 0–7 and the corresponding step budget reproduce a completed condition.

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
campaign=outputs/stk-repeat-65536
mkdir "$campaign"
cp -a outputs/stk-study-20261108/expert "$campaign/expert"
python - "$campaign" <<'PY'
import json
from pathlib import Path
import sys

protocol = json.loads(Path("docs/STK_STUDY_PROTOCOL.json").read_text())
protocol["training_seeds"] = list(range(100, 116))
protocol["online_steps"] = 65536
(Path(sys.argv[1]) / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
PY

for seed in {100..115}; do
    for method in dagger symmetric sac; do
        python -m scripts.stk_study "$method" --seed "$seed" \
          --steps 65536 --bc-updates 2048 --dataset "$campaign/expert/expert.npz" \
          --output "$campaign/$method/$(printf 'seed-%02d' "$seed")"
    done
done

python -m scripts.analyze_stk_study "$campaign"
```

Record the source revision, runtime versions, dataset and initial actor hashes, planned seeds, and output paths before handing off the results. Retain each `run.json` and `policy.zip`, all lap rows, the aggregate `summary.json`, and the paired plot. The analyzer checks that the dataset, BC actor hashes, evaluation starts, and replay counts match across the comparison.
