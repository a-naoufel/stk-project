# DAgger and symmetric replay across all 21 SuperTuxKart tracks

[Study results and figures](results/all-tracks/RESULTS.md) · [Controller baselines](results/all-tracks/CONTROLLERS.md)

## Research questions

1. At the same online interaction budget, which method gives lower capped solo lap times on the official course tracks?
2. How much do the results change when the demonstration controller changes? Does the choice affect DAgger and symmetric SAC differently?
3. Do the learned policies beat the original controllers under matched solo conditions? How stable is each method across independent training seeds on a track, and how much does the mean performance change across tracks?

Train a separate policy for each track. The study measures performance across 21 training tasks. Evaluating generalization to unseen tracks would require a separate training design. Use the earlier Lighthouse experiment as a pilot. [Lighthouse study](STK_STUDY.md)

## Design

| Factor | Levels |
| --- | --- |
| Track | All 21 official race tracks in the controller profile. See the machine-readable protocol for IDs. |
| Demonstration source | Team L’Eclair (team 3) and The Winners (team 4), crossed with every track and method. |
| Training seed | 0–9 in every track, source, and method cell. |
| Training condition | DAgger, SB3 SAC with symmetric sampling, and SB3 SAC with online-only replay. |
| Baselines | Shared behavior cloning (BC) at each track, source, and seed. The five source controllers also run solo with both action configurations below. |

This gives **1,260 training jobs**, including the online-only SAC ablation. The two main methods account for 840 jobs. Keep the same dataset and starting actor within a track, expert source, and seed. Each track and expert source gets the same weight in the primary contrast. Compare each learned method with its own BC initialization and both SAC variants with each other.

The supplied controller profile ran the five teams together with shuffled grids over seeds 0–9 and a 200 s cap. The Winners had the best overall capped mean (58.36 s, 210/210 finishes). Team L’Eclair won eight tracks by capped mean and finished 209/210, making it a second strong expert with a different track profile. Fort Magma illustrates why finish rates matter. Donkey Bombs had the fastest mean among finishers (56.91 s, 8/10). The Winners finished 10/10 with a 59.75 s mean and won by capped time. Use these multiplayer results to motivate the design and describe historical rank and variability. The attached summary has aggregate statistics, so paired learner comparisons use the new solo lap records. [Machine-readable protocol](STK_ALL_TRACKS_PROTOCOL.json)

## Simulator and controller baselines

- Run one Tux kart in headless time trial, difficulty 2, for one lap. Control starts at GO. Each simulator control step advances 0.1 s. The cap is 2,000 steps (200 s). Training starts sample lateral offsets uniformly from −1.5 to +1.5 m with the nominal heading and zero starting speed. Training methods share the reset distribution. Record runtime versions with each run.
- Evaluate each final policy on the same 10 starts per track using deterministic actions. Pair race seeds 50000–50009 with offsets −1.2, −0.9, −0.6, −0.3, 0, 0, +0.3, +0.6, +0.9, and +1.2 m. Each race yields simulator finish time if completed and a capped score of 200 s on timeout. Report completion alongside mean capped time and mean time among finishers. Retain timeouts and record any missing runs or retried jobs.
- Evaluate all five upstream controllers under these starts. For the restricted configuration, execute only throttle and steering, as in learner training. For the native driving configuration, also execute brake, drift, nitro, and manual rescue when requested. Both configurations retain track-object observations for the controllers and mask held power-ups. Firing is disabled. All karts use the same physics setup. A learner can beat a controller under one action configuration and trail it under the other.
- Use upstream controller revision `ddf2cb6573850f890bba26d4e5f82888d9dac82a`. Collect 8,192 transitions per track and expert source with throttle and steering. Run episodes with race seeds starting at 12300 and offsets drawn from the same distribution as training. Keep the first 8,192 transitions in collection order, including transitions from unfinished laps. Keep the full episode outcomes in the dataset manifest and report their completion alongside the solo expert results. Freeze each dataset and record its checksum. [Upstream controller code](https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents)

## Training

- All learners receive the same 32 track and kart features and output continuous throttle and steering. Use two hidden layers of 64 ReLU units. Initialize all online conditions from 2,048 BC Adam updates at learning rate 0.001, batch size 128, and squared action error. The stochastic actor's initial log standard deviation has zero weights and bias −2.
- DAgger runs eight rounds of 2,048 simulator steps. Query the selected expert for every visited state, execute its action with probability `0.5**round`, and aggregate its labels with the fixed demonstrations. After each round, make 384 supervised updates using the BC optimizer and loss. Record both expert queries and interventions.
- Both SAC conditions use SB3 with standard twin critics, 128 transitions per batch, learning rate 0.0001, discount 0.995, target smoothing coefficient 0.005, fixed entropy coefficient 0.02, online replay capacity 16,384, and one gradient update per four online steps from the first rollout. Symmetric sampling follows RLPD’s per-batch rule and draws 64 transitions from the fixed expert data and 64 from online replay. The online-only ablation uses all 128 from online replay. Use the same critic architecture and update schedule in both. [Ball et al., 2023](https://proceedings.mlr.press/v202/ball23a.html) · [Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)
- Both SAC conditions use the same `progress-v1` reward. It awards 0.1 per metre of verified new on-track progress, subtracts 0.02 per racing step and 0.08 per off-track step, and awards 10 at the solo finish. Treat timeouts as truncations for bootstrap targets. The 16,384-step cap matches online interaction counts. DAgger receives 16,384 live expert labels, and SAC receives the simulator reward. [Trainer](../scripts/stk_study.py) · [Reward](../stk_actor/learn.py)

## Analysis and report figures

Define a policy’s score as its mean capped time over the 10 fixed starts. The paired unit is the training seed. Average across starts first, then subtract scores within each track, source, and seed. A positive symmetric SAC minus DAgger gap favors DAgger.

| Question | Analysis |
| --- | --- |
| Overall method effect | Divide each paired difference by that track’s Team L’Eclair native solo capped mean, then multiply by 100. Average equally over the 21 tracks and both sources for each seed. Report the mean across the 10 seed blocks and a 95% paired bootstrap interval from 20,000 resamples. This estimate is conditional on the fixed course suite and start grid. |
| Expert source effect | Within each method, compute Team 4 minus Team 3 under the same track and seed. Report the normalized paired effect and the difference between the two methods’ source effects. The comparison includes differences in demonstration quality and coverage, and live labels for DAgger. |
| Learner versus controllers | Report each learner’s capped-time gap and completion against each of the five solo baselines in both action configurations. Intervals describe variation across learner training seeds on the fixed starts. Include BC and online-only SAC to show how much training and replay contributed. |
| Variation within a track | Report the sample SD of the 10 policy scores for each track, source, and method. Also report mean within-policy SD across starts as a separate measure of sensitivity to initial conditions. |
| Variation between tracks | Report the sample SD of the 21 mean normalized policy scores and of the 21 mean normalized method gaps. Show the per-track paired gaps and intervals. Treat track-specific intervals and multiple controller comparisons as exploratory. |

Lead the report with the paired effect plot and a table of completion and capped time. Follow with the expert-source comparison and the training-seed variance heatmap. Read the variance together with completion. A cell with ten identically capped policies has zero variance and poor performance. Discuss Fort Magma as a case study of how the action interface affects the expert demonstrations. Use normalized effects to compare tracks with different lap lengths.

## Reproduce and audit

Install the locked project environment, clone the pinned upstream checkout, and make the controller dependencies available to the project Python. The baseline evaluator needs OmegaConf 2.3.0, Pandas 2.2.3, and Plotly 6.3.1 because the upstream controllers import those packages. The analysis figures use Matplotlib 3.10.8. Set `PYTHONPATH` to the dependency directory if installed separately. Run these commands from the project root.

```bash
python -m scripts.stk_all_tracks prepare outputs/stk-all-tracks-20261112 --workers 2
python -m scripts.stk_all_tracks train outputs/stk-all-tracks-20261112 --workers 2
python -m scripts.analyze_stk_all_tracks outputs/stk-all-tracks-20261112
```

Pass `--expert-repo` if the checkout differs from `.cache/time-trial/pystk_project`. Both runner phases accept `--tracks` to assign tracks to machines. Use `--seeds` to distribute training for a single track across machines with the same reference files. Completed jobs are published with their result metadata after evaluation and skipped on restart. Preserve the individual results, checkpoints, demo arrays, per-lap records, and analysis outputs. Use `--baselines-only` for the solo controller table after all reference runs finish. Pass `--controller-profile /path/to/RESULTS.md` to join the historical five-kart aggregates to the new solo references. The full analyzer validates exact evaluation starts, shared BC weights, dataset hashes, and replay draw counts before writing the method comparison. [Campaign runner](../scripts/stk_all_tracks.py) · [Analysis code](../scripts/analyze_stk_all_tracks.py)
