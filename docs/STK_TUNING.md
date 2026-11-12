# Optuna and replay split follow-up

The follow-up asks whether the DAgger versus symmetric SAC result persists after both methods receive the same search budget and how the fraction of expert replay affects SAC. The original 21-track study supplies the fixed demonstrations, controller reference times, and original hyperparameters. [Original experiment](STK_ALL_TRACKS_STUDY.md)

## Search

Optuna 4.5.0 uses a random sampler with seed 112 for DAgger and 113 for SAC. Each method gets 12 trials, including its original settings. Each trial trains a separate policy for each of four development tracks (Lighthouse, Fort Magma, Sandtrack, and XR591), both experts, and training seeds 20 and 21. The tracks cover different driving conditions from the first study. Evaluation uses ten matched starts with race seeds 52000–52009 and the original offset grid. The score is mean capped lap time divided by Team L’Eclair’s native solo reference on each track, with equal weight across tracks, experts, and seeds. A timeout costs 200 s. The trial with the lowest mean score supplies the configuration for the held-out runs.

| Method | Search space |
| --- | --- |
| DAgger | Expert mixing decay sampled uniformly from 0.20 to 0.85. Supervised updates per round are 192, 384, or 768. |
| SAC at 50/50 | Learning rate sampled log-uniformly from 0.00001 to 0.0003. Entropy coefficient sampled log-uniformly from 0.001 to 0.05. Gradient steps per four simulator interactions are 1, 2, or 4. |

The development replay screen evaluates all five ratios at the original SAC settings. It shares the original 50/50 Optuna trial and adds 64 training runs. This provides a descriptive comparison before tuning. The Optuna searches use 384 training runs in total. Separate training and test seeds help limit selection bias from variation between seeds. [Eimer et al., 2023](https://proceedings.mlr.press/v202/eimer23a.html) · [Optuna sampling API](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.RandomSampler.html)

## Held-out comparison

The 17 tracks excluded from search each use both experts, training seeds 10–19, and evaluation race seeds 51000–51009. Each policy trains for 16,384 online simulator steps from the same BC initialization for its track, expert, and training seed. The feature and action spaces, reward, frozen demonstration dataset, and evaluation timeout match the original study. We compare the original and tuned DAgger settings, the original 50/50 SAC settings, and five SAC replay ratios at the settings selected by Optuna at 50/50. This gives 2,720 held-out training runs.

| Offline/online replay | Expert draws | Online draws |
| --- | ---: | ---: |
| 0/100 | 0 | 128 |
| 25/75 | 32 | 96 |
| 50/50 | 64 | 64 |
| 75/25 | 96 | 32 |
| 100/0 | 128 | 0 |

The gradient batch size is 128. The 100/0 condition still collects simulator steps to preserve the SB3 update cadence. Those transitions contribute zero replay samples. The 0/100 condition uses the expert demonstrations for the shared BC initialization. The ratio comparison measures the effect of replay composition at a common optimizer configuration selected for 50/50. It therefore answers how the ratios perform under those settings. The two original configurations run again on the same held-out seeds so their tuning effects can be paired.

We report capped time, completion, gradient updates, and elapsed training time per arm. All differences use paired training seed blocks averaged over tracks and experts. The primary tuned comparison is 50/50 SAC versus tuned DAgger. The four ratio comparisons against 50/50 are exploratory and use 95% percentile bootstrap intervals over the ten seed blocks. These intervals describe variation over training seeds for the fixed benchmark.

## Run and reproduce

Run from the project root with the same recorded simulator, SB3, and Python environment as the baseline campaign. Install Optuna 4.5.0 in that environment or make it available on `PYTHONPATH`.

```bash
python -m scripts.stk_tuning outputs/stk-tuning \
  --baseline outputs/stk-all-tracks-ppti-20261112 --workers 4
```

The workflow records the development and held-out job plans, Optuna storage, selected configurations, per-policy run files, and matched comparisons. It checks the dataset and initial actor hashes, evaluation starts, training settings, and realized replay counts. The PPTI campaign writes its progress and results under `outputs/stk-tuning-ppti-20261112`. [Runner](../scripts/stk_tuning.py) · [Training implementation](../scripts/stk_study.py)

The campaign completed all 448 development and 2,720 held-out runs. [Development screen](results/tuning/REPLAY_SCREEN.md) · [Optuna trials](results/tuning/optuna-trials.json) · [Held-out comparisons](results/tuning/RESULTS.md)
