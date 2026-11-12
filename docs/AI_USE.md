# AI assistance log — draft for your course declaration

On 23 October 2026, OpenAI Codex helped locate the official STK template,
install the pinned environment, and implement the initial feature/action
wrappers, heuristic controller, PPO training script, evaluation script,
packaging, tests, and setup documentation. The exact model identifier was
not exposed in this session. The instructor's `agent.py` shim was preserved.

This is a bootstrap, not a completed experimental report. A short PPO run
only checks the training pipeline; it does not establish racing performance.
Record further AI use and your own modifications here. Review and adapt this
log to the instructor's actual AI use statement before handing in your work.

On 6 November 2026, OpenAI Codex added the PPTI deployment script, the runner
for an existing course environment, and deployment documentation. It checked
the source archive and SSH command construction, then exercised deployment
in a local temporary directory with the four actor tests. PPTI connections
initially timed out before authentication. Existing dependency edits were
preserved.

After SSH access was restored, Codex installed the locked runtime and game
assets under `/temporary/21501292/stk-project` on PPTI hosts 01 and 04–16.
All 14 hosts passed the four actor tests and a 100-step headless race check.
Hosts 02 and 03 remained unreachable. The teacher-provided activation path
was checked but invoked Python 3.13 against Python 3.11 packages, so isolated
Python 3.12 environments were used. Codex also updated the Makefile to select
the PPTI runtime automatically and ran an isolated 256-step PPO smoke test
and a full heuristic race on host 01. The smoke checkpoint was not installed
as the default policy. No ML-Arena upload or deployment was performed.

Later on 6 November, at the user's request, Codex launched 14 independent
PPO training runs on hosts 01 and 04–16, requesting one million steps per
host, different seeds, random tracks and three CPU threads per process.
It added an explicit training run directory, a detached job supervisor,
progress inspection and result collection. Lifecycle checks covered jobs
surviving launch disconnection, duplicate launch protection, successful
completion and failure reporting. The actor tests passed. All 14 real jobs
were observed running and advancing after SSH disconnection; this records
training startup, not successful completion or competitive performance.

On 7 November, Codex verified that all 14 jobs had completed successfully,
then evaluated all final checkpoints and the heuristic/built-in AI on 21
tracks and seeds 100–104. It fixed the three leading PPO models before
validating them and both baselines on fresh seeds 200–204: 2,205 races total.
Codex added evaluation provenance and incremental result saving, checked
seed repeatability and checkpoint hashes, backed up all final models,
computed paired track-bootstrap summaries, and generated reports and plots.
It also checked the leading seed-8 checkpoint through the instructor's flat
loader. Analysis checks covered failure penalties, completion-based ranking,
duplicate-race rejection and AI latency handling. The four actor tests
passed. The heuristic completed more races than the best PPO checkpoint on
both suites; no model was uploaded, deployed or substituted automatically.

On 12 November 2026, Codex helped design and run the 21-track solo study
comparing DAgger and symmetric replay in SB3 SAC. It implemented the campaign
and analysis scripts, ran the controller benchmarks and 1,260 training jobs
on PPTI, and checked the saved results against the experimental protocol.
Codex prepared the first Markdown report draft from those results and verified
the summary statistics against the per-lap records. The report is a draft for
the project authors to review.
