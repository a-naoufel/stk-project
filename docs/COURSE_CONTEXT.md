# Course context checked on 23 October 2026

- The instructor's dedicated template is
  https://git.isir.upmc.fr/bpiwowar/pystk2-project-template
  at commit `ad63660df0384a47e843dfd05f6c3c87b060f996`.
- The pinned Mattermost repository `mlia/master-mind-rl` contains the shared
  practical environment, not this project template.
- Benjamin Piwowarski's 20 October announcement switches STK to ML-Arena:
  https://channel.isir.upmc.fr/m1-mind-25-27/pl/43orzeyh9igqzkefuyb6mqxpso
- Course enrollment: https://ml-arena.com/enroll/78XQ77QX
- Challenge: https://ml-arena.com/viewchallenge/169
- The enrollment page identifies “Reinforcement Learning (Master MIND)” and
  “SuperTuxKart Grand Prix”; its private details require a sign-in.
- Zimbra searches for STK, SuperTuxKart, ML Arena and RL did not locate
  additional STK-specific requirements. No deadline is assumed here.

According to the template, platform races use four karts on a random track,
with a CPU runtime (3 CPUs, 3 GB RAM, 1 second per action). The upload is flat,
at most 10 files and 100 MB. Runtime versions are pinned in `pyproject.toml`.
The original instructor README is preserved in `UPSTREAM_README.md`.

Confirm the current deadline, report requirements, and teammate arrangements
on Moodle/in class. An RL channel announcement on 13 October explicitly
requires pairs for the second mini-project and final project, except the
single student permitted when the class size is odd.

Next experiments:

1. Measure the heuristic across tracks and seeds; retain a fixed evaluation set.
2. Run PPO for a useful budget and compare completion rate, distance, position,
   and inference time. A high return alone does not prove a race was finished.
3. Consider behavior cloning from the built-in AI before PPO if exploration
   stalls. The simulator can expose the AI action when `AgentSpec(use_ai=True)`.
4. Add controls such as brake/rescue/nitro only with matching training and
   inference wrappers, a new feature version where appropriate, and new checks.
5. Check the flat package using the instructor's script before uploading it.
