# Project 3 in Webots: setup and navigation

This folder makes the vacuum robot from `main.py` drive the nine rooms of the Freiburg lab in
Webots. The controller **imports your `main.py` and calls your functions live**; nothing here is
autograded. Report questions 3.1b, 3.2b, 4.4 and 5.3b ask for screenshots and console lines from it.

![POLICY mode after 8 steps](assets/policy_mode_example.jpg)

## 1. Before you start

- Finish Sections 3.1 to 3.5 of the notebook first. Each Webots mode calls a few of your `main.py`
  functions (table in section 4 below); a mode whose functions are missing shows
  `STUDENT CODE ERROR` in the console and on the HUD, and the other modes still work.
- You installed **Webots R2025a** in Project 1B. Keep that version: the world and every model in
  it are pinned to R2025a.
- Keep this `webots/` folder **next to `main.py`**, exactly as shipped. The controller finds your
  code by walking up from `webots/controllers/project3_vacuum/` to the folder that holds `main.py`.
  Do not move, rename or copy the world elsewhere.

## 2. Point Webots at your Python

`main.py` imports NumPy, GTSAM, gtbook, pandas and plotly at module scope, so Webots must run the
same interpreter you use for the notebook (the `cs3630_p3` environment). Print its path:

```bash
conda activate cs3630_p3
python -c "import numpy, gtsam, gtbook, plotly, pandas; print('Project 3 imports OK')"
python -c "import sys; print(sys.executable)"          # macOS / Linux
python -c "import sys; print(sys.executable)"          # Windows (WSL: same command inside WSL)
```

Then in Webots: **Tools > Preferences** (macOS: **Webots > Preferences**) > **General** > **Python
command** > paste the full path printed above > OK. Restart Webots. This is the same step as
Projects 1B and 2, just with the `cs3630_p3` path.

## 3. Open the world and run

1. **File > Open World...** > `webots/worlds/project3_vacuum.wbt`. The first open downloads a few
   official Webots models (walls, lights, furniture) from GitHub and caches them; later opens are
   instant.
2. Press the **Run** (play) button. The robot appears in the bathroom (it is parked out of sight
   until the controller starts) and the console should show
   `P3 WEBOTS READY | student code: /.../main.py | mode=POLICY seed=3630 start=8 steps=8`.
   If it says `STUDENT CODE ERROR: ...` instead, read section 6.
3. **Click once inside the 3D view** so it has keyboard focus, then press a mode key (section 4).
   Nothing runs until you do; the HUD status line says `Ready`.
4. The default view is a top-down look at the whole lab. Camera: left-drag orbits, right-drag pans,
   scroll zooms; **View > Restore Viewpoint** (or the toolbar button) brings it back. Do not drag
   the robot or the furniture; the room and door coordinates are fixed in the controller.

## 4. Modes and keys

| Key | Mode | What happens | Your `main.py` functions it calls |
|---|---|---|---|
| `L` | **LIGHT** | The robot parks and prints its light sensor once per second as `[LIGHT] room 9 (stairs area) \| raw=0.23 \| level=dark`. Press `1`-`9` to drive to another room. | none |
| `T` | **TAPE** | Starts in the kitchen and replays the 6-step cycle from `create_custom_action_sequence` for 7 steps. Each transition is *sampled* from your `prob_spec`, so the robot can slip. | `create_action_series`, `create_custom_action_sequence`, `get_transition_prob` |
| `M` | **MPE** | Drives the same tape with the true rooms hidden, records the light readings, then calls `GTSAM_MPE` on the factor graph from `create_factor_graph(..., measurements=..., actions=..., prior=...)` and shows inferred versus true rooms. | `create_factor_graph`, `GTSAM_MPE`, `get_kitchen_start_prior`, plus the TAPE functions |
| `P` | **POLICY** | From the start room, follows the policy returned by `policy_iteration(R, T, get_test_policy())` for 8 steps and accumulates the kitchen reward. | `generate_reward_table`, `get_transition_prob`, `get_test_policy`, `policy_iteration` |
| `C` | **CUSTOM** | Same as POLICY but follows `get_custom_policy()`. | `generate_reward_table`, `get_transition_prob`, `get_custom_policy` |
| `1`-`9` | start room | In LIGHT mode: drive there now. Otherwise: park the robot there and use it as the start room of the next POLICY or CUSTOM run. If a run is in progress it is stopped first. TAPE and MPE always start in the kitchen, like the notebook. | |
| `R` | replay | Stop the run, wipe the cleaned trail from the floor, and run the last mode again with the same seed and start room. | |

Keys work at any time: pressing a mode key during a run stops it, clears the floor and starts the
new mode. **Seeds.** The first run of each mode uses seed 3630; pressing the same mode key again
starts a new run with the next seed (3631, then 3632, ...), and `R` replays the last run with the
same seed. The seed drives every sampled transition through NumPy, so the same seed and the same
`prob_spec` give the same route every time; it is shown on the HUD and in every `... DONE` line.
The light readings come from a real Webots light sensor and always vary a little.

The HUD (top left) shows: mode, seed, start room, step; current room, action, intended and sampled
outcome; the live light reading and its level; the cumulative reward and `V(start)` (POLICY /
CUSTOM) or the true-versus-inferred rooms (MPE); the key help; and a status line.

## 5. What to put in the report

Copy console lines from the Webots console (select, Ctrl/Cmd-C). The lines the questions ask for:

```
POLICY from policy_iteration(): 1:1->2 2:2->3 3:2->1 4:4->2 5:5->2 6:6->2 7:7->6 8:8->9 9:9->2   V(8)=77.63
[POLICY] step 3/8 | room 2 (corridor) | action 2->3 | intended 3 | sampled 3 (p=0.88) | light raw=0.73 level=light | reward +10 | total 10
POLICY DONE seed=3630 start=8 steps=8 total_reward=60 rooms=8,9,2,3,3,3,3,3,3
[TAPE] step 5/7 | room 6 (one-person office) | action 6->2 | intended 2 | sampled 7 (p=0.02) | light raw=0.50 level=medium
TAPE DONE seed=3630 final_room=7 rooms=3,2,6,7,6,7,7,7
[LIGHT] room 9 (stairs area) | raw=0.23 | level=dark
MPE RESULT seed=3630 | correct 5/8 | true=3,2,6,7,6,7,7,7 inferred=3,2,6,7,6,2,3,2 | P(inferred)=3.92e-03 P(true)=1.08e-04 (36x)
[MPE]  k  action  z        true  inferred
[MPE]  1  -       medium   3     3
```

Three lines explain what happened when the robot did not go where the action pointed:
`attempted 2->1 but stayed in 2 (p=0.04)` (the sampled outcome was the current room);
`tape action 2->3 is not available from room 7: robot stays (p=1.00)` (the tape's action is not a
doorway from where the robot slipped to, so under the notebook's model it stays put); and
`policy chose 2->1 in room 3: not a doorway from room 3, so the robot stays and collects the kitchen
reward again (p=1.00)` (the policy is staying on purpose). The `MPE RESULT` line also prints the
model probability of the inferred and the true trajectory under your own factor graph, so you can
see why MPE preferred one over the other.

Screenshots must show the HUD (mode, seed, step) and the robot. Any mode run is valid evidence;
the only invalid runs are ones whose HUD shows `SELF-TEST MODE` or `STUDENT CODE ERROR`.

## 6. Troubleshooting

- **`STUDENT CODE ERROR: No module named 'gtbook'` (or `gtsam`, `plotly`, `pandas`).** Webots is
  running a different Python than your notebook. Redo section 2 with the exact path from
  `sys.executable`, restart Webots.
- **`STUDENT CODE ERROR: Project 3 main.py was not found ...`.** The `webots/` folder is not next
  to `main.py`. Restore the shipped layout.
- **`STUDENT CODE ERROR: main.py is missing ... for mode POLICY`.** That mode needs functions you
  have not implemented yet; the message lists them. Other modes still work. Fix `main.py`, press `R`.
- **A mode raises inside your code** (traceback in the console, HUD shows `ERROR`). Fix `main.py`,
  press `R`; the controller re-imports your file on every restart.
- **`ModuleNotFoundError: No module named 'controller'`.** You ran `project3_vacuum.py` from a
  terminal. It only runs inside Webots.
- **The view is a strange close-up or sideways.** Webots saved a per-user view file
  (`.project3_vacuum.wbproj`) from an earlier session. **View > Projection > Perspective**, then
  **View > Restore Viewpoint**; or delete that hidden file next to the world.
- **First load is slow / grey placeholders.** The models are downloading; wait for the console to
  settle. If a download fails, check your network and reload the world.
- **Keys do nothing.** Click inside the 3D view first so it has keyboard focus. Use the number row,
  not the numeric keypad.
- **I pressed a room number but TAPE/MPE started in the kitchen.** By design: the tape's first
  action is `3->2` and the MPE prior is the kitchen prior. Room numbers apply to POLICY, CUSTOM
  and LIGHT.
- **MPE recovers few rooms with seed 3630.** That is a real result (the true trajectory slips off
  the action sequence and gets stuck); it is a good example for question 4.4. The third MPE run
  (seed 3632) recovers all eight rooms.
- **A key seems to fire several times.** It should not: one press is one event. If it does, you
  are probably holding the key while the view loses focus; click the 3D view and press once.
