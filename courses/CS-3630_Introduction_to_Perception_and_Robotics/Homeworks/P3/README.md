# CS 3630 Project 3 (Fall 2026): Vacuum Cleaning Robot

Dynamic Bayes nets, hidden Markov models, factor-graph inference and Markov decision processes,
for a vacuum robot in the nine rooms of the COLD-Freiburg lab. This is Chapter 3 of
[roboticsbook.org](https://www.roboticsbook.org). New this year: after you finish the code, the
same robot drives the lab in Webots, using your `main.py`.

## 1. Set up (once)

You need the conda environment from Project 1B/2 style setup, plus Graphviz for the optional
diagrams. Windows users: `gtsam` has no Windows wheels, use WSL as in Project 2.

```bash
conda create -n cs3630_p3 python=3.12 graphviz -c conda-forge
conda activate cs3630_p3
python -m pip install -r requirements.txt
python -c "import numpy, gtsam, gtbook, plotly, pandas; print('Project 3 imports OK')"
```

Then open `CS3630Proj3.ipynb` with the `cs3630_p3` kernel (VS Code or JupyterLab). We do not
support Colab for this assignment.

## 2. Work on the project

- **All the code you write goes in `main.py`**, between the `START OF YOUR CODE` /
  `END OF YOUR CODE` markers of each TODO (and the two blanks in the `prob_spec` table). Do not
  change function signatures and do not add helper functions outside the markers: the autograder
  keeps only what is between the markers.
- **The notebook explains each step and checks your work.** It imports `main.py` with
  `%autoreload 2`, so save `main.py` and re-run a check cell; no kernel restart needed.
- **Run all checks at once** from a terminal: `python -m pytest project3_test.py -q`.
- **Answer the `REPORT QUESTIONS`** (they are in the notebook and in `project3_report.pptx`).

## 3. Webots (Section 3.6)

After Section 3.5, follow [`webots/STUDENT_SETUP_AND_NAVIGATION.md`](webots/STUDENT_SETUP_AND_NAVIGATION.md).
In short: point Webots' *Python command* preference at your `cs3630_p3` interpreter, open
`webots/worlds/project3_vacuum.wbt`, press Run, click the 3D view, and use the keys:

| Key | Mode | What it shows |
|---|---|---|
| `L` | LIGHT | the robot's light sensor in the room you pick (`1`-`9`) |
| `T` | TAPE | the 6-step action cycle with outcomes sampled from your `prob_spec` |
| `M` | MPE | a hidden trajectory, real readings, then your `GTSAM_MPE` estimate |
| `P` | POLICY | the robot following your `policy_iteration` policy from the room you pick |
| `C` | CUSTOM | the robot following your `get_custom_policy` |
| `R` | | replay the last run with the same seed (clears the cleaned trail); pressing a mode key again starts a new run with the next seed |

Keep the `webots/` folder next to `main.py`: the controller finds your code by that relative path.
Report questions 3.1b, 3.2b, 4.4 and 5.3b ask for screenshots and console lines from these modes.

## 4. Submit

| Assignment | File | Points |
|---|---|---|
| Project 3 code | `main.py` (exactly that name) | 60, autograded |
| Project 3 report | `project3_report.pptx` exported to PDF, every question assigned to its page | 60 + 2 EC |

## 5. Files

| File | Purpose |
|---|---|
| `main.py` | your code (the only file you edit and submit) |
| `CS3630Proj3.ipynb` | the guided notebook with explanations, demos and checks |
| `project3_test.py` | the sanity checks the notebook runs (`pytest`-able) |
| `project3_report.pptx` | the report template |
| `requirements.txt` | Python dependencies |
| `webots/` | the Webots world, the controller that imports your `main.py`, and the guides |
