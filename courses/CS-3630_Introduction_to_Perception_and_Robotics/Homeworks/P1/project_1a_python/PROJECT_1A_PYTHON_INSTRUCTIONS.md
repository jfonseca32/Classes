# Project 1A — Python Fundamentals

**Points:** 50
**Gradescope submission:** `project1.py` only
**Estimated time:** 1–2 hours, plus one-time environment setup

## What to use

- `project1.ipynb` — primary tutorial, examples, and practice sequence
- `project1.py` — starter code and the only submitted file
- `PROJECT_1A_PYTHON_REFERENCE.md` — optional Python reference
- `test_project1.py` — public tests

Work through the notebook from top to bottom and implement each matching TODO in
`project1.py`. The notebook is instructional and is not submitted.

## Environment setup

Projects 1A and 1B share one Conda environment named `cs3630_p1`. Create it once
here; Project 1B later points Webots at this same environment. The environment
is named for Project 1 because later projects may define their own
environments with different dependencies.

### 1. Install Miniforge (once)

Miniforge supplies the `conda` environment manager used in this course.
Download the installer for your operating system and processor from the
[official Miniforge repository](https://github.com/conda-forge/miniforge) and
follow its installation instructions. Then close and reopen your terminal and
verify:

```bash
conda --version
```

If you already use another Conda distribution, you may keep it as long as you
can reproduce a Python 3.12 environment. On Windows, run all commands below in
**Miniforge Prompt** or another shell where `conda` works.

### 2. Create and activate the `cs3630_p1` environment (once)

```bash
conda create -n cs3630_p1 python=3.12
conda activate cs3630_p1
python --version
```

The last command should print a version beginning with `Python 3.12`. A new
terminal does not activate the environment automatically, so run
`conda activate cs3630_p1` whenever you work on Project 1.

### 3. Install the Project 1A dependencies

With `cs3630_p1` active, change into the `project_1a_python` directory (the
one containing `requirements.txt`) and install:

```bash
cd path/to/project_1a_python
python -m pip install -r requirements.txt
```

Replace `path/to` with wherever you saved the project, for example
`cd ~/cs3630/project_1a_python`. In most terminals you can type `cd ` and
then drag the `project_1a_python` folder onto the window to paste its full
path. Using `python -m pip` ensures the packages install into the Python
interpreter of the active `cs3630_p1` environment.

### 4. Open the notebook

`requirements.txt` already includes `ipykernel`, the package notebook editors
need in order to run cells with the `cs3630_p1` environment. Pick either
Jupyter front end:

- **VS Code:** install the Python and Jupyter extensions, open
  `project1.ipynb`, and select the `cs3630_p1` interpreter as the notebook
  kernel; or
- **JupyterLab:** run `python -m pip install jupyterlab` once, then launch it
  from this directory with `jupyter lab`.

If VS Code reports that running cells "requires the ipykernel package", or the
import cell cannot find NumPy, then either the selected kernel is not the
`cs3630_p1` environment or step 3 was skipped. Reselect the kernel, and rerun
`python -m pip install -r requirements.txt` with `cs3630_p1` active.

### 5. Verify the setup

```bash
pytest -q test_project1.py
```

A correctly set up environment collects every test and reports failures whose
messages contain `NotImplementedError: TODO ...`. The starter code is expected
to fail tests until its TODOs are implemented. A `ModuleNotFoundError` or
`ImportError` instead means the environment or dependencies are not set up
yet.

## Grading

| Section | Topic | Points |
|---|---|---:|
| 0 | Student information and import check | 2 |
| 1 | Variables, operators, and strings | 4 |
| 2 | Conditionals and loops | 8 |
| 3 | Lists, tuples, and dictionaries | 8 |
| 4 | Useful Python iteration patterns | 8 |
| 5 | NumPy basics | 12 |
| 6 | Classes and imports | 8 |
| | **Project 1A total** | **50** |

Every TODO in `project1.py` is labeled with its individual point value. Each
graded function is worth 4 points: 3 from its public tests and 1 from a hidden
test of documented edge cases or specifically required syntax
(`student_info` is 2 points, public only). Hidden tests do not introduce
requirements absent from the notebook or function docstrings.

## Submission checklist

- Complete `student_info()` in `project1.py`.
- Replace every `NotImplementedError` associated with a TODO.
- Keep all provided public names and signatures unchanged.
- Use only the dependencies in `requirements.txt`.
- Run `pytest -q test_project1.py` and review any failures.
- Submit only `project1.py` to the **Project 1A** Gradescope assignment.

Project 1B is a separate 50-point Webots screenshot submission. See
`PROJECT_1_OVERVIEW.md` in the parent directory for the complete 100-point
structure.
