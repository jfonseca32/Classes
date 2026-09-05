# Project 1B — Webots Installation Check

**Points:** 50
**Submission:** one screenshot (`.png`, `.jpg`, `.jpeg`, or `.pdf`)
**Gradescope assignment:** Project 1B — Webots Installation Check

## Goal

Install Webots, connect it to the Project 1 Python environment, and prove that
both work by running the provided verification world.

Project 1B is separate from the 50-point Python assignment (Project 1A).
Upload the setup screenshot directly to Gradescope; do not put it in a
notebook or PowerPoint.

## Prerequisite — the `cs3630_p1` environment

Complete the environment setup in the
[Project 1A instructions](../project_1a_python/PROJECT_1A_PYTHON_INSTRUCTIONS.md)
first: Miniforge, the `cs3630_p1` Python 3.12 environment, and
`python -m pip install -r requirements.txt`. Project 1B reuses that
environment and installs nothing new into Python.

Before starting:

- Complete these steps on the computer you plan to use for later Webots
  projects.
- Preserve the provided `verification/` directory structure.
- Use the course-specified Webots version if staff announce one; otherwise use
  the current stable version.
- Check the
  [official Webots system requirements](https://cyberbotics.com/doc/guide/system-requirements),
  especially on a virtual machine or remote desktop.

## Step 1 — Install Webots

Webots is a free, open-source robot simulator. Installing it requires
administrator rights and a few gigabytes of free disk space. Download the
installer for your operating system from an official source, then follow the
instructions for your platform below.

- [Cyberbotics download page](https://cyberbotics.com/#download)
- [Official Webots GitHub releases](https://github.com/cyberbotics/webots/releases)
- [Official installation procedure](https://cyberbotics.com/doc/guide/installation-procedure)

### macOS

1. Download the macOS disk image (`.dmg`) and open it.
2. Drag `Webots.app` into your `Applications` folder.
3. Launch Webots from `Applications`. If macOS blocks the first launch,
   Control-click `Webots.app` and choose **Open**, or allow it under
   **System Settings → Privacy & Security**.
4. On Apple Silicon, use the native build and leave **Open using Rosetta**
   disabled (Control-click the app and choose **Get Info** to check) unless
   course staff state otherwise.

### Linux

1. On supported Ubuntu systems, download the `.deb` package. Other
   distributions should follow the
   [official installation procedure](https://cyberbotics.com/doc/guide/installation-procedure)
   for the tarball or APT repository instead.
2. Install it from your download directory:

   ```bash
   sudo apt install ./webots-<version>-amd64.deb
   ```

   Replace the placeholder with the exact filename you downloaded. The
   leading `./` tells `apt` to install the local file rather than search its
   repositories.
3. Do not use the Snap package for this course; its sandbox can prevent
   custom Python controllers from finding your environment's dependencies.
4. Launch Webots from your application menu or by running `webots` in a
   terminal.

### Windows

1. Download the 64-bit Windows installer (`webots-<version>_setup.exe`).
2. Run the installer. If Windows SmartScreen warns about an unrecognized
   app, confirm the file came from one of the official sources above, then
   choose **More info → Run anyway**.
3. Follow the prompts; the default settings are fine.
4. Launch Webots from the Start menu. Use the native Windows application;
   WSL2 is not required unless course staff announce a WSL-specific workflow.

### First-launch check (all platforms)

Before configuring Python, confirm that Webots itself works:

1. In Webots, open any built-in example world (**File → Open Sample World**,
   or from the welcome screen).
2. Press the triangular **Run** button.
3. Confirm that the simulation time advances and the 3D view renders
   normally.

If a built-in example cannot run, fix the Webots installation or graphics
issue before continuing — see **Troubleshooting** below and the official
system requirements.

## Step 2 — Point Webots at the `cs3630_p1` Python

1. In a terminal with `cs3630_p1` active, print the interpreter's full path:

   - macOS or Linux: `which python`
   - Windows: `where.exe python`, then copy the first printed path

   The path must contain `cs3630_p1`. If it does not, run
   `conda activate cs3630_p1` and check again.

2. In Webots, open **Tools → Preferences** (on macOS, **Webots →
   Preferences**), find the **Python command** field on the **General** tab,
   and paste the complete path.

3. Apply, and restart Webots if it does not immediately use the new
   interpreter.

The absolute path ensures Webots launches Python from the `cs3630_p1`
environment instead of another system installation. See the official
[Using Python guide](https://cyberbotics.com/doc/guide/using-python) for
details.

## Step 3 — Run the verification world

The supplied files must retain this structure:

```text
project_1b_webots/
├── PROJECT_1B_WEBOTS_SETUP_INSTRUCTIONS.md
└── verification/
    ├── controllers/
    │   └── setup_check/
    │       └── setup_check.py
    └── worlds/
        └── setup_check.wbt
```

1. In Webots, select **File → Open World** and open
   `verification/worlds/setup_check.wbt`.
2. Press the triangular **Run** button.
3. Confirm that the green cylinder moves from side to side.
4. Show the console if it is hidden and confirm that it reports your
   `cs3630_p1` Python path, the NumPy version, and:

   ```text
   CS 3630 WEBOTS SETUP SUCCESS
   ```

The moving object proves that simulation time is advancing. The console lines
prove that Webots launched Python from your environment and that the
environment's packages import correctly. If the console instead shows
`CS 3630 WEBOTS SETUP FAILURE`, read its message — it names what is missing.

## Step 4 — Take and submit one screenshot

Take one readable screenshot showing all of the following at the same time:

- the Webots application window with the `setup_check.wbt` world and green
  cylinder;
- simulation time greater than `0:00:00`, or another clear running indicator;
  and
- the console message `CS 3630 WEBOTS SETUP SUCCESS`.

Upload the image directly to the **Project 1B — Webots Installation Check**
assignment on Gradescope. Do not embed it in `project1.ipynb`, wrap it in a
PowerPoint, or submit it to the Project 1A programming assignment.

## Rubric — 50 points

| Evidence | Points |
|---|---:|
| Webots and `setup_check.wbt` are clearly visible | 10 |
| Simulation is running and the green cylinder is visible | 15 |
| Console shows `CS 3630 WEBOTS SETUP SUCCESS` | 15 |
| Screenshot is readable and uses an accepted format | 10 |
| **Total** | **50** |

An unreadable, cropped, or unrelated screenshot receives only the points
supported by visible evidence. Contact course staff before the deadline if an
operating-system, permissions, accessibility, or graphics limitation prevents
you from completing the setup.

## Troubleshooting

### Console shows `CS 3630 WEBOTS SETUP FAILURE` or a wrong Python path

- Repeat Step 2: activate `cs3630_p1`, copy the path again, paste it into
  Webots preferences, and restart Webots.
- If the failure message names NumPy, install the Project 1A
  `requirements.txt` into `cs3630_p1` and reload the world.

### Webots cannot start the controller at all

- Repeat the first-launch check from Step 1: if a built-in sample world
  cannot run either, the problem is the Webots installation or graphics, not
  Python.
- Restart Webots after changing the Python command.
- Check the first error shown in the Webots console.
- Preserve the `verification/controllers` and `verification/worlds` directory
  relationship.

### `ModuleNotFoundError: No module named 'controller'`

- Do not run `setup_check.py` directly from a terminal. Webots supplies the
  `controller` module when Webots launches the controller.

### Black or empty 3D view

- Update your graphics driver, compare your system with the official Webots
  requirements, and avoid unsupported virtual-machine or remote-display
  configurations.

## Official resources

- [Installing Webots](https://cyberbotics.com/doc/guide/installing-webots)
- [Using Python with Webots](https://cyberbotics.com/doc/guide/using-python)
- [Webots User Guide](https://cyberbotics.com/doc/guide/index)
