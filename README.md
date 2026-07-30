# Classes

College coursework archive for Mechanical Engineering, Computer Science, and related electives.

## Repository Layout

```text
courses/
  MECH-101-introduction-to-mechanical-engineering/
    README.md
    notes/
    homework/
    labs/
    projects/
    exams/
    resources/
  CS-201-data-structures/
    README.md
    notes/
    homework/
    projects/
```

## Large Files

Git LFS for large binary coursework such as PDFs, CAD files, images, archives, datasets, and media. After cloning:

```bash
git lfs install
git lfs pull
```

`.gitattributes`; `git lfs track`.

## Python Setup

This repo uses `uv` for lightweight Python tooling shared across classes.

```bash
make install
```

## Common Commands

```bash
make test       # run available Python, C++, and Java checks
make lint       # run pre-commit hooks
make format     # format supported source files
make check      # run lint and tests
make clean      # remove common local build/cache files
```
