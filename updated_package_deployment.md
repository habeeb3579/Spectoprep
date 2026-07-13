# Package deployment and release (current workflows)

## Packaging strategy

| Channel | How it ships |
|---------|----------------|
| **PyPI** | Automated in `release.yml` on `v*` tags (Trusted Publishing) |
| **Conda (`habeebest`)** | Automated in `release.yml`: `conda build` + `anaconda upload` |
| **CI** | Pip matrix in `ci.yml` (lint + OS × Python tests), plus conda env smoke via `ci/conda-environment.yml` |
| **Docs (GitHub Pages)** | `docs-pages.yml` builds Sphinx and deploys to the `gh-pages` branch on every push to `main` |
| **Docs (Read the Docs)** | Rebuilds via RTD GitHub integration (`.readthedocs.yaml`) on push |

## Normal development

1. Branch off `main` / `master`, open a PR.
2. CI runs automatically: ruff, mypy, pytest (pip), plus conda env smoke on Ubuntu.
3. Docs are built on the PR (`docs-pages.yml`); deploy to github.io happens only after merge to `main`.
4. Merge when green.

## Creating a release

### Step 1 — Bump version

1. Actions → **Release Prep** → Run workflow.
2. Choose `patch` / `minor` / `major`.
3. A PR updates `src/spectoprep/__init__.py` and `conda-recipe/meta.yaml`.

### Step 2 — Merge the version PR

Merging a PR labeled `version-bump` creates and pushes tag `vX.Y.Z`, which triggers **Release**.

### Step 3 — Automated PyPI + Conda + GitHub Release

`release.yml` on the tag:

1. Builds sdist and wheel, runs `twine check`
2. Publishes to PyPI via OIDC Trusted Publishing (`environment: pypi`)
3. Builds `conda-recipe/` and uploads to Anaconda.org channel **`habeebest`**
4. Creates a GitHub Release attaching both PyPI and conda artifacts

## Required GitHub configuration

| Name | Type | Purpose |
|------|------|---------|
| PyPI Trusted Publisher | PyPI project ↔ GitHub `release.yml` / `pypi` environment | Publish without API token |
| GitHub Environment `pypi` | Repo → Settings → Environments | Must exist; matches OIDC `environment: pypi` |
| `ANACONDA_TOKEN` | Secret (**required** for conda publish) | Upload to anaconda.org/habeebest |
| `CODECOV_TOKEN` | Secret (optional) | Coverage uploads |
| `PROJECT_TOKEN` | Secret (optional) | Issue → Project v2 automation |
| `PROJECT_ID`, `STATUS_FIELD_ID`, `TODO_OPTION_ID` | Variables (optional) | Project board field IDs |

### PyPI Trusted Publisher (required)

On [pypi.org](https://pypi.org) → project **spectoprep** → **Publishing** → Add a new pending/trusted publisher:

| Field | Value |
|-------|--------|
| Owner | `habeeb3579` |
| Repository | `Spectoprep` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Those must match the OIDC claims from the workflow (`repo:…:environment:pypi`, `workflow_ref: …/release.yml@…`). A mismatch yields `invalid-publisher`.

Also create a GitHub Environment named **`pypi`** (Settings → Environments) so the job’s `environment: pypi` is valid.

Create an Anaconda.org API token with upload rights for the `habeebest` channel and add it as `ANACONDA_TOKEN` under repository secrets.

## Sequence

```
Developer → Release Prep (bump PR) → merge (version-bump)
  → tag vX.Y.Z → Release workflow
      → PyPI (Trusted Publishing)
      → Conda (habeebest via ANACONDA_TOKEN)
      → GitHub Release (wheel/sdist + conda packages)
```
