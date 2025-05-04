# 📦 Package Deployment and Release Workflow Documentation

# 📚 Table of Contents

- [🚀 Normal Development Workflow](#-normal-development-workflow)
- [🏷️ Creating a New Release](#️-creating-a-new-release)
  - [Step 1: Bump Version](#step-1-bump-version)
  - [Step 2: Review and Merge the Version PR](#step-2-review-and-merge-the-version-pr)
- [🔥 Running Deployments](#-running-deployments)
  - [Step 3: Deploy Python Package and Conda Package](#step-3-deploy-python-package-and-conda-package)
  - [Step 4: Create GitHub Release](#step-4-create-github-release)
- [📝 Optional: Manual Changelog Update](#-optional-manual-changelog-update)
- [⚙️ Required GitHub Secrets](#️-required-github-secrets)
- [🔄 Updated Workflow Sequence](#-updated-workflow-sequence)
- [📝 Suggested Commit Message](#-suggested-commit-message)


# 🚀 Normal Development Workflow

- Develop new features or bug fixes on branches.
- Open Pull Requests (PRs) against `main` or `master`.
- When PRs are merged, all tests (`pytest`, `flake8`, `mypy`) automatically run to ensure code quality.

# 🏷️ Creating a New Release

## Step 1: Bump Version

- Go to the **Actions** tab on GitHub.
- Select the **"Version Tagging"** workflow.
- Click **"Run workflow"**.
- Choose the version increment:
  - `patch` (e.g., 1.0.0 → 1.0.1)
  - `minor` (e.g., 1.0.0 → 1.1.0)
  - `major` (e.g., 1.0.0 → 2.0.0)
- (Optional) Enter a **pre-release suffix** (e.g., `alpha`, `beta`, `rc`) if needed.
- Submit and trigger the workflow.

➡️ This automatically creates a **Pull Request** bumping the version.

## Step 2: Review and Merge the Version PR

- Review the created version bump PR.
- Merge it into `main` or `master` after review.
- When the PR is merged, it automatically triggers:
  - **Create Release Tag** workflow
  - **Generate Changelog** workflow

✅ The tag is created **programmatically** using a Personal Access Token (PAT) to allow triggering downstream workflows.

# 🔥 Running Deployments

> **Due to GitHub token limitations**, we now manually run the final packaging and release steps.

After the tag is created:

## Step 3: Deploy Python Package and Conda Package

- Go to **Actions** tab.
- Select **"📦 Python Package Deployment"** workflow.
- Click **"Run workflow"**.
- Choose the environment (`staging` or `production`) and publishing targets (`pypi`, `conda`, or `both`).
- Trigger the deployment.

➡️ This builds and uploads:

- PyPI packages (`sdist`, `wheel`)
- Conda packages (`.tar.bz2`, `.conda` files)

## Step 4: Create GitHub Release

- Go to **Actions** tab.
- Select **"🚀 Create GitHub Release"** workflow.
- Click **"Run workflow"**.
- Provide the **release version** in the `release_version` input field (e.g., `v1.0.0-alpha`).
- (Optional) Adjust wait time if necessary.
- Trigger the workflow.

➡️ This:

- Prepares the PyPI and Conda artifacts
- Creates a new GitHub release with auto-generated release notes
- Uploads the built packages into the GitHub release
- **Updates the `CHANGELOG.md` file automatically** with the release information

# 📝 Optional: Manual Changelog Update

If you need to update the changelog manually:

- Go to **Actions** tab.
- Select **"Generate CHANGELOG"** workflow.
- Click **"Run workflow"**.
- Provide the version number (e.g., `1.1.0`).
- Review and merge the automatically created changelog update PR.

# ⚙️ Required GitHub Secrets

Ensure these secrets are configured in your repository (Settings → Secrets and variables → Actions):

| Secret             | Purpose                                          |
|--------------------|--------------------------------------------------|
| `PYPI_API_TOKEN`    | For publishing to PyPI                           |
| `TEST_PYPI_API_TOKEN` | For publishing to TestPyPI                     |
| `ANACONDA_TOKEN`    | For uploading packages to Anaconda.org           |
| `CODECOV_TOKEN`     | (Optional) For uploading test coverage reports to Codecov |
| `GH_PAT`            | Personal Access Token used to push tags and trigger workflows properly |

# 🔄 Updated Workflow Sequence

Developer → Version Tagging Workflow → Version PR → Merge →
    → Create Release Tag + Generate Changelog →
    → (Manual) Run Python Package Deployment →
    → (Manual) Run GitHub Release Creation →
    → PyPI, Conda, and GitHub Release completed
    → CHANGELOG updated automatically during GitHub release