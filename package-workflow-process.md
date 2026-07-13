# Workflow Usage Instructions

## Normal Development Workflow

1. Develop features in branches
2. Create pull requests to main/master
3. When merged, tests run automatically

## Creating a New Release

### Step 1: Bump Version
1. Go to Actions tab in GitHub
2. Select "Version Tagging" workflow
3. Click "Run workflow"
4. Choose version type (patch, minor, major)
5. Click "Run workflow" button
6. A PR will be created - review and merge it

### Step 2: Wait for Automated Processes
After merging the version bump PR:
1. A git tag is automatically created
2. The tag triggers both package building/publishing and GitHub release creation
3. No manual action is needed

## Manual Changelog Update (Optional)
If you want to manually update the changelog:
1. Go to Actions tab in GitHub
2. Select "Generate CHANGELOG" workflow
3. Click "Run workflow"
4. Enter version number
5. Click "Run workflow" button
6. A PR will be created - review and merge it

## Workflow Sequence Diagram

```
Developer → Version Tagging workflow → Version PR → Merge →
  → Git Tag Created → Triggers Package Deployment → PyPI/Conda Packages Published
                    → Triggers GitHub Release → GitHub Release Created
```

## Required Secret Configuration

For these workflows to function properly, set up these secrets in your repository settings:

- `PYPI_API_TOKEN`
- `TEST_PYPI_API_TOKEN`
- `ANACONDA_TOKEN`
- `CODECOV_TOKEN` (optional)