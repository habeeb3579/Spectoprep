# 📊 GitHub Issues Management Setup

This guide documents how to set up **automatic GitHub Issues Management** with workflows using GitHub Actions, ProjectV2 (Beta Projects), and API integrations.

It covers:

* How to create a GitHub Project for issues.
* How to find the Project ID, Field IDs, and Option IDs.
* How to set up GitHub Actions automation.
* How to generate and configure necessary tokens.
* Common troubleshooting tips.

---

# 📅 Step 1: Create a GitHub Project for Issue Management

1. Go to your GitHub account or organization.
2. Click on `Projects` > `New project`.
3. Choose **Board**, **Table**, or **Roadmap** view.

   * For simple issue tracking, **Table** is recommended.
4. Name your project (e.g., `Issues Triage and Management`).
5. Save.

Optional:

* Add columns like "Todo", "In Progress", and "Done".
* Add fields like "Status", "Assignee", "Labels", etc.

> **Note**: GitHub Projects (Beta) are now "ProjectV2".

---

# 🔢 Step 2: Find Your Project ID, Field IDs, and Option IDs

You need these to automate assigning issues to your project.

### 2.1 Open the GitHub GraphQL Explorer

* Go to [GitHub GraphQL Explorer](https://docs.github.com/en/graphql/overview/explorer)
* Authorize if needed.

### 2.2 Run This Query to List Your Projects

```graphql
query {
  viewer {
    projectsV2(first: 10) {
      nodes {
        id
        title
      }
    }
  }
}
```

* Find the project with the name you created.
* Copy the **Project ID** (looks like `PVT_kwHOAXXXXX`).

### 2.3 Find Project Field IDs (like Status, Labels, etc.)

Run:

```graphql
query {
  node(id: "YOUR_PROJECT_ID") {
    ... on ProjectV2 {
      fields(first: 20) {
        nodes {
          id
          name
          dataType
          ... on ProjectV2SingleSelectField {
            options {
              id
              name
            }
          }
        }
      }
    }
  }
}
```

* This gives all fields like `Status`, `Labels`, `Assignees`.
* Copy the `Field ID` for "Status".
* Copy the `Option ID` for "Todo", "In Progress", etc.

---

# 🔑 Step 3: Set Up Tokens

### 3.1 GitHub Token (`GITHUB_TOKEN`)

Automatically available inside every GitHub Action as `${{ secrets.GITHUB_TOKEN }}`.

* Used for commenting on issues, labeling issues, etc.
* Already injected by GitHub, no manual setup needed.

### 3.2 Project Token (`PROJECT_TOKEN`)

Needed for making **GraphQL** mutation requests to assign issues to Projects.

**Steps to create:**

1. Go to **GitHub Settings > Developer Settings > Personal Access Tokens**.
2. Create a **Fine-grained Token** OR a **Classic Token**.
3. Grant permissions:

   * **Repository** access: `Read and Write` for `Issues`.
   * **Organization** access (if using org projects).
   * Scopes needed: `project`, `issues`, `read/write`.
4. Save the token.
5. Add it to your repository secrets as `PROJECT_TOKEN`.

> ⚠️ Fine-grained tokens are preferred over classic PATs.

---

# 🔧 Step 4: Example GitHub Actions Workflow for Issue Management
```yaml
name: Issue Management

on:
  issues:
    types: [opened, labeled, unlabeled, reopened]
  issue_comment:
    types: [created]

permissions:
  contents: write
  issues: write

jobs:
  first-response:
    runs-on: ubuntu-latest
    if: github.event_name == 'issues' && github.event.action == 'opened'
    steps:
      - name: Initial response to new issues
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: 'Thanks for opening this issue! We will look into it soon.'
            });

  triage-issue:
    runs-on: ubuntu-latest
    if: github.event_name == 'issues' && github.event.action == 'opened'
    steps:
      - name: Add triage label for new issues
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            if (!context.payload.issue.labels.some(label => label.name === 'triage')) {
              github.rest.issues.addLabels({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                labels: ['triage']
              });
            }

  assign-to-project:
    runs-on: ubuntu-latest
    if: github.event_name == 'issues' && (github.event.action == 'opened' || github.event.action == 'labeled')
    steps:
      - name: Assign issues to project
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.PROJECT_TOKEN }}
          script: |
            const PROJECT_ID = 'your_project_id';
            const STATUS_FIELD_ID = 'your_status_field_id';
            const TODO_OPTION_ID = 'your_todo_option_id';

            const addToProjectMutation = `
              mutation {
                addProjectV2ItemById(input: {
                  projectId: "${PROJECT_ID}",
                  contentId: "${context.payload.issue.node_id}"
                }) {
                  item {
                    id
                  }
                }
              }
            `;

            try {
              await github.graphql(addToProjectMutation);
              console.log('Successfully added issue to project!');
            } catch (error) {
              console.error('Error adding to project:', error);
            }
```

> Replace `your_project_id`, `your_status_field_id`, and `your_todo_option_id` with real IDs you fetched.

---

# 🌍 Extra: Handling Common Errors

| Problem                                  | Solution                                                                                        |
| :--------------------------------------- | :---------------------------------------------------------------------------------------------- |
| `Resource not accessible by integration` | You are using a token without enough permissions. Make sure `PROJECT_TOKEN` has correct scopes! |
| `GraphQL forbidden error`                | You used the wrong token or missing trusted permissions for projects.                           |
| `No matching project`                    | Project ID or field ID is wrong. Double check via GraphQL queries.                              |
| `403 error on GitHub REST API`           | Make sure your GitHub token is used correctly and has repo access enabled.                      |

---

# 🎓 Final Notes

* Use **GraphQL** Explorer heavily when debugging!
* Keep your **IDs** and **Secrets** properly stored in GitHub Settings > Secrets and variables > Actions.
* Monitor **workflow runs** in the GitHub Actions tab.

---

# 🚀 Related Resources

* [GitHub GraphQL Explorer](https://docs.github.com/en/graphql/overview/explorer)
* [GitHub Actions Documentation](https://docs.github.com/en/actions)
* [GitHub Projects (Beta)](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects)
* [Managing fine-grained PATs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
* [GitHub REST API for Issues](https://docs.github.com/en/rest/issues)

---

# ✨ You now have a full GitHub Issues Management system with professional automation!
