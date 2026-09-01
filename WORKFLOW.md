# RegulSense Team GitHub Workflow

This document defines the GitHub collaboration workflow for the RegulSense project.

## 1. Branching Strategy

The `main` branch contains stable and releasable code.

Direct commits to `main` should be avoided. All development work should be completed on separate feature branches and merged through Pull Requests.

### Branch Naming Convention

Branches follow the format:

```text
[type]/[short-description]

Supported branch types include:

feature/ - New functionality
fix/ - Bug fixes
docs/ - Documentation changes
refactor/ - Code restructuring
chore/ - Maintenance tasks

Examples:

feature/document-ingestion
feature/github-workflow-setup
fix/chunking-validation
docs/rag-architecture
refactor/retrieval-pipeline
chore/update-dependencies
Branch Lifecycle
Create a branch from main
Make focused changes
Commit changes using the commit convention
Push the branch to GitHub
Create a Pull Request
Request code review
Merge after approval
Delete the feature branch after merging
2. Commit Message Convention

RegulSense follows a structured commit message convention.

Format:

[type]: [description]

Supported commit types:

feat - Introduces a new feature
fix - Fixes a bug
docs - Documentation changes
refactor - Code restructuring without changing functionality
test - Adding or updating tests
chore - Maintenance or configuration changes

Examples:

feat: add regulatory document ingestion pipeline
fix: handle invalid document metadata
docs: update RAG architecture documentation
refactor: simplify retrieval service
test: add chunking pipeline tests
chore: update project dependencies
Why We Use This Convention

Consistent commit messages provide:

Clear project history
Easier code review
Better collaboration
Easier debugging and rollback
Support for automated changelog generation

Pull Request Review Process

All changes must be submitted through a Pull Request before merging into main.

PR Requirements

Each Pull Request should include:

A clear and descriptive title
A summary of changes
The reason for the changes
A link to the related GitHub issue
Testing or verification details
Review Requirements

Pull Requests require at least one approval before merging.

Code review focuses on:

Correctness
Code clarity
Data integrity
Test coverage
Security considerations
Commit message quality

Commit messages are also reviewed to ensure the project history remains clear and consistent.

4. GitHub Issue Tracking

Every new feature, bug fix, or significant task should begin with a GitHub issue.

Issue Requirements

Each issue should contain:

A clear action-oriented title
A description explaining why the work is needed
Completion criteria
Appropriate labels
An assigned team member
Issue Lifecycle
Create Issue
     ↓
Assign Team Member
     ↓
Create Feature Branch
     ↓
Implement Changes
     ↓
Create Pull Request
     ↓
Code Review
     ↓
Merge Pull Request
     ↓
Close Issue

Issues are closed when the corresponding Pull Request is successfully merged.