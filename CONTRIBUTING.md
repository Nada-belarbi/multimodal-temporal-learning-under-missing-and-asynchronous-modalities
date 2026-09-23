# Team development workflow

## Assign responsibilities

The five roles are not assigned yet. Choose one primary owner per role, then set the GitHub **Assignees** field on its issues. Titles `[Member 1]` through `[Member 5]` identify roles, not people. The repository owner grants the required repository/project access once usernames are supplied; assignment itself does not grant write access. Agree a reviewer for cross-component interface changes.

## Daily work

1. Open [the project](https://github.com/users/Nada-belarbi/projects/2) and select an issue from the [task index](docs/TASKS.md).
2. Read its linked work package and prerequisite issues. Put it in **Ready** only when inputs/interfaces are available.
3. Assign yourself and move it to **In progress** when work actually starts.
4. Create a branch such as `feat/D04-pamap2-adapter`; reference the issue in your commits and PR.
5. Post a progress comment when work changes materially or a blocker appears, using the format below.
6. Open a PR with `Closes #<issue-number>`, actual test commands/results and artifact links; move to **In review**.
7. Request a teammate review. Complete the acceptance checklist, resolve review comments and merge before moving to **Done**. Do not close an issue merely because code exists.

## Progress comment format

```text
Completed:
Evidence (PR, tests, artifacts):
Blocked by (issue link and missing input):
Next action:
Expected next update (agreed by the owner):
```

Do not invent progress, test results, performance scores, estimates or deadlines. A blocker is recorded on the affected issue with a link to the prerequisite; keep the current status and explain what is needed. Dates and effort estimates are set by the team after checking availability.

## Review progress

- **Backlog**: status board for all tasks.
- **Team items** and **My items**: inspect work by assignee after assignments are made.
- **Priority board**: use after the team sets priorities.
- **Roadmap**: use after the team agrees dates.
- **Insights**: inspect task distribution; task counts are not effort estimates.
- Filter repository issues by title `[Member 1]`, `[Member 2]`, etc. to see one work package.
- Review In progress, In review and blocker comments at each team check-in. Check linked PRs and evidence before reporting completion.

New matching open issues are configured to enter the Project automatically with Backlog status. Verify automation behavior when creating future tasks. GitHub notifications follow each member's watch/subscription settings; this setup does not schedule an external monitoring service.

## Evidence and data

Implementation specifications describe work to build; no application is currently claimed as implemented. Keep raw datasets, credentials, local paths and generated run artifacts out of Git. Share only permitted fixtures and appropriate result summaries. Use explicit manifests, fixed partitions and measured evidence for model qualification.
