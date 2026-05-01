---
name = "git-expert"
description = "Branching strategies, rebase vs merge, bisect, reflog recovery, hooks, and workflows"
tags = ["git", "version-control", "branching", "workflows", "devops"]
max_inject_tokens = 2000
---

# Git Expert

## Branching Strategies

**Trunk-based development** (recommended for CI/CD): short-lived feature branches (< 2 days), merge to `main` via PR. No long-lived branches. Feature flags for incomplete work.

**GitHub Flow**: `main` is always deployable. Feature branches from `main`, PR + review, merge, deploy. Simple and effective for most teams.

**Git Flow**: `main` + `develop` + feature/release/hotfix branches. Overhead justified only for versioned releases with parallel maintenance (mobile apps, on-prem software).

Branch naming: `feat/short-description`, `fix/issue-123`, `chore/update-deps`. Keep it under 50 chars.

## Rebase vs Merge

**Rebase** for feature branches before merging — linear history, easier `bisect` and `log`:

```bash
git fetch origin
git rebase origin/main
# Resolve conflicts per-commit (easier than one big merge conflict)
git push --force-with-lease  # Safe force push (fails if remote changed)
```

**Merge** for integrating into `main` — preserves branch topology, non-destructive:

```bash
git merge --no-ff feat/my-feature  # Always create merge commit
```

Never rebase public/shared branches. `--force-with-lease` over `--force` — prevents overwriting teammates' pushes.

Interactive rebase for cleaning up before PR: `git rebase -i HEAD~5` to squash WIP commits, reword messages, reorder.

## Bisect

Find the commit that introduced a bug in O(log n) steps:

```bash
git bisect start
git bisect bad              # Current commit is broken
git bisect good v1.2.0      # This tag was working

# Git checks out middle commit. Test it, then:
git bisect good  # or  git bisect bad

# Automate with a test script:
git bisect run cargo test --test regression_test
git bisect reset  # When done
```

Works across hundreds of commits in under 20 steps. The test script must exit 0 (good) or 1-127 except 125 (bad). Exit 125 means "skip this commit" (can't test).

## Reflog Recovery

Git almost never deletes data. `reflog` is your safety net:

```bash
# Accidentally reset --hard or deleted a branch?
git reflog                         # Find the commit hash
git checkout -b recovery abc1234   # Restore it

# Recover a dropped stash
git fsck --no-reflog | grep "dangling commit"
git show <hash>                    # Inspect, then cherry-pick or branch

# Undo a bad rebase
git reflog
git reset --hard HEAD@{5}          # Go back to pre-rebase state
```

Reflog entries expire after 90 days (reachable) or 30 days (unreachable). Run `git gc` sparingly.

## Hooks

Pre-commit hooks for quality gates:

```bash
# .git/hooks/pre-commit (or use pre-commit framework)
#!/bin/sh
cargo fmt -- --check || { echo "Run cargo fmt"; exit 1; }
cargo clippy -- -D warnings || exit 1
cargo test || exit 1
```

Use the `pre-commit` framework for language-agnostic hooks with `.pre-commit-config.yaml`. Hooks run client-side — enforce the same checks in CI (hooks can be skipped with `--no-verify`).

Useful hooks: `pre-commit` (lint/format), `commit-msg` (conventional commit format), `pre-push` (full test suite), `prepare-commit-msg` (template injection).

## Workflows

**Conventional Commits**: `type(scope): description` — enables automated changelogs and semantic versioning.

```
feat(auth): add JWT refresh token rotation
fix(api): handle null response in user endpoint
chore(deps): update tokio to 1.40
```

Types: `feat` (minor bump), `fix` (patch bump), `BREAKING CHANGE` in footer (major bump). `chore`, `docs`, `refactor`, `test`, `ci` for non-release changes.

**Useful commands**:
- `git log --oneline --graph --all` — visualize branch topology.
- `git diff --stat main..HEAD` — summary of changes vs main.
- `git shortlog -sn --no-merges` — contributor ranking.
- `git blame -L 10,20 file.rs` — who changed these lines and when.
- `git stash push -m "description" -- path/to/file` — stash specific files.
- `git cherry-pick -x <hash>` — apply commit with reference to original.
- `git worktree add ../feature-branch feature-branch` — parallel checkout without stashing.
