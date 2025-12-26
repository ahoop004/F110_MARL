# Rollback Procedure

**Purpose:** Emergency procedure to revert refactoring changes if critical issues are discovered.

---

## 🚨 When to Rollback

Trigger rollback if ANY of these occur:
- ❌ Baseline tests fail and can't be fixed quickly
- ❌ Training performance degrades >10%
- ❌ Checkpoint compatibility breaks
- ❌ Critical bug introduced that blocks work
- ❌ Scope creep makes refactor unmanageable

---

## 🔄 Quick Rollback (5 minutes)

### Option A: Switch to Backup Branch

If you're on `refactor/v2-pipeline`:

```bash
# 1. Stash any uncommitted work
git stash

# 2. Switch to backup branch
git checkout backup/pre-refactor

# 3. Verify you're back to working state
python3 -m pytest tests/unit/ -v

# 4. Continue work on backup branch
git checkout -b hotfix/issue-name
```

**You're back to v1.0-pre-refactor** ✅

---

### Option B: Reset to Tag

If backup branch is lost or corrupted:

```bash
# 1. Find the pre-refactor tag
git tag --list "v1.0-pre-refactor"

# 2. Create new branch from tag
git checkout -b restore/v1 v1.0-pre-refactor

# 3. Verify working state
python3 -m pytest tests/unit/ -v
```

---

## 📦 Partial Rollback (Revert Specific Changes)

If only certain changes are problematic:

### Revert Last Commit
```bash
git revert HEAD
```

### Revert Specific Commit
```bash
# Find the bad commit
git log --oneline

# Revert it
git revert <commit-hash>
```

### Revert Multiple Commits
```bash
# Revert commits from HEAD to specific commit
git revert <commit-hash>..HEAD
```

---

## 🔍 Diagnose Before Rollback

Before rolling back, try to identify the issue:

### Check Test Failures
```bash
# Run tests with verbose output
python3 -m pytest tests/ -vv --tb=long

# Run specific failing test
python3 -m pytest tests/path/to/test.py::TestClass::test_method -vv
```

### Check Git Diff
```bash
# See all changes since pre-refactor
git diff v1.0-pre-refactor

# See changes in specific file
git diff v1.0-pre-refactor -- path/to/file.py
```

### Check Recent Commits
```bash
# See last 10 commits
git log --oneline -10

# See detailed changes in last commit
git show HEAD
```

---

## 🛡️ Safe Rollback (Preserves Work)

If you want to save refactoring work before rolling back:

```bash
# 1. Create archive branch
git checkout refactor/v2-pipeline
git checkout -b archive/refactor-attempt-$(date +%Y%m%d)

# 2. Push to remote (if available)
git push origin archive/refactor-attempt-$(date +%Y%m%d)

# 3. Switch to backup
git checkout backup/pre-refactor

# 4. Your refactor work is saved in archive/ branch
```

---

## 📋 Post-Rollback Checklist

After rolling back:
- [ ] Run all tests to verify working state
- [ ] Check that training scenarios work
- [ ] Document what went wrong in `REFACTOR_TODO.md`
- [ ] Decide: abandon refactor, or retry with different approach?
- [ ] If retrying: create new refactor branch from backup

---

## 🔐 Recovery from Data Loss

### If Backup Branch is Lost

```bash
# Check reflog (git's safety net)
git reflog

# Find commit before refactoring started
# Look for entry like: "7589c10 HEAD@{5}: commit: Fix: DQN reset..."

# Create branch from that point
git checkout -b recovery/<commit-hash> <commit-hash>
```

### If Tags are Lost

```bash
# Check all refs
git show-ref

# If tag still exists remotely
git fetch --tags

# Recreate tag if needed
git tag v1.0-pre-refactor <commit-hash>
```

---

## 🚀 Resume Refactoring After Rollback

If you rolled back and want to try again:

```bash
# 1. Start fresh from backup
git checkout backup/pre-refactor

# 2. Create new refactor branch
git checkout -b refactor/v2-pipeline-attempt2

# 3. Apply lessons learned
# - Start with smaller changes
# - Test more frequently
# - Commit after each working phase

# 4. Reference your previous attempt (if archived)
git log archive/refactor-attempt-YYYYMMDD
git show archive/refactor-attempt-YYYYMMDD:<file>
```

---

## 📞 Emergency Contacts

If rollback fails or you need help:

1. **Check Git documentation:** `git help <command>`
2. **Check reflog:** `git reflog` (your safety net!)
3. **GitHub issues:** Open issue at repo (if applicable)
4. **Stack Overflow:** Git rollback questions

---

## 🧪 Validate Rollback Success

After rollback, verify:

```bash
# 1. Check git status
git status
git log --oneline -5

# 2. Run baseline tests
python3 -m pytest tests/unit/ -v

# 3. Verify scenarios work
python3 run.py --scenario scenarios/gaplock_ppo.yaml --episodes 10

# 4. Check file structure
ls -la src/f110x/
```

If all checks pass: **Rollback successful** ✅

---

## 📝 Notes

- **Backup branch `backup/pre-refactor` is read-only** - don't commit to it!
- **Tag `v1.0-pre-refactor` is permanent** - marks safe restore point
- **Refactor branch can be recreated** - work is never truly lost with git
- **When in doubt, commit!** - More commits = easier recovery

---

**Remember: Git is your safety net. You can always get back to a working state!**
