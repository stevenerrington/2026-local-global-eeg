---
title: Working on this project
subtitle: Conventions for a shared folder without version control
---

The project lives in one shared, synced folder. These rules stop people overwriting each other.

## Ownership

Each analysis has one owner who edits its scripts and regenerates its derivatives. To change someone else's script, ask them or leave a comment; don't edit it directly.

:::{list-table}
:header-rows: 1
* - Analysis
  - Scripts
  - Owner
* - Preprocessing and ERPs (Fig. 1)
  - `01`–`02`, `07`, `10`, `14`, `27`
  - *name*
* - Learner models (Fig. 2)
  - `03`–`06`, `15`–`17`, `26`
  - *name*
* - Sources (Fig. 3)
  - `11`, `18`–`20`
  - *name*
* - DCM
  - `21`–`25`, `28`, `scripts/dcm/`
  - *name*
* - Notebook and this site
  - `notebooks/`, `site/`, `myst.yml`
  - *name*
:::

## Rules

1. **Never edit the same file at the same time.** The sync service cannot merge code; it makes "conflicted copy" files. If you see one, tell the owner before deleting anything.
2. **Explore in your own copy.** For example `notebooks/scratch_<initials>.ipynb` or `scripts/scratch_<initials>/`. The owner moves finished changes into the main script.
3. **Log every rerun** in `CHANGELOG.md`, one line each: date, who, what changed, which outputs were regenerated.
4. **Check before you report.** After changing an analysis, rerun the notebook and check the match table at the end (section 5). If a number moved, update the Results text and this site.
5. **Don't move or rename `derivatives/` folders.** Other people's scripts read them by path.
6. **Undoing mistakes:** previous versions of every file are kept by the sync service (*Version history*).
