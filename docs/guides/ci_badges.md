# CI badges and the `badges` branch

The README shows live status badges for the pandas and polars version matrices. Their data does not live on `master`.
Each matrix workflow publishes a [shields.io endpoint](https://shields.io/badges/endpoint-badge) document to a separate
orphan branch, `badges`, and the README reads that document straight from the branch.

## How it works

1. `pandas-matrix.yml` and `polars-matrix.yml` run the version matrix. After the matrix, a final job writes a
   `pandas.json` or `polars.json` document (`schemaVersion`, `label`, `message`, `color`) and pushes it to the `badges`
   branch as a single commit.
2. The README badges point at the raw JSON on that branch:

   ```
   https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/fingoldo/pyutilz/badges/pandas.json
   ```

   shields.io fetches the document and renders it. Each badge links to its workflow's run history.

## Why the branch is separate

- **`master`'s history stays about code.** A matrix run can produce several badge commits a day. Written to `master`, they
  would fill its log with bot commits that say nothing about the code.
- **No CI loop.** Every push to `master` starts the main CI. A bot commit there would start CI, and CI would write a badge,
  which would start CI again.
- **No changes to `master`'s rules.** Writing badges to `master` would need the bot to push there, which required checks
  and push restrictions on `master` are meant to prevent. Keeping the writes on `badges` leaves those rules alone.
- **Small and history-free.** An orphan branch holds only the two JSON files; it duplicates no code.

## What to keep in mind

- **Do not merge `badges` into `master`.** That would add the JSON files to `master` and break the separation above.
- **Deleting `badges` breaks the badges.** The matrix workflows recreate it on the next run, but until then the README
  shows an error. The workflows use `git ls-remote --heads` to check for the branch and create it as an orphan when missing.
- **A failing matrix is visible.** The badge message lists the versions that ran. The colour follows the matrix result:
  green on success, red on failure, grey when cancelled or skipped (`BADGE_COLORS` in `scripts/ci_version_matrix.py`).
