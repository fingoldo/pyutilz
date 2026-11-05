# Gate defects (py-ci-shared)

Residues of baselined debt that are defects in a py-ci-shared gate, not in the code.

## G-01 patch_target_read_at_call (2026-11-05)

- Gate: `patch_target_read_at_call`
- Hits: 38, all `mock.patch` targets inside the `openrouter_provider` and `kernel_tuning.cache` facade packages.
- Why it is a gate defect: those packages read the patched names late, through `sys.modules` / `_pkg()`, on purpose to keep the import graph acyclic. The patches are live. The gate does not model that lookup.
- Disposition: not wired, reported to py-ci-shared as a gate defect. No code change.
- Status: RESOLVED as gate defect. Tracked in `audits/implemented/2026-11-05/TRACKER.md`.
