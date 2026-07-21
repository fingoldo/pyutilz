# Safe pickle & sidecar verification

## What is a sidecar?

A **sidecar** here is a small companion file, `<path>.sha256`, written next to a pickled artifact (`<path>`). It holds the sha256 digest of the payload, in the same one-line format GNU `sha256sum` produces. Before loading `<path>`, `pyutilz.core.safe_pickle` recomputes the digest of the bytes on disk and compares it against what the sidecar says it should be. If the sidecar is missing, or the digest doesn't match, the load is refused by default.

```python
from pyutilz.core.safe_pickle import safe_dump, safe_load

safe_dump(model, "model.pkl")   # writes model.pkl AND model.pkl.sha256
model = safe_load("model.pkl")  # recomputes the hash, compares, then unpickles
```

`safe_dump` writes atomically (dump to a per-process-and-thread temp file, fsync, `os.replace` onto the final path) so a crash mid-write leaves the previous file intact rather than a truncated one, and the sidecar is written only *after* the payload is fully on disk — the hash always matches the final bytes, never a partial write.

## Why this exists

Four separate pickle entry points across mlframe had independently converged on the same sidecar pattern before this module existed — clear signal that the pattern belongs in shared infrastructure rather than being re-invented per project. `pyutilz.core.safe_pickle` is that shared primitive; `mlframe.utils.safe_pickle` is now a thin backward-compatible re-export of it.

## What it protects against — and what it does NOT

This is the caveat that matters most, so it is worth stating plainly (the module's own docstring in `src/pyutilz/core/safe_pickle.py` carries the same language verbatim):

> The sha256 sidecar is a **corruption / integrity check**, **not** an authenticity / tamper-resistance control. An attacker who can WRITE to the directory holding the payload can trivially rewrite BOTH the payload and its `.sha256` sidecar so the digest matches again — the load then succeeds and arbitrary pickle executes. The sidecar only defends against accidental corruption (truncated copy, mid-rename crash, bit-rot) and against a payload swapped WITHOUT a matching sidecar rewrite. It buys nothing against an adversary with write access to the same directory.

In plain terms: this stops "the cached artifact got half-written and now loads garbage" or "someone dropped a stray file into the cache directory without regenerating the sidecar." It does **not** stop "someone with write access to the directory deliberately swapped in a malicious pickle and updated the hash to match" — that requires a *keyed* integrity control (an HMAC with a secret key, or a detached cryptographic signature), which is explicitly out of scope here. Callers loading pickles from a directory an untrusted party can write to need that keyed control layered on top; `pyutilz` deliberately does not implement key management or distribution.

## Fail-closed by default

`verify_sidecar()` returns `False` (refuses the load) when the sidecar is missing, unless the environment variable `PYUTILZ_ALLOW_UNVERIFIED_PICKLE` is set truthy — which permits the legacy unverified path with a loud `WARN` log, intended only for migrating pre-existing un-sidecar'd artifacts. A digest **mismatch** always raises `PickleVerificationError`, regardless of the env var: a corrupted or swapped-without-resigning file never silently loads.

Projects with their own historical env var name for this opt-in (mlframe used `MLFRAME_ALLOW_UNVERIFIED_PICKLE` before adopting this shared module) can pass `env_var=` to `verify_sidecar` / `safe_load` to check their own name instead of the pyutilz default.

## API surface

- `verify_sidecar(path)` — `True` iff the sidecar exists and matches; `False` on missing/mismatch (subject to the env-var opt-in above).
- `write_sidecar(path)` — computes and writes the `.sha256` sidecar for an existing file. Idempotent.
- `safe_load(path)` — verifies then `pickle.load`s; raises `PickleVerificationError` on failure.
- `safe_dump(obj, path)` — atomic `pickle.dump` + sidecar write, so anything written with `safe_dump` is immediately loadable with `safe_load`.
