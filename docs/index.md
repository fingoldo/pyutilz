# pyutilz

[![CI](https://github.com/fingoldo/pyutilz/workflows/CI/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![MyPy](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml/badge.svg)](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml)
[![Black](https://github.com/fingoldo/pyutilz/workflows/Black/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![codecov](https://codecov.io/gh/fingoldo/pyutilz/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/pyutilz)
[![codecov-numba](https://codecov.io/gh/fingoldo/pyutilz/branch/master/graph/badge.svg?flag=numba-disabled)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![numba coverage](https://github.com/fingoldo/pyutilz/actions/workflows/numba-coverage.yml/badge.svg)](https://github.com/fingoldo/pyutilz/actions/workflows/numba-coverage.yml)
[![PyPI](https://img.shields.io/pypi/v/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![Python](https://img.shields.io/pypi/pyversions/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fingoldo/pyutilz/blob/master/LICENSE)
[![docs](https://github.com/fingoldo/pyutilz/actions/workflows/docs.yml/badge.svg)](https://fingoldo.github.io/pyutilz/)

A Python utilities library covering data-frame ops, databases, web/cloud, system monitoring, parallelism, and a unified async LLM-provider interface. The core has few hard dependencies (`numba`, `joblib`, `portalocker` -- used throughout `pyutilz.core.pythonlib`, the module nearly every other subpackage imports); every domain-specific extra beyond that ships as an optional extras group so `pip install pyutilz` stays light and you opt into what you need.

See the [Modules](modules.md) reference for what each sub-package does, and the guides below for the subsystems with real design decisions behind them.

## Installation

```bash
pip install pyutilz[all,dev]          # full install (recommended)

pip install pyutilz                   # core only, no hard deps
pip install pyutilz[dataframes]       # pandas + numpy + pyarrow + polars
pip install pyutilz[database]         # SQLAlchemy + psycopg2 + pymysql
pip install pyutilz[web]              # selenium, requests, undetected-chromedriver
pip install pyutilz[cloud]            # boto3 + google-cloud-storage
pip install pyutilz[nlp]              # spacy + nltk + tiktoken + jellyfish
pip install pyutilz[llm]              # anthropic + google-genai + httpx + tenacity + pydantic
pip install pyutilz[system]           # psutil + numba + GPUtil + tqdm + py-cpuinfo
pip install pyutilz[dev]              # pytest, ruff, black, mypy, bandit
```

For development:

```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e ".[all,dev]"
pre-commit install
pytest
```

Requires Python 3.8+. Tested on 3.8 through 3.14.

## Quick examples

**Shrink a DataFrame's memory** — auto-downcast every column to the
narrowest type that holds the data without precision loss; typical
50-80% reduction on real-world tabular data:

```python
from pyutilz.data.pandaslib import optimize_dtypes
df = optimize_dtypes(df)
```

**Unified LLM interface across 7 providers** — same `generate()` /
`generate_json()` surface (plus `generate_stream()` on the
OpenAI-compatible providers); switch by changing one string. See the
[LLM providers guide](guides/llm_providers.md) for the full picture.

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")
```

**Per-host kernel-tuning cache** — auto-tunes which CUDA/numba/cupy
variant of a hot kernel to run, per hardware fingerprint. See the
[kernel tuning cache guide](guides/kernel_tuning_cache.md).

```python
from pyutilz.performance.kernel_tuning import KernelTuningCache, hw_fingerprint

cache = KernelTuningCache.load_or_create()
print(hw_fingerprint())
```

**Sidecar-verified pickle load** — refuses to unpickle a payload whose
`.sha256` companion is missing or mismatched. See the
[safe pickle guide](guides/safe_pickle.md).

```python
from pyutilz.core.safe_pickle import safe_dump, safe_load

safe_dump(model, "model.pkl")   # writes model.pkl + model.pkl.sha256
model = safe_load("model.pkl")  # verifies the sidecar before unpickling
```

## Security

- Database operations use parameterised queries; `validate_sql_identifier`
  rejects identifiers that don't match `^[A-Za-z_][A-Za-z0-9_]*$`.
- `subprocess` calls never pass `shell=True`.
- Bandit (`bandit -ll`) and Vulture dead-code scans run as blocking
  gates both locally (pre-commit) and in CI, triaged to zero findings.
- LLM API keys are read from `.env` via `pydantic-settings`; the file is
  gitignored and a [detect-secrets](https://github.com/Yelp/detect-secrets)
  pre-commit hook blocks accidental in-source commits.

## License

MIT — see [LICENSE](https://github.com/fingoldo/pyutilz/blob/master/LICENSE).
