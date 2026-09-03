"""Package-presence checking and on-demand pip installation.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

from ._common import importlib, logger, subprocess, sys


def ensure_installed(packages, sep: str = " ") -> None:
    """Installs any packages from `packages` that are not importable yet, via pip.

    `packages` can be a single package name, a `sep`-separated string of names, or an
    iterable of names. Known import-name/package-name mismatches (e.g. scikit-learn/sklearn)
    are resolved before checking. Installation failures are logged, not raised.
    """
    known_abbreviations = {"scikit-learn": "sklearn", "imbalanced-learn": "imblearn"}
    if packages:
        if isinstance(packages, str):
            if sep in packages:
                packages = packages.split(sep)
            else:
                packages = [packages]
        missing_packages = []
        for pkg in packages:
            import_name = known_abbreviations.get(pkg, pkg)
            try:
                found = importlib.util.find_spec(import_name) is not None
            except Exception as e:
                # find_spec raises ModuleNotFoundError for a DOTTED name whose parent is missing
                # ("nosuch.child"). Outside the try that aborted the whole call at the CHECK stage,
                # so none of the earlier packages got installed -- against this function's
                # documented "failures are logged, not raised" contract.
                logger.debug("Failed to check presence of package %s: %s", pkg, e)
                found = False
            if not found:
                missing_packages.append(pkg)
        if missing_packages:
            mes = f"Installing missing packages: {missing_packages}"
            logger.info(mes)
            for pkg in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])  # nosec B603 - sys.executable is the running interpreter's own path (not attacker-controlled), and pkg names come from ensure_installed's own `packages` argument (developer-supplied dependency list), not from untrusted/network input. Using `sys.executable -m pip` (rather than a bare "pip" resolved via PATH search order) guarantees installation into the running interpreter's own environment and avoids Windows' CWD-before-PATH executable search order picking up an attacker-planted pip.exe/pip.bat.
                except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (one failed install shouldn't abort the rest)
                    logger.debug("Failed to install package %s: %s", pkg, e)


# from pyutilz.core.pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order

# ensure_installed("joblib")
