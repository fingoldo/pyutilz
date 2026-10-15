"""Package-presence checking and on-demand pip installation.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

from typing import Iterable, Optional, Union

from ._common import importlib, logger, subprocess, sys


def _package_present(import_name: str) -> Optional[bool]:
    """Whether ``import_name`` resolves, or None when the check itself raised.

    find_spec raises ModuleNotFoundError for a DOTTED name whose parent is missing ("nosuch.child"). The caller
    treats that as not installed, and keeps checking the other packages instead of aborting the whole call, as
    the function's documented "failures are logged, not raised" contract requires.
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception as exc:  # audit-exempt: non_neutral_except_fallback -- a raising find_spec means not installed
        logger.debug("find_spec raised for %s, so it is treated as unchecked: %s", import_name, type(exc).__name__)
        return None


def _warn_unchecked(unchecked_packages: list[str]) -> None:
    """Log, once, the packages whose presence check raised: they are installed as if missing."""
    if unchecked_packages:
        logger.warning("Could not check whether these packages are installed, so they are treated as missing: %s", unchecked_packages)


def _install_each(packages_to_install: list[str]) -> list[str]:
    """Pip-install each package in turn, isolating failures; return the ones that did not install (each logged at WARNING)."""
    failed_packages: list[str] = []
    for pkg in packages_to_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])  # nosec B603 - sys.executable is the running interpreter's own path (not attacker-controlled), and pkg names come from ensure_installed's own `packages` argument (developer-supplied dependency list), not from untrusted/network input. Using `sys.executable -m pip` (rather than a bare "pip" resolved via PATH search order) guarantees installation into the running interpreter's own environment and avoids Windows' CWD-before-PATH executable search order picking up an attacker-planted pip.exe/pip.bat.
        except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (one failed install shouldn't abort the rest)
            logger.warning("Failed to install package %s: %s", pkg, e)  # audit-exempt: unthrottled_hot_loop_log -- one line per package the caller asked to install; the list is short by construction
            failed_packages.append(pkg)
    return failed_packages


def ensure_installed(packages: Union[str, Iterable[str]], sep: str = " ") -> None:
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
        unchecked_packages: list[str] = []
        for pkg in packages:
            import_name = known_abbreviations.get(pkg, pkg)
            present = _package_present(import_name)
            if present is None:
                unchecked_packages.append(pkg)
            if not present:
                missing_packages.append(pkg)
        _warn_unchecked(unchecked_packages)
        if missing_packages:
            mes = f"Installing missing packages: {missing_packages}"
            logger.info(mes)
            failed_packages = _install_each(missing_packages)
            if failed_packages:
                # Still logged, not raised (the documented contract): the WARNING names every package left uninstalled,
                # so a later ImportError can be traced back to this call.
                logger.warning("ensure_installed: these packages are still not installed: %s", failed_packages)


# from pyutilz.core.pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order
