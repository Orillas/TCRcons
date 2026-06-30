"""User-side installation of external clustering backends (GIANA, TCRMatch).

Why this exists
---------------
GIANA and TCRMatch use **non-MIT, non-commercial licenses** (GIANA: UT Southwestern
academic-research-only; TCRMatch: Non-Profit OSL 3.0) and ship as source /
compiled C++ binaries plus reference data (IEDB). None of that can be
redistributed inside an MIT package published on PyPI. So instead of bundling
them, ``tcrconsensus install-backends`` clones and builds them **on the user's
machine** into a standard backends directory. tcrconsensus itself never carries
these binaries or data files — the user fetches them directly from upstream,
which is the license-clean path.

After install, the GIANA and TCRMatch wrappers auto-discover the backends
directory, so no ``TCR_*`` environment variables are required. The standard
layout is::

    <backends>/GIANA/GIANA4.1.py              # cloned from github.com/s175573/GIANA
    <backends>/TCRMatch/tcrmatch              # built (make) from github.com/IEDB/TCRMatch
    <backends>/TCRMatch/data/IEDB_data.tsv    # fetched from downloads.iedb.org
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Override the backends directory with this env var.
BACKEND_DIR_ENV = "TCRCONS_BACKEND_DIR"

_GIANA_REPO = "https://github.com/s175573/GIANA.git"
_TCRMATCH_REPO = "https://github.com/IEDB/TCRMatch.git"
#: clusTCR bundles the GLIPH2 ``irtools`` binary + v2.0 reference files under
#: ``clustcr/modules/gliph2/lib/``; clone it to obtain them. clusTCR itself is
#: MIT; the bundled irtools has its own (academic-use) license in that dir.
_CLUSCTR_REPO = "https://github.com/svalkiers/clusTCR.git"
#: IEDB reference data, fetched directly by the user (IEDB's own terms apply).
_IEDB_DATA_URL = "https://downloads.iedb.org/misc/TCRMatch/IEDB_data.tsv"


def backends_dir(override: str | os.PathLike | None = None) -> Path:
    """Resolve the backends directory.

    Priority: explicit override > ``$TCRCONS_BACKEND_DIR``
    > ``$VIRTUAL_ENV/tcrconsensus/backends`` (uv / pip venv)
    > ``$XDG_DATA_HOME/tcrconsensus/backends``
    > ``~/.local/share/tcrconsensus/backends``.

    Creates the directory if it does not exist.
    """
    if override:
        path = Path(override).expanduser()
    elif env := os.environ.get(BACKEND_DIR_ENV):
        path = Path(env).expanduser()
    elif venv := os.environ.get("VIRTUAL_ENV"):
        path = Path(venv) / "tcrconsensus" / "backends"
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        path = Path(base) / "tcrconsensus" / "backends"
    path.mkdir(parents=True, exist_ok=True)
    return path


def giana_script_path(base: Path | None = None) -> Path:
    """Expected location of the GIANA4.1.py script under the backends dir."""
    return (base or backends_dir()) / "GIANA" / "GIANA4.1.py"


def tcrmatch_bin_path(base: Path | None = None) -> Path:
    """Expected location of the compiled ``tcrmatch`` binary under the backends dir."""
    return (base or backends_dir()) / "TCRMatch" / "tcrmatch"


def tcrmatch_iedb_path(base: Path | None = None) -> Path:
    """Expected location of the IEDB reference TSV under the backends dir."""
    return (base or backends_dir()) / "TCRMatch" / "data" / "IEDB_data.tsv"


def gliph2_lib_path(base: Path | None = None) -> Path:
    """Expected location of the GLIPH2 irtools+ref bundle (inside a clusTCR
    checkout) under the backends dir."""
    return (base or backends_dir()) / "clusTCR" / "clustcr" / "modules" / "gliph2" / "lib"


def _run(cmd: list[str], dry_run: bool, cwd: Path | None = None) -> None:
    """Log and (unless dry_run) run a command, raising on non-zero exit."""
    pretty = " ".join(cmd)
    logger.info("  $ %s%s", f"cd {cwd} && " if cwd else "", pretty)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def install_giana(base: Path, force: bool = False, dry_run: bool = False) -> Path:
    """Clone GIANA (pure Python, no build step) into ``base/GIANA``.

    Returns the path to ``GIANA4.1.py``.
    """
    target = base / "GIANA"
    script = target / "GIANA4.1.py"
    if script.exists() and not force:
        logger.info("GIANA already installed at %s (use --force to reinstall)", target)
        return script
    if target.exists() and not dry_run and force:
        shutil.rmtree(target)
    logger.info("Installing GIANA -> %s", target)
    _run(["git", "clone", "--depth", "1", _GIANA_REPO, str(target)], dry_run)
    if not dry_run and not script.exists():
        raise FileNotFoundError(f"Clone finished but {script} was not found")
    return script


def install_tcrmatch(
    base: Path, force: bool = False, dry_run: bool = False
) -> tuple[Path, Path]:
    """Clone + build TCRMatch and fetch its IEDB reference data.

    Requires ``git``, ``g++`` (with OpenMP), and network access for the IEDB
    data fetch. Returns ``(binary_path, iedb_data_path)``.
    """
    target = base / "TCRMatch"
    binary = target / "tcrmatch"
    iedb = target / "data" / "IEDB_data.tsv"
    if binary.exists() and iedb.exists() and not force:
        logger.info("TCRMatch already installed at %s (use --force to reinstall)", target)
        return binary, iedb
    if target.exists() and not dry_run and force:
        shutil.rmtree(target)
    logger.info("Installing TCRMatch (clone + make + IEDB data) -> %s", target)
    _run(["git", "clone", "--depth", "1", _TCRMATCH_REPO, str(target)], dry_run)
    _run(["make"], dry_run, cwd=target)  # needs g++ + OpenMP
    if not dry_run:
        (target / "data").mkdir(parents=True, exist_ok=True)
    _run(["curl", "-fL", _IEDB_DATA_URL, "-o", str(iedb)], dry_run)
    if not dry_run:
        if not binary.exists():
            raise FileNotFoundError(
                f"Build finished but {binary} was not found (is g++ installed?)"
            )
        if not iedb.exists():
            raise FileNotFoundError(
                f"IEDB data fetch finished but {iedb} was not found"
            )
    return binary, iedb


def install_gliph2(base: Path, force: bool = False, dry_run: bool = False) -> Path:
    """Clone clusTCR (MIT) to obtain the bundled ``irtools`` binary + GLIPH2 v2.0
    reference files (``clustcr/modules/gliph2/lib/``).

    The irtools binary is Linux-only. Returns the lib directory path.
    """
    target = base / "clusTCR"
    lib = gliph2_lib_path(base)

    def _has_irtools() -> bool:
        return lib.is_dir() and any(p.name.startswith("irtools") for p in lib.iterdir())

    if _has_irtools() and not force:
        logger.info("GLIPH2 (via clusTCR) already installed at %s (use --force to reinstall)", target)
        return lib
    if target.exists() and not dry_run and force:
        shutil.rmtree(target)
    logger.info("Installing GLIPH2 (clone clusTCR for irtools + reference) -> %s", target)
    _run(["git", "clone", "--depth", "1", _CLUSCTR_REPO, str(target)], dry_run)
    if not dry_run and not _has_irtools():
        raise FileNotFoundError(
            f"Clone finished but no irtools binary under {lib} "
            "(expected clustcr/modules/gliph2/lib/irtools*)"
        )
    return lib
