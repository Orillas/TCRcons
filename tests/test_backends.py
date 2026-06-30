"""Tests for the user-side backend installer (no network; dry-run only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcrconsensus.backends import (
    BACKEND_DIR_ENV,
    backends_dir,
    giana_script_path,
    gliph2_lib_path,
    install_giana,
    install_gliph2,
    install_tcrmatch,
    tcrmatch_bin_path,
    tcrmatch_iedb_path,
)


def test_backends_dir_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(BACKEND_DIR_ENV, str(tmp_path / "from_env"))
    resolved = backends_dir(override=str(tmp_path / "override"))
    assert resolved == tmp_path / "override"
    assert resolved.is_dir()  # created on demand


def test_backends_dir_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(BACKEND_DIR_ENV, str(tmp_path / "from_env"))
    assert backends_dir() == tmp_path / "from_env"


def test_backends_dir_default_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv(BACKEND_DIR_ENV, raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert backends_dir() == tmp_path / "tcrconsensus" / "backends"


def test_backends_dir_virtual_env(tmp_path, monkeypatch):
    monkeypatch.delenv(BACKEND_DIR_ENV, raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    assert backends_dir() == tmp_path / ".venv" / "tcrconsensus" / "backends"
    assert (tmp_path / ".venv" / "tcrconsensus" / "backends").is_dir()


def test_path_layout(tmp_path):
    assert giana_script_path(tmp_path) == tmp_path / "GIANA" / "GIANA4.1.py"
    assert tcrmatch_bin_path(tmp_path) == tmp_path / "TCRMatch" / "tcrmatch"
    assert tcrmatch_iedb_path(tmp_path) == tmp_path / "TCRMatch" / "data" / "IEDB_data.tsv"
    assert gliph2_lib_path(tmp_path) == (
        tmp_path / "clusTCR" / "clustcr" / "modules" / "gliph2" / "lib"
    )


def test_install_giana_dry_run_no_fs_touch(tmp_path):
    script = install_giana(tmp_path, dry_run=True)
    assert script == tmp_path / "GIANA" / "GIANA4.1.py"
    # dry-run must not clone anything
    assert not (tmp_path / "GIANA").exists()


def test_install_tcrmatch_dry_run_no_fs_touch(tmp_path):
    binary, iedb = install_tcrmatch(tmp_path, dry_run=True)
    assert binary == tmp_path / "TCRMatch" / "tcrmatch"
    assert iedb == tmp_path / "TCRMatch" / "data" / "IEDB_data.tsv"
    assert not (tmp_path / "TCRMatch").exists()


def test_install_gliph2_dry_run_no_fs_touch(tmp_path):
    lib = install_gliph2(tmp_path, dry_run=True)
    assert lib == gliph2_lib_path(tmp_path)
    assert not (tmp_path / "clusTCR").exists()


def test_install_giana_idempotent_when_present(tmp_path):
    # simulate a completed install
    (tmp_path / "GIANA").mkdir()
    (tmp_path / "GIANA" / "GIANA4.1.py").write_text("# stub")
    script = install_giana(tmp_path, dry_run=False)  # would clone, but skips since present
    assert script.exists()


def test_backends_module_imports_clean():
    # guards against syntax/typo regressions in the installer module
    import tcrconsensus.backends as b
    for name in ("install_giana", "install_tcrmatch", "backends_dir"):
        assert callable(getattr(b, name))


@pytest.mark.parametrize(
    "fn", [install_giana, install_tcrmatch]
)
def test_dry_run_funcs_accept_force(fn, tmp_path):
    # force + dry_run together must not raise and must not touch the fs
    fn(tmp_path, force=True, dry_run=True)
    assert not any(tmp_path.iterdir()) or all(
        p.is_dir() and not p.name.startswith(("GIANA", "TCRMatch")) for p in tmp_path.iterdir()
    )
