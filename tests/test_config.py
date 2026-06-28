"""Tests for layered config loading, incl. packaged default.yaml discovery.

Guards the fix that ships configs/default.yaml INSIDE the package (loaded via
importlib.resources) so the default config is found after a plain
``pip install`` — not just in the source tree.
"""

from tcrconsensus.config import load_config, _packaged_configs_dir


def test_packaged_configs_dir_exists():
    # The bundled default.yaml must be discoverable in source tree / editable / wheel.
    cfg_dir = _packaged_configs_dir()
    assert (cfg_dir / "default.yaml").is_file()


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg.io, dict)
    assert cfg.io.get("input_format") == "auto"
    # balanced consensus threshold default is 0.3 (see configs/default.yaml)
    assert cfg.consensus["balanced"]["threshold"] == 0.3


def test_preset_override():
    cfg = load_config(preset="high_purity")
    assert cfg.consensus["balanced"]["threshold"] == 0.5
    assert cfg.selection["default_objective"] == "high_purity"


def test_user_yaml_override(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("io:\n  input_format: vdjdb\n")
    cfg = load_config(user_yaml=str(user))
    assert cfg.io["input_format"] == "vdjdb"
