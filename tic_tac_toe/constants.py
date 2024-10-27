from pathlib import Path

CONFIGS_DIR: Path = Path(__file__).parents[1] / "configs"

DEFAULT_GAME_CFG: Path = CONFIGS_DIR / "game-cfg.json"

DEFAULT_TD_CFG: Path = CONFIGS_DIR / "td-learn-cfgs" / "td-learn-cfg.json"

OUTPUTS_DIR: Path = Path(__file__).parents[1] / "outputs"
