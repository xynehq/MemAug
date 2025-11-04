
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory by looking for config/config.yaml."""
    current = Path.cwd()
    # Look for config/config.yaml starting from current directory
    while current != current.parent:
        config_file = current / "config" / "config.yaml"
        if config_file.exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find project root (config/config.yaml not found)"
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
