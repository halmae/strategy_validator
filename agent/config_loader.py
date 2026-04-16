"""Config loader — reads settings.yaml and prompt files from config/.

Single source of truth for paths and runtime settings.
Other modules (schema_loader, runner, main) import from here.
"""
import yaml
from pathlib import Path
from functools import lru_cache

# Project root is the parent of agent/
ROOT = Path(__file__).parent.parent

# config/ location is fixed (not configurable, would be circular)
CONFIG_DIR = ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Load settings.yaml. Cached — modify file requires fresh process."""
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


def get_path(key: str) -> Path:
    """Resolve a path defined in settings.yaml under 'paths' to an absolute Path.

    Example:
        get_path("schema_dir")  -> /abs/path/to/validation_agent/schema
    """
    settings = load_settings()
    rel = settings.get("paths", {}).get(key)
    if rel is None:
        raise KeyError(f"paths.{key} not defined in settings.yaml")
    return ROOT / rel


def load_prompt(name: str) -> str:
    """Load a prompt template from <prompts_dir>/<name>.md"""
    path = get_path("prompts_dir") / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def get_model_config() -> dict:
    """Convenience accessor for the model section."""
    return load_settings().get("model", {})


def get_language() -> str:
    return load_settings().get("language", "english")
