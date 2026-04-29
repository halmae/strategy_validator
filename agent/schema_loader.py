"""Schema loader — reads YAML schema files from the configured schema/ directory."""
import yaml
from .config_loader import get_path

STAGE_SCHEMA_MAP = {
    0: "classification",
    1: "stage_1",
    2: "stage_2",
    3: "stage_3",
}


def load_schema(name: str) -> dict:
    path = get_path("schema_dir") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def schema_for_stage(stage: int) -> dict:
    name = STAGE_SCHEMA_MAP.get(stage)
    if name is None:
        raise ValueError(f"No schema defined for stage {stage}")
    return load_schema(name)


def load_routing() -> dict:
    return load_schema("routing")
