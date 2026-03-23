"""File and path utilities."""

import pathlib

import yaml


def get_package_path() -> pathlib.Path:
    """Return the root of the thinkdepthai package (src/thinkdepthai/)."""
    return pathlib.Path(__file__).parent.parent


def load_prompts(fn: str | pathlib.Path) -> dict[str, str]:
    """Load prompts from a YAML file relative to the package prompts/ directory."""
    if isinstance(fn, str):
        if not fn.endswith(".yaml"):
            fn += ".yaml"
        fn = get_package_path() / "prompts" / fn
    assert fn.exists(), f"Prompt file {fn} does not exist!"
    with fn.open(encoding="utf-8") as f:
        result: dict[str, str] = yaml.safe_load(f)
        return result


def load_yaml(file_path: pathlib.Path | str) -> dict:
    """Load a YAML file and return its contents."""
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    with file_path.open() as f:
        return yaml.safe_load(f)
