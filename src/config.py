"""Configuration loading: YAML file + dotted command-line overrides."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["Config", "load_config", "add_config_args", "config_from_args"]


class Config(dict):
    """A dict whose keys are also reachable as attributes.

    Nested mappings are converted recursively, so ``cfg.train.lr`` works the
    same as ``cfg["train"]["lr"]``.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        """Return a plain, JSON/YAML-serialisable ``dict``."""
        return {k: _unwrap(v) for k, v in self.items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def _parse_scalar(text: str) -> Any:
    """Interpret an override value using YAML rules, plus loose float syntax.

    YAML 1.1 only recognises a float exponent when it has a decimal point, so
    ``1e-4`` would otherwise arrive as the string ``"1e-4"``.
    """
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def set_by_path(cfg: Config, dotted_key: str, value: Any) -> None:
    """Set ``cfg.a.b.c = value`` given ``dotted_key='a.b.c'``.

    Intermediate sections are created if missing so that new keys can be
    introduced from the CLI.
    """
    parts = dotted_key.split(".")
    node: Config = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Config):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = value


def get_by_path(cfg: Config, dotted_key: str, default: Any = None) -> Any:
    """Read ``cfg.a.b.c``, returning ``default`` when any level is missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def apply_overrides(cfg: Config, overrides: Iterable[str] | None) -> Config:
    """Apply ``key.path=value`` strings onto a copy-free ``cfg`` in place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"Override {item!r} is not of the form key.path=value"
            )
        key, raw_value = item.split("=", 1)
        set_by_path(cfg, key.strip(), _parse_scalar(raw_value.strip()))
    return cfg


def load_config(
    path: str | Path = "configs/default.yaml",
    overrides: Iterable[str] | None = None,
) -> Config:
    """Load a YAML config file and apply dotted overrides."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config root must be a mapping, got {type(raw)}")
    cfg = Config(copy.deepcopy(dict(raw)))
    cfg["config_path"] = str(path)
    return apply_overrides(cfg, overrides)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared ``--config`` / ``--set`` arguments to a parser."""
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config entry, e.g. --set train.lr=1e-4. Repeatable.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from a parsed namespace produced above."""
    return load_config(args.config, getattr(args, "overrides", None))


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    print(json.dumps(config_from_args(parser.parse_args()).to_dict(), indent=2))
