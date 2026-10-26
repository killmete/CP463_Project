"""Load and validate the YAML configuration into frozen dataclasses.

`default_config.yaml` (shipped inside the package) is the single source of default
values. A user config (or CLI overrides) is deep-merged on top of it, then every key
is validated:
unknown keys, missing keys and wrong types raise `ConfigError`.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"

CAMERA_BACKENDS = ("auto", "msmf", "dshow", "any")


class ConfigError(ValueError):
    """The configuration is invalid."""


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    source: int | str
    backend: str
    width: int
    height: int
    fps: float
    buffer_size: int
    read_timeout_s: float
    fps_override: float | None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    display: bool
    max_frames: int | None
    latency_budget_ms: float
    fps_window_s: float
    log_level: str


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    duration_s: float
    min_valid_frames_ratio: float


@dataclass(frozen=True, slots=True)
class EyesConfig:
    closed_ratio: float
    ear_fallback_closed: float
    microsleep_s: float
    sleep_s: float


@dataclass(frozen=True, slots=True)
class PerclosConfig:
    window_s: float
    warning: float
    alarm: float


@dataclass(frozen=True, slots=True)
class YawnConfig:
    mar_threshold: float
    min_duration_s: float
    count_window_s: float
    warning_count: int


@dataclass(frozen=True, slots=True)
class HeadNodConfig:
    pitch_delta_deg: float
    min_duration_s: float


@dataclass(frozen=True, slots=True)
class DistractionConfig:
    yaw_deg: float
    pitch_deg: float
    long_s: float
    short_total_s: float
    short_window_s: float


@dataclass(frozen=True, slots=True)
class NoFaceConfig:
    timeout_s: float


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    clahe: bool
    clahe_clip: float
    clahe_tile: int


@dataclass(frozen=True, slots=True)
class Config:
    capture: CaptureConfig
    runtime: RuntimeConfig
    calibration: CalibrationConfig
    eyes: EyesConfig
    perclos: PerclosConfig
    yawn: YawnConfig
    head_nod: HeadNodConfig
    distraction: DistractionConfig
    no_face: NoFaceConfig
    preprocess: PreprocessConfig


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load defaults, merge the YAML at `path` and then `overrides` on top, and validate."""
    data = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        data = _deep_merge(data, _read_yaml(Path(path)))
    if overrides:
        data = _deep_merge(data, overrides)
    config = _build(Config, data, "config")
    _check_semantics(config)
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return data


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise ConfigError(f"{where}: missing key(s) {sorted(missing)}")
    values = {name: _convert(data[name], hints[name], f"{where}.{name}") for name in names}
    return cls(**values)


def _convert(value: Any, hint: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    options = typing.get_args(hint) if typing.get_origin(hint) is not None else (hint,)
    for option in options:
        if option is type(None) and value is None:
            return None
        if option is bool and isinstance(value, bool):
            return value
        if option is int and isinstance(value, int) and not isinstance(value, bool):
            return value
        if option is float and isinstance(value, int | float) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ConfigError(f"{where}: must be a finite number, got {value!r}")
            return float(value)
        if option is str and isinstance(value, str):
            return value
    expected = " | ".join(getattr(o, "__name__", str(o)) for o in options)
    raise ConfigError(f"{where}: expected {expected}, got {value!r}")


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _check_semantics(config: Config) -> None:
    # Every number is a size, rate, duration, angle, ratio or count: all must be > 0.
    # (capture.source is an index, so 0 is valid there.)
    for section in dataclasses.fields(config):
        values = getattr(config, section.name)
        for f in dataclasses.fields(values):
            value = getattr(values, f.name)
            if f.name == "source" or isinstance(value, bool) or value is None:
                continue
            if isinstance(value, int | float) and value <= 0:
                raise ConfigError(f"config.{section.name}.{f.name}: must be > 0, got {value!r}")

    if config.capture.backend not in CAMERA_BACKENDS:
        raise ConfigError(f"config.capture.backend: must be one of {CAMERA_BACKENDS}")
    if config.runtime.log_level not in LOG_LEVELS:
        raise ConfigError(f"config.runtime.log_level: must be one of {LOG_LEVELS}")
    for where, value in [
        ("eyes.closed_ratio", config.eyes.closed_ratio),
        ("calibration.min_valid_frames_ratio", config.calibration.min_valid_frames_ratio),
        ("perclos.warning", config.perclos.warning),
        ("perclos.alarm", config.perclos.alarm),
    ]:
        if value > 1:
            raise ConfigError(f"config.{where}: must be a ratio in (0, 1], got {value!r}")
    for where, lower, upper in [
        ("eyes.microsleep_s < eyes.sleep_s", config.eyes.microsleep_s, config.eyes.sleep_s),
        ("perclos.warning < perclos.alarm", config.perclos.warning, config.perclos.alarm),
    ]:
        if not lower < upper:
            raise ConfigError(f"config: expected {where}")
    if config.distraction.short_total_s > config.distraction.short_window_s:
        raise ConfigError("config: expected distraction.short_total_s <= short_window_s")
