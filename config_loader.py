#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any

import yaml  # type: ignore


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML or JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs. pip install pyyaml")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def apply_config_to_env(cfg: Dict[str, Any]) -> None:
    """Apply selected config values to environment variables used by the pipeline."""
    # 환경 변수에 설정값 직접 적용
    for k, v in cfg.items():
        if v is not None:
            os.environ[k] = str(v)
    
def load_and_apply(config_path: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    apply_config_to_env(cfg)
    return cfg


