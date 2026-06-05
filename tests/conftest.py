"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detectors.rule_based import RuleBasedDetector
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def config() -> dict:
    return load_config(str(PROJECT_ROOT / "config.yaml"))


@pytest.fixture
def rule_detector(config: dict) -> RuleBasedDetector:
    paths = config["paths"]
    return RuleBasedDetector(
        patterns_path=PROJECT_ROOT / paths["patterns_path"],
        keywords_path=PROJECT_ROOT / paths["toxicity_keywords_path"],
    )
