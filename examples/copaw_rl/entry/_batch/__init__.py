"""batch_run 内部实现包。

外部只 import 顶层 `from _batch import ...`；不要直接 import 子模块。
"""

from .log_tee import BatchStdoutTee
from .modes import (
    list_models_and_exit,
    list_packages_and_exit,
    parse_cli,
    resolve_models,
    run_full_batch,
    run_retry_from_dir,
)
from .providers import build_provider_config, build_provider_config_from_json
from .summary import print_trial_summary

__all__ = [
    "BatchStdoutTee",
    "build_provider_config",
    "build_provider_config_from_json",
    "list_models_and_exit",
    "list_packages_and_exit",
    "parse_cli",
    "print_trial_summary",
    "resolve_models",
    "run_full_batch",
    "run_retry_from_dir",
]
