#!/usr/bin/env python3
"""Export task data and matched experiences by model version.

Steps:
1. Load Trinity Config from yaml.
2. Read experiences from data_processor.experience_pipeline.input_save_path (SQLite).
3. Filter experiences by given model version.
4. Read taskset data from config.buffer.explorer_input.tasksets.
5. Join by experience.info["task_index"] => {"taskset_id", "index"}.
6. Save merged data as a pickle file.

Unmatched data will be reported as warnings and skipped.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from datasets import load_dataset
from transformers import AutoTokenizer

from trinity.common.config import Config, TasksetConfig, load_config
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger

DEFAULT_SQLITE_TABLE = "pipeline_input"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export matched task and experience data by model version from Trinity config."
    )
    parser.add_argument("--config", required=True, help="Path to Trinity config yaml")
    parser.add_argument(
        "--model-version", type=int, required=True, help="Target model_version to export"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pkl path. If unset, auto-generate in current directory.",
    )
    parser.add_argument(
        "--sqlite-table",
        default=DEFAULT_SQLITE_TABLE,
        help=f"SQLite table name for experiences (default: {DEFAULT_SQLITE_TABLE})",
    )
    return parser.parse_args()


def _normalize_sqlite_path(input_save_path: str) -> str:
    """Normalize input_save_path to a local sqlite file path."""
    if input_save_path.startswith("sqlite:"):
        parsed = urlparse(input_save_path)
        if parsed.scheme != "sqlite":
            raise ValueError(f"Only sqlite scheme is supported, got: {input_save_path}")

        # sqlite URLs may use query-style path: sqlite:///x.db?path=/real/path.db
        query = parse_qs(parsed.query)
        if "path" in query and query["path"]:
            return os.path.abspath(unquote(query["path"][0]))

        raw_path = unquote(parsed.path)
        if not raw_path:
            raise ValueError(f"Invalid sqlite path: {input_save_path}")
        return os.path.abspath(raw_path)

    # Fallback: treat as local file path.
    return os.path.abspath(input_save_path)


def _extract_taskset_configs(config: Config) -> list[TasksetConfig]:
    explorer_input = config.buffer.explorer_input
    tasksets = list(explorer_input.tasksets)
    if not tasksets and explorer_input.taskset is not None:
        tasksets = [explorer_input.taskset]
    return tasksets


def _experience_to_export_dict(exp: Experience, tokenizer=None, logger=None) -> dict[str, Any]:
    """Convert an Experience object to a JSON/pickle-friendly dict."""
    prompt_text = exp.prompt_text
    response_text = exp.response_text
    should_rebuild_text = (not prompt_text or not response_text) and tokenizer is not None

    if should_rebuild_text:
        try:
            token_ids = exp.tokens.tolist()
            prompt_len = int(exp.prompt_length)
            prompt_len = max(0, min(prompt_len, len(token_ids)))

            if not prompt_text:
                prompt_text = tokenizer.decode(
                    token_ids[:prompt_len],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            if not response_text:
                response_text = tokenizer.decode(
                    token_ids[prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to rebuild prompt/response text from tokens: %s", exc)

    exp.prompt_text = prompt_text
    exp.response_text = response_text
    data = exp.to_dict()
    data["eid"] = asdict(exp.eid)
    return data


def _load_filtered_experiences(
    sqlite_path: str,
    table_name: str,
    model_version: int,
    logger,
    tokenizer=None,
) -> list[dict[str, Any]]:
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    query = (
        f"SELECT id, task_id, run_id, msg_id, model_version, experience_bytes, reward "
        f"FROM {table_name} WHERE model_version = ? ORDER BY id ASC"
    )

    rows: list[dict[str, Any]] = []
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, (model_version,))
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Failed querying table '{table_name}'. Please check --sqlite-table and DB schema."
            ) from exc

        for row in cursor:
            try:
                exp = Experience.deserialize(row["experience_bytes"])
            except Exception as exc:
                logger.warning(
                    "Skip row id=%s: failed to deserialize experience (%s)", row["id"], exc
                )
                continue

            exp.eid.task = row["task_id"]
            exp.eid.run = row["run_id"]
            exp.eid.suffix = row["msg_id"]
            exp.reward = row["reward"]
            exp.info["model_version"] = row["model_version"]

            rows.append(_experience_to_export_dict(exp, tokenizer=tokenizer, logger=logger))

    logger.info("Loaded %d experiences with model_version=%s", len(rows), model_version)
    return rows


def _build_needed_index_map(
    experiences: list[dict[str, Any]], logger
) -> tuple[dict[int, set[int]], dict[tuple[int, int], list[dict[str, Any]]]]:
    needed_indices: dict[int, set[int]] = {}
    grouped_experiences: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for exp in experiences:
        info = exp.get("info") or {}
        task_index = info.get("task_index")
        if not isinstance(task_index, dict):
            logger.warning("Skip experience without valid info.task_index")
            continue

        raw_taskset_id = task_index.get("taskset_id")
        raw_index = task_index.get("index")
        try:
            taskset_id = int(raw_taskset_id)  # type: ignore
            index = int(raw_index)  # type: ignore
        except (TypeError, ValueError):
            logger.warning(
                "Skip experience with invalid task_index: taskset_id=%s, index=%s",
                raw_taskset_id,
                raw_index,
            )
            continue

        needed_indices.setdefault(taskset_id, set()).add(index)
        grouped_experiences.setdefault((taskset_id, index), []).append(exp)

    return needed_indices, grouped_experiences


def _load_taskset_raw_tasks(
    taskset_configs: list[TasksetConfig],
    needed_indices: dict[int, set[int]],
    logger,
) -> dict[tuple[int, int], dict[str, Any]]:
    task_lookup: dict[tuple[int, int], dict[str, Any]] = {}

    for taskset_id, taskset_cfg in enumerate(taskset_configs):
        target_indices = needed_indices.get(taskset_id, set())
        if not target_indices:
            continue

        if taskset_cfg.storage_type != StorageType.FILE.value:
            logger.warning(
                "Taskset[%d] name=%s has unsupported storage_type=%s. Only FILE taskset is supported.",
                taskset_id,
                taskset_cfg.name,
                taskset_cfg.storage_type,
            )
            continue

        if not taskset_cfg.path:
            logger.warning(
                "Taskset[%d] name=%s has empty path. Skip loading.",
                taskset_id,
                taskset_cfg.name,
            )
            continue

        logger.info(
            "Loading taskset[%d] name=%s from path=%s (needed indices=%d)",
            taskset_id,
            taskset_cfg.name,
            taskset_cfg.path,
            len(target_indices),
        )

        dataset = load_dataset(
            taskset_cfg.path,
            name=taskset_cfg.subset_name,
            split=taskset_cfg.split,
        )

        dataset_size = len(dataset)
        invalid_indices = [i for i in target_indices if i < 0 or i >= dataset_size]
        if invalid_indices:
            logger.warning(
                "Taskset[%d] has %d out-of-range indices (dataset size=%d).",
                taskset_id,
                len(invalid_indices),
                dataset_size,
            )

        for idx in target_indices:
            if 0 <= idx < dataset_size:
                task_lookup[(taskset_id, idx)] = dict(dataset[idx])

    return task_lookup


def _build_export_records(
    grouped_experiences: dict[tuple[int, int], list[dict[str, Any]]],
    task_lookup: dict[tuple[int, int], dict[str, Any]],
    logger,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for (taskset_id, task_index), exp_list in sorted(grouped_experiences.items()):
        raw_task = task_lookup.get((taskset_id, task_index))
        if raw_task is None:
            logger.warning(
                "No matching task found for taskset_id=%s, task_index=%s (experience_count=%d)",
                taskset_id,
                task_index,
                len(exp_list),
            )
            continue

        records.append(
            {
                "taskset_id": taskset_id,
                "task_index": task_index,
                "raw_task": raw_task,
                "experiences": exp_list,
            }
        )

    return records


def _default_output_path(model_version: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_experience_model_version_{model_version}_{ts}.pkl"
    return str(Path.cwd() / filename)


def main() -> None:
    args = _parse_args()
    logger = get_logger("export_task_experience")

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    try:
        config = config.check_and_update()
    except Exception as exc:
        logger.warning(
            "Config check_and_update failed, fallback to raw config object. error=%s", exc
        )

    input_save_path = config.data_processor.experience_pipeline.input_save_path
    if not input_save_path:
        raise ValueError("config.data_processor.experience_pipeline.input_save_path is empty.")

    sqlite_path = _normalize_sqlite_path(input_save_path)
    logger.info("Reading experiences from sqlite: %s", sqlite_path)

    tokenizer = None
    model_path = config.model.model_path
    if model_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=config.model.trust_remote_code,
            )
            logger.info("Tokenizer initialized from model path: %s", model_path)
        except Exception as exc:
            logger.warning(
                "Failed to initialize tokenizer from model path=%s. "
                "prompt_text/response_text fallback will be skipped. error=%s",
                model_path,
                exc,
            )
    else:
        logger.warning("config.model.model_path is empty; skip prompt/response text fallback.")

    experiences = _load_filtered_experiences(
        sqlite_path=sqlite_path,
        table_name=args.sqlite_table,
        model_version=args.model_version,
        logger=logger,
        tokenizer=tokenizer,
    )

    needed_indices, grouped_experiences = _build_needed_index_map(experiences, logger)
    if not grouped_experiences:
        logger.warning(
            "No experiences contain valid task_index after filtering model_version=%s",
            args.model_version,
        )

    taskset_configs = _extract_taskset_configs(config)
    if not taskset_configs:
        raise ValueError("No taskset config found in config.buffer.explorer_input")

    logger.info("Found %d taskset configs", len(taskset_configs))
    task_lookup = _load_taskset_raw_tasks(taskset_configs, needed_indices, logger)

    records = _build_export_records(grouped_experiences, task_lookup, logger)
    output_path = args.output or _default_output_path(args.model_version)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(records, f)

    logger.info("Export completed: %d task records", len(records))
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()
