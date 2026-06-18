"""Tests for the ``trinity view`` command and its config-resolution helpers.

Covers the behaviours added/changed for the viewer CLI:
- ``--url`` accepting a plain ``.db`` file path (relative/absolute) and a DB URL.
- ``--config`` resolving the database URL, table name and tokenizer from a config.
- the experience-pipeline default ``input_save_path`` living in the pipeline
  (``default_input_save_path``) rather than a config validator.
- the rename edge-case warning.
"""

import os
import tempfile
import unittest
from unittest import mock

import typer
from typer.testing import CliRunner

from trinity.cli.view import (
    _resolve_db_url,
    _resolve_from_config,
    _warn_if_renamed,
    view_command,
)
from trinity.common.config import Config

runner = CliRunner()


def _build_config(**pipeline_overrides) -> Config:
    """Build a minimal Config suitable for view resolution (no validation)."""
    config = Config()
    config.checkpoint_root_dir = tempfile.mkdtemp()
    config.project = "Proj"
    config.group = "Grp"
    config.name = "run1"
    config.model.model_path = "/path/to/model"
    pipeline = config.data_processor.experience_pipeline
    pipeline.save_input = pipeline_overrides.pop("save_input", True)
    pipeline.input_save_path = pipeline_overrides.pop("input_save_path", None)
    return config


class TestResolveDbUrl(unittest.TestCase):
    def test_plain_relative_path_converted_to_sqlite_url(self):
        # A bare path is resolved against CWD and prefixed with sqlite:///.
        resolved = _resolve_db_url("debug_buffer.db")
        self.assertTrue(resolved.startswith("sqlite:///"))
        self.assertTrue(resolved.endswith("debug_buffer.db"))
        self.assertTrue(os.path.isabs(resolved[len("sqlite:///") :]))

    def test_plain_absolute_path_converted(self):
        abs_path = os.path.abspath("debug_buffer.db")
        resolved = _resolve_db_url(abs_path)
        # Absolute paths yield the four-slash sqlite URL form: "sqlite:///" + "/abs/...".
        self.assertEqual(resolved, "sqlite:///" + abs_path)

    def test_existing_sqlite_url_passthrough(self):
        url = "sqlite:////tmp/abs/debug_buffer.db"
        self.assertEqual(_resolve_db_url(url), url)

    def test_postgres_url_passthrough(self):
        url = "postgresql://user:pass@host:5432/db"
        self.assertEqual(_resolve_db_url(url), url)

    def test_mysql_url_passthrough(self):
        url = "mysql://user:pass@host/db"
        self.assertEqual(_resolve_db_url(url), url)


class TestDefaultInputSavePath(unittest.TestCase):
    def test_produces_sqlite_url_under_buffer_cache(self):
        from trinity.buffer.pipelines.experience_pipeline import default_input_save_path

        config = _build_config()
        path = default_input_save_path(config)
        self.assertTrue(path.startswith("sqlite:///"))
        self.assertTrue(path.endswith("explorer_output.db"))
        self.assertIn("buffer", path)
        self.assertIn(os.path.join("Proj", "Grp", "run1"), path)

    def test_abs_path_matches_validator_derivation(self):
        # The pipeline-owned default must equal what the validator used to set:
        # sqlite:///<checkpoint_job_dir>/buffer/explorer_output.db (abs-ified).
        from trinity.buffer.pipelines.experience_pipeline import default_input_save_path

        config = _build_config()
        expected_cache = os.path.abspath(os.path.join(config.get_checkpoint_job_dir(), "buffer"))
        expected = "sqlite:///" + os.path.join(expected_cache, "explorer_output.db")
        self.assertEqual(default_input_save_path(config), expected)


class TestResolveFromConfig(unittest.TestCase):
    def _patch_load(self, config: Config):
        return mock.patch("trinity.common.config.load_config", return_value=config)

    def test_save_input_false_is_rejected(self):
        config = _build_config(save_input=False, input_save_path="sqlite:///x.db")
        with self._patch_load(config):
            with self.assertRaises(typer.BadParameter):
                _resolve_from_config("dummy.yaml")

    def test_json_save_path_is_rejected(self):
        config = _build_config(input_save_path="/tmp/foo.jsonl")
        with self._patch_load(config):
            with self.assertRaises(typer.BadParameter):
                _resolve_from_config("dummy.yaml")

    def test_none_input_save_path_uses_pipeline_default(self):
        config = _build_config(input_save_path=None)
        from trinity.buffer.pipelines.experience_pipeline import default_input_save_path

        with self._patch_load(config):
            db_url, table, tokenizer = _resolve_from_config("dummy.yaml")
        self.assertEqual(db_url, default_input_save_path(config))
        self.assertEqual(table, "pipeline_input")
        self.assertEqual(tokenizer, config.model.model_path)

    def test_explicit_db_url_is_used(self):
        config = _build_config(input_save_path="sqlite:////explicit/buffer.db")
        with self._patch_load(config):
            db_url, table, tokenizer = _resolve_from_config("dummy.yaml")
        self.assertEqual(db_url, "sqlite:////explicit/buffer.db")
        self.assertEqual(table, "pipeline_input")
        self.assertEqual(tokenizer, config.model.model_path)

    def test_missing_model_path_is_rejected(self):
        config = _build_config(input_save_path="sqlite:///x.db")
        config.model.model_path = ""
        with self._patch_load(config):
            with self.assertRaises(typer.BadParameter):
                _resolve_from_config("dummy.yaml")


class TestWarnIfRenamed(unittest.TestCase):
    def _capture(self):
        calls = []
        patcher = mock.patch("typer.secho", side_effect=lambda *a, **k: calls.append(a[0]))
        return patcher, calls

    def test_silent_when_continue_from_checkpoint(self):
        config = _build_config()
        config.continue_from_checkpoint = True
        os.makedirs(os.path.join(config.get_checkpoint_job_dir(), "buffer"), exist_ok=True)
        patcher, calls = self._capture()
        with patcher:
            _warn_if_renamed(config)
        self.assertEqual(calls, [])

    def test_silent_when_job_dir_missing(self):
        config = _build_config()
        config.continue_from_checkpoint = False
        # job dir does not exist on disk -> no rename happened
        patcher, calls = self._capture()
        with patcher:
            _warn_if_renamed(config)
        self.assertEqual(calls, [])

    def test_warns_and_lists_candidates_when_renamed(self):
        config = _build_config()
        config.continue_from_checkpoint = False
        job_dir = config.get_checkpoint_job_dir()
        # The clean dir exists and is non-empty (rename condition).
        os.makedirs(os.path.join(job_dir, "buffer"), exist_ok=True)
        # A timestamped sibling that actually holds the data.
        sibling = f"{job_dir}_20260101000000"
        os.makedirs(os.path.join(sibling, "buffer"), exist_ok=True)
        try:
            patcher, calls = self._capture()
            with patcher:
                _warn_if_renamed(config)
            self.assertEqual(len(calls), 1)
            self.assertIn("auto-renamed", calls[0])
            self.assertIn(os.path.basename(sibling), calls[0])
        finally:
            import shutil

            shutil.rmtree(sibling, ignore_errors=True)


class TestViewCommandCli(unittest.TestCase):
    """Drive the typer-decorated view_command via a throwaway app."""

    def _app(self):
        app = typer.Typer()
        app.command()(view_command)
        return app

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_url_plain_path(self, mock_run):
        result = runner.invoke(
            self._app(), ["--url", "debug_buffer.db", "--table", "t", "--tokenizer", "/m"]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        _, kwargs = mock_run.call_args
        self.assertTrue(kwargs["db_url"].startswith("sqlite:///"))
        self.assertTrue(kwargs["db_url"].endswith("debug_buffer.db"))
        self.assertEqual(kwargs["table_name"], "t")
        self.assertEqual(kwargs["model_path"], "/m")

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_url_existing_url_passthrough(self, mock_run):
        result = runner.invoke(
            self._app(),
            ["--url", "sqlite:////abs/debug.db", "--table", "t", "--tokenizer", "/m"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(mock_run.call_args.kwargs["db_url"], "sqlite:////abs/debug.db")

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_config_resolves_values(self, mock_run):
        config = _build_config(input_save_path="sqlite:////cfg/buffer.db")
        with mock.patch("trinity.common.config.load_config", return_value=config):
            result = runner.invoke(self._app(), ["--config", "dummy.yaml"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["db_url"], "sqlite:////cfg/buffer.db")
        self.assertEqual(kwargs["table_name"], "pipeline_input")
        self.assertEqual(kwargs["model_path"], config.model.model_path)

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_explicit_cli_overrides_config(self, mock_run):
        config = _build_config(input_save_path="sqlite:////cfg/buffer.db")
        with mock.patch("trinity.common.config.load_config", return_value=config):
            result = runner.invoke(
                self._app(),
                [
                    "--config",
                    "dummy.yaml",
                    "--url",
                    "override.db",
                    "--table",
                    "override_table",
                    "--tokenizer",
                    "/override_model",
                ],
            )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        _, kwargs = mock_run.call_args
        self.assertTrue(kwargs["db_url"].endswith("override.db"))
        self.assertEqual(kwargs["table_name"], "override_table")
        self.assertEqual(kwargs["model_path"], "/override_model")

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_config_rejects_save_input_false(self, mock_run):
        config = _build_config(save_input=False, input_save_path="sqlite:///x.db")
        with mock.patch("trinity.common.config.load_config", return_value=config):
            result = runner.invoke(self._app(), ["--config", "dummy.yaml"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("save_input", result.output)

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_missing_url_and_config_errors(self, mock_run):
        result = runner.invoke(self._app(), ["--table", "t", "--tokenizer", "/m"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--url", result.output)

    @mock.patch("trinity.buffer.viewer.SQLExperienceViewer.run_viewer")
    def test_missing_table_without_config_errors(self, mock_run):
        result = runner.invoke(self._app(), ["--url", "debug.db", "--tokenizer", "/m"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--table", result.output)


if __name__ == "__main__":
    unittest.main()
