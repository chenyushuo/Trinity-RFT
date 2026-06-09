import multiprocessing
import os
import shutil
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock

from typer.testing import CliRunner as TyperCliRunner

from tests.tools import (
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli import launcher
from trinity.common.config import (
    AlgorithmConfig,
    BufferConfig,
    StageConfig,
    TrainerInput,
)
from trinity.common.constants import (
    DEBUG_NAMESPACE,
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
)

runner = TyperCliRunner()


class TestLauncherMain(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self._orig_argv = sys.argv.copy()
        self.config = get_template_config()
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.model.model_path = get_model_path()
        self.config.check_and_update()
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)

    def tearDown(self):
        sys.argv = self._orig_argv
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
        import trinity.utils.log as log

        log._ray_logger_ctx.set(None)
        log._ray_logger = None

    @mock.patch("trinity.cli.launcher.serve")
    @mock.patch("trinity.cli.launcher.explore")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.bench")
    @mock.patch("trinity.cli.launcher.load_config")
    def test_main_run_command(
        self, mock_load, mock_bench, mock_both, mock_train, mock_explore, mock_serve
    ):
        config = get_template_config()
        mapping = {
            "explore": mock_explore,
            "train": mock_train,
            "both": mock_both,
            "bench": mock_bench,
            "serve": mock_serve,
        }
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "explore": mock_explore,
                "train": mock_train,
                "both": mock_both,
                "bench": mock_bench,
                "serve": mock_serve,
            },
        ):
            for mode in ["explore", "train", "both", "bench", "serve"]:
                config.mode = mode
                mock_load.return_value = config
                result = runner.invoke(
                    launcher.app,
                    ["run", "--config", "dummy.yaml"],
                )
                self.assertEqual(result.exit_code, 0, msg=result.output)
                mock_load.assert_called_once_with("dummy.yaml")
                mapping[mode].assert_called_once_with(config)
                mock_load.reset_mock()
                mapping[mode].reset_mock()

    @mock.patch("trinity.cli.launcher.stop_ray_cluster")
    @mock.patch("trinity.cli.launcher.setup_ray_cluster")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.load_config")
    @mock.patch("ray.init")
    def test_main_run_in_dlc(self, mock_init, mock_load, mock_both, mock_setup, mock_stop):
        config = get_template_config()
        namespace = f"{config.project}-{config.name}"
        config.mode = "both"
        config.log.level = "WARNING"
        config.log.group_by_node = True
        mock_setup.return_value = "auto"
        mock_load.return_value = config
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "both": mock_both,
            },
        ):
            result = runner.invoke(
                launcher.app,
                ["run", "--config", "dummy.yaml", "--dlc", "--plugin-dir", "/path/to/plugins"],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            mock_init.assert_called_once()
            mock_init.assert_called_once_with(
                address="auto",
                ignore_reinit_error=True,
                namespace=config.ray_namespace,
                runtime_env={
                    "env_vars": {
                        launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                        LOG_DIR_ENV_VAR: config.log.save_dir,
                        LOG_LEVEL_ENV_VAR: config.log.level,
                        LOG_NODE_IP_ENV_VAR: "1",
                    }
                },
            )
            mock_load.assert_called_once_with("dummy.yaml")
            mock_both.assert_called_once_with(config)
            mock_setup.assert_called_once_with(
                namespace=namespace,
            )
            mock_stop.assert_called_once_with(
                namespace=namespace,
            )

    @mock.patch("trinity.manager.config_manager.ConfigManager.run")
    def test_main_studio_command(self, mock_studio_fn):
        result = runner.invoke(
            launcher.app,
            ["studio", "--port", "9999"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_studio_fn.assert_called_once()
        # Typer calls the function with keyword args; verify port was passed
        call_kwargs = mock_studio_fn.call_args
        # The typer-decorated function receives port=9999
        self.assertEqual(
            call_kwargs.kwargs.get("port", call_kwargs.args[0] if call_kwargs.args else None), 9999
        )

    @mock.patch("trinity.cli.launcher.get_latest_hf_checkpoint_path")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.load_config")
    @mock.patch("ray.shutdown")
    @mock.patch("ray.init")
    def test_multi_stage_run(
        self,
        mock_init: MagicMock,
        mock_shutdown: MagicMock,
        mock_load: MagicMock,
        mock_train: MagicMock,
        mock_both: MagicMock,
        mock_checkpoint_path: MagicMock,
    ):
        config = get_template_config()
        config.ray_namespace = ""
        config.checkpoint_root_dir = get_checkpoint_path()
        config.model.model_path = get_model_path()
        config.stages = [
            StageConfig(
                mode="train",
                stage_name="sft_warmup",
                algorithm=AlgorithmConfig(
                    algorithm_type="sft",
                ),
                buffer=BufferConfig(
                    train_batch_size=32,
                    total_steps=100,
                    trainer_input=TrainerInput(
                        experience_buffer=get_unittest_dataset_config("sft_for_gsm8k")
                    ),
                ),
            ),
            StageConfig(
                mode="both",
                stage_name="grpo",
                algorithm=AlgorithmConfig(
                    algorithm_type="grpo",
                ),
            ),
        ]
        mock_load.return_value = config
        mock_checkpoint_path.return_value = "/path/to/hf/checkpoint"
        with mock.patch.dict(
            launcher.MODE_MAP,
            {
                "train": mock_train,
                "both": mock_both,
            },
        ):
            result = runner.invoke(
                launcher.app,
                [
                    "run",
                    "--config",
                    "dummy.yaml",
                    "--plugin-dir",
                    "/path/to/plugins",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertEqual(mock_init.call_count, 2)
            self.assertEqual(mock_shutdown.call_count, 2)
            mock_train.assert_called_once()
            mock_both.assert_called_once()
            expected_calls = [
                mock.call(
                    address="auto",
                    ignore_reinit_error=True,
                    namespace=f"{config.project}/{config.name}/sft_warmup",
                    runtime_env={
                        "env_vars": {
                            launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                            LOG_DIR_ENV_VAR: os.path.join(
                                config.checkpoint_root_dir,
                                config.project,
                                f"{config.name}/sft_warmup",
                                "log",
                            ),
                            LOG_LEVEL_ENV_VAR: config.log.level,
                            LOG_NODE_IP_ENV_VAR: "0",
                        }
                    },
                ),
                mock.call(
                    address="auto",
                    ignore_reinit_error=True,
                    namespace=f"{config.project}/{config.name}/grpo",
                    runtime_env={
                        "env_vars": {
                            launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                            LOG_DIR_ENV_VAR: os.path.join(
                                config.checkpoint_root_dir,
                                config.project,
                                f"{config.name}/grpo",
                                "log",
                            ),
                            LOG_LEVEL_ENV_VAR: config.log.level,
                            LOG_NODE_IP_ENV_VAR: "0",
                        }
                    },
                ),
            ]
            mock_init.assert_has_calls(expected_calls)
            self.assertEqual(mock_checkpoint_path.call_count, 2)
            self.assertEqual(mock_train.call_args[0][0].model.model_path, config.model.model_path)
            self.assertEqual(mock_both.call_args[0][0].model.model_path, "/path/to/hf/checkpoint")
            self.assertEqual(
                mock_both.call_args[0][0].buffer.trainer_input.experience_buffer.tokenizer_path,
                "/path/to/hf/checkpoint",
            )

    @mock.patch("trinity.cli.debug.asyncio.run")
    @mock.patch("trinity.cli.debug.create_debug_models", new_callable=mock.MagicMock)
    @mock.patch("trinity.cli.debug.ray.init")
    @mock.patch("trinity.cli.debug.load_plugins")
    @mock.patch("trinity.cli.debug.load_config")
    def test_debug_inference_model_module(
        self,
        mock_load,
        mock_load_plugins,
        mock_ray_init,
        mock_create_debug_models,
        mock_asyncio_run,
    ):
        self.config.explorer.rollout_model.engine_num = 4
        for index, auxiliary_model in enumerate(self.config.explorer.auxiliary_models, start=2):
            auxiliary_model.engine_num = index
        mock_load.return_value = self.config

        result = runner.invoke(
            launcher.app,
            [
                "debug",
                "--config",
                "dummy.yaml",
                "--module",
                "inference_model",
                "--plugin-dir",
                "/path/to/plugins",
            ],
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_load_plugins.assert_called_once_with()
        mock_load.assert_called_once_with("dummy.yaml")
        self.assertEqual(self.config.mode, "explore")
        self.assertEqual(self.config.ray_namespace, DEBUG_NAMESPACE)
        self.assertEqual(self.config.explorer.rollout_model.engine_num, 1)
        self.assertTrue(
            all(aux_model.engine_num == 1 for aux_model in self.config.explorer.auxiliary_models)
        )
        self.assertEqual(os.environ[launcher.PLUGIN_DIRS_ENV_VAR], "/path/to/plugins")
        mock_ray_init.assert_called_once_with(
            namespace=DEBUG_NAMESPACE,
            runtime_env={"env_vars": self.config.get_envs()},
            ignore_reinit_error=True,
        )
        mock_create_debug_models.assert_called_once_with(self.config)
        mock_asyncio_run.assert_called_once_with(mock_create_debug_models.return_value)

    @mock.patch("trinity.cli.debug.asyncio.run")
    @mock.patch("trinity.cli.debug.ray.init")
    @mock.patch("trinity.cli.debug.load_plugins")
    @mock.patch("trinity.cli.debug.load_config")
    @mock.patch("trinity.explorer.workflow_runner.DebugWorkflowRunner")
    def test_debug_workflow_module(
        self,
        mock_runner_cls,
        mock_load,
        mock_load_plugins,
        mock_ray_init,
        mock_asyncio_run,
    ):
        output_dir = os.path.join(self.config.checkpoint_job_dir, "debug_output")
        mock_load.return_value = self.config
        runner_instance = mock_runner_cls.return_value
        runner_instance.prepare = mock.MagicMock(return_value=object())
        runner_instance.debug = mock.MagicMock(return_value=object())

        result = runner.invoke(
            launcher.app,
            [
                "debug",
                "--config",
                "dummy.yaml",
                "--module",
                "workflow",
                "--enable-profiling",
                "--disable-overwrite",
                "--output-dir",
                output_dir,
            ],
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_load_plugins.assert_called_once_with()
        mock_load.assert_called_once_with("dummy.yaml")
        self.assertEqual(self.config.mode, "explore")
        self.assertEqual(self.config.ray_namespace, DEBUG_NAMESPACE)
        self.assertEqual(self.config.explorer.rollout_model.engine_num, 1)
        mock_ray_init.assert_called_once_with(
            namespace=DEBUG_NAMESPACE,
            runtime_env={"env_vars": self.config.get_envs()},
            ignore_reinit_error=True,
        )
        mock_runner_cls.assert_called_once_with(self.config, output_dir, True, True)
        runner_instance.prepare.assert_called_once_with()
        runner_instance.debug.assert_called_once_with()
        self.assertEqual(mock_asyncio_run.call_count, 2)
        self.assertEqual(
            mock_asyncio_run.call_args_list,
            [
                mock.call(runner_instance.prepare.return_value),
                mock.call(runner_instance.debug.return_value),
            ],
        )

    @mock.patch("trinity.manager.log_manager.LogManager")
    @mock.patch("trinity.cli.log.load_config")
    def test_log_mode(self, mock_load_config, mock_log_manager):
        result = runner.invoke(launcher.app, ["log"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Either --config or --log-dir must be provided", result.output)

        mock_cfg = mock.Mock()
        mock_cfg.get_checkpoint_job_dir.return_value = "/tmp/job"
        mock_load_config.return_value = mock_cfg
        with mock.patch("os.path.exists", return_value=True):
            result = runner.invoke(
                launcher.app,
                [
                    "log",
                    "--config",
                    "dummy.yaml",
                    "-k",
                    "trainer",
                    "-l",
                    "DEBUG",
                    "-n",
                    str(5),
                    "-p",
                    "ERROR",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            mock_log_manager.assert_called_once_with(
                log_dir="/tmp/job/log",
                keyword="trainer",
                min_level="DEBUG",
                color_output=True,
                last_n_lines=5,
                search_pattern="ERROR",
            )
            mock_log_manager.return_value.monitor.assert_called_once()

        with mock.patch("os.path.exists", return_value=True):
            result = runner.invoke(launcher.app, ["log", "--log-dir", "/tmp/job/log"])
            self.assertEqual(result.exit_code, 0)
            mock_log_manager.assert_called_with(
                log_dir="/tmp/job/log",
                keyword=None,
                min_level="INFO",
                color_output=True,
                last_n_lines=0,
                search_pattern=None,
            )

        with mock.patch("os.path.exists", return_value=False):
            result = runner.invoke(launcher.app, ["log", "--config", "dummy.yaml"])
            print("result.exc_info:", result.exc_info)
            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(result.exc_info[0], FileNotFoundError)
