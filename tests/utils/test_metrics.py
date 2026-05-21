"""Tests for trinity.utils.metrics aggregation utilities."""

import pytest

from trinity.utils.metrics import (
    AggType,
    aggregate_eval_metrics,
    aggregate_metrics,
    aggregate_run_level_metrics,
    calculate_task_level_metrics,
    parse_metric_key,
)


class TestMetricsUtils:
    def test_parse_key(self):
        name, agg = parse_metric_key("reward")
        # no suffix
        assert name == "reward"
        assert agg == AggType.MEAN
        # sum
        name, agg = parse_metric_key("experience_count:sum")
        assert name == "experience_count"
        assert agg == AggType.SUM
        # max
        name, agg = parse_metric_key("latency:max")
        assert name == "latency"
        assert agg == AggType.MAX
        # min
        name, agg = parse_metric_key("latency:min")
        assert name == "latency"
        assert agg == AggType.MIN
        # last
        name, agg = parse_metric_key("model_version:last")
        assert name == "model_version"
        assert agg == AggType.LAST
        # mean
        name, agg = parse_metric_key("reward:mean")
        assert name == "reward"
        assert agg == AggType.MEAN
        # unknown suffix defaults to mean
        name, agg = parse_metric_key("some:unknown_suffix")
        assert name == "some:unknown_suffix"
        assert agg == AggType.MEAN
        # with slash
        name, agg = parse_metric_key("time/task_execution:sum")
        assert name == "time/task_execution"
        assert agg == AggType.SUM
        # multiple colons, only last one is suffix
        name, agg = parse_metric_key("ns:metric_name:sum")
        assert name == "ns:metric_name"
        assert agg == AggType.SUM

    def test_aggregate(self):
        # empty
        assert aggregate_metrics([]) == {}
        # single dict with no suffix
        result = aggregate_metrics([{"reward": 1.0}], prefix="rollout")
        assert result == {
            "rollout/reward/mean": 1.0,
            "rollout/reward/max": 1.0,
            "rollout/reward/min": 1.0,
        }
        # multiple dicts with no suffix should compute mean, max, min
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result["rollout/reward/mean"] == 2.0
        assert result["rollout/reward/max"] == 3.0
        assert result["rollout/reward/min"] == 1.0

        # keys with :sum suffix should be summed
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/tokens/sum": 300.0}

        # keys with :max suffix should take max
        dicts = [{"latency:max": 5.0}, {"latency:max": 10.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/latency/max": 10.0}

        # keys with :min suffix should take min
        dicts = [{"latency:min": 5.0}, {"latency:min": 2.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/latency/min": 2.0}

        # keys with :last suffix should take last value
        dicts = [{"model_version:last": 1.0}, {"model_version:last": 3.0}]
        result = aggregate_metrics(dicts, prefix="rollout")
        assert result == {"rollout/model_version/last": 3.0}

        # mixed keys
        dicts = [
            {"reward": 1.0, "tokens:sum": 100.0, "version:last": 1.0},
            {"reward": 3.0, "tokens:sum": 200.0, "version:last": 2.0},
        ]
        result = aggregate_metrics(dicts, prefix="r")
        assert result["r/reward/mean"] == 2.0
        assert result["r/tokens/sum"] == 300.0
        assert result["r/version/last"] == 2.0

        # no prefix
        dicts = [{"reward": 2.0}]
        result = aggregate_metrics(dicts, prefix="")
        assert result == {"reward/mean": 2.0, "reward/max": 2.0, "reward/min": 2.0}

        # custom default_output_stats
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_metrics(dicts, prefix="p", default_output_stats=["mean", "std"])
        assert "p/reward/mean" in result
        assert "p/reward/std" in result
        assert "p/reward/max" not in result

        # ignore non-numeric keys
        dicts = [{"reward": 1.0, "name": "hello"}]
        result = aggregate_metrics(dicts, prefix="p")
        assert "p/name/mean" not in result
        assert "p/reward/mean" in result

    def test_aggregate_eval_metrics(self):
        # empty
        assert aggregate_eval_metrics([]) == {}
        # simple stats
        dicts = [{"accuracy": 0.8}, {"accuracy": 0.9}]
        result = aggregate_eval_metrics(dicts, prefix="eval/gsm8k", detailed_stats=False)
        assert result == {"eval/gsm8k/accuracy": pytest.approx(0.85)}
        # detail stats
        dicts = [{"accuracy": 0.8}, {"accuracy": 0.9}]
        result = aggregate_eval_metrics(dicts, prefix="eval/gsm8k", detailed_stats=True)
        assert "eval/gsm8k/accuracy/mean" in result
        assert "eval/gsm8k/accuracy/std" in result
        assert result["eval/gsm8k/accuracy/mean"] == pytest.approx(0.85)
        # sum
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_eval_metrics(dicts, prefix="eval/test", detailed_stats=False)
        assert result == {"eval/test/tokens/sum": 300.0}
        # last
        dicts = [{"version:last": 1.0}, {"version:last": 5.0}]
        result = aggregate_eval_metrics(dicts, prefix="eval/test", detailed_stats=False)
        assert result == {"eval/test/version/last": 5.0}

    def test_run_level_metrics(self):
        # empty
        assert aggregate_run_level_metrics([]) == {}

        # no suffix should compute mean
        dicts = [{"reward": 1.0}, {"reward": 3.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"reward": 2.0}
        # sum
        dicts = [{"tokens:sum": 100.0}, {"tokens:sum": 200.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"tokens:sum": 300.0}
        # max
        dicts = [{"latency:max": 5.0}, {"latency:max": 10.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"latency:max": 10.0}
        # last
        dicts = [{"version:last": 1.0}, {"version:last": 2.0}]
        result = aggregate_run_level_metrics(dicts)
        assert result == {"version:last": 2.0}
        # suffix should be preserved in output keys
        dicts = [{"reward": 1.0, "total_steps:sum": 5.0}]
        result = aggregate_run_level_metrics(dicts)
        assert "reward" in result
        assert "total_steps:sum" in result
        # mix of keys
        dicts = [
            {"reward": 1.0, "steps:sum": 3.0, "peak:max": 10.0},
            {"reward": 3.0, "steps:sum": 4.0, "peak:max": 8.0},
        ]
        result = aggregate_run_level_metrics(dicts)
        assert result["reward"] == 2.0
        assert result["steps:sum"] == 7.0
        assert result["peak:max"] == 10.0

    def test_backward_compatibility(self):
        # workflow metrics
        exp_metrics = [
            {"reward": 0.5, "actual_env_steps": 3.0, "time/run_execution": 1.2},
            {"reward": 0.8, "actual_env_steps": 5.0, "time/run_execution": 1.5},
        ]
        # Run level aggregation (experience → run)
        run_result = aggregate_run_level_metrics(exp_metrics)
        assert run_result["reward"] == pytest.approx(0.65)
        assert run_result["actual_env_steps"] == pytest.approx(4.0)
        assert run_result["time/run_execution"] == pytest.approx(1.35)

        # batch level metrics
        task_metrics = [
            {"reward": 0.5, "time/run_execution": 1.2},
            {"reward": 0.8, "time/run_execution": 1.5},
        ]
        result = aggregate_metrics(task_metrics, prefix="rollout")
        assert "rollout/reward/mean" in result
        assert "rollout/reward/max" in result
        assert "rollout/reward/min" in result
        assert result["rollout/reward/mean"] == pytest.approx(0.65)

        # eval metrics
        task_metrics = [{"accuracy": 0.9}, {"accuracy": 0.7}]
        result = aggregate_eval_metrics(task_metrics, prefix="eval/gsm8k", detailed_stats=False)
        assert result == {"eval/gsm8k/accuracy": pytest.approx(0.8)}

    def test_calculate_task_level_metrics(self):
        # train level
        metrics = [
            {"reward": 1.0, "tokens:sum": 3.0, "version:last": 1.0},
            {"reward": 3.0, "tokens:sum": 4.0, "version:last": 2.0},
        ]

        result = calculate_task_level_metrics(metrics, is_eval=False)

        assert result == {
            "reward": 2.0,
            "tokens:sum": 7.0,
            "version:last": 2.0,
        }
        # eval level
        metrics = [
            {"accuracy": 0.2, "tokens:sum": 3.0, "version:last": 1.0, "time/run_execution": 1.0},
            {"accuracy": 0.8, "tokens:sum": 4.0, "version:last": 2.0, "time/run_execution": 3.0},
        ]

        result = calculate_task_level_metrics(metrics, is_eval=True)

        assert result["accuracy/mean@2"] == pytest.approx(0.5)
        assert result["accuracy/std@2"] == pytest.approx(0.3)
        assert "accuracy/best@2" in result
        assert "accuracy/worst@2" in result
        assert result["tokens:sum"] == 7.0
        assert result["version:last"] == 2.0
        assert result["time/run_execution"] == pytest.approx(2.0)
