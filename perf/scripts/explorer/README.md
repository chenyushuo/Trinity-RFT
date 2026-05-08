# Explorer 性能评测工具

## 目标

该工具用于评测 Trinity-RFT 中 Explorer 模块的运行性能，不关注模型训练效果，也不复用 Trinity 主流程中的 benchmark 或 eval 语义。

第一版的设计目标如下：

1. 不修改 Trinity 主流程代码，只在 perf 目录下独立实现。
2. 资源采集能力可独立复用，后续可直接供 trainer perf 使用。
3. 吞吐量和任务平均完成时间优先复用 Explorer 已有 step metrics。
4. 同时提供全局汇总指标和 step 级指标。
5. 单次运行完成后直接输出汇总结果，不考虑 warmup 和多轮重复实验。

## 统计范围

该工具计划统计以下性能指标：

1. 初始化时间：从开始启动到初始化完成的时间。
2. 吞吐量：单位时间内完成的任务数量，单位为 task / min。
3. 每个任务的平均完成时间：单位为 sec / task。
4. 资源使用情况：CPU、GPU 利用率、内存以及 GPU 显存使用情况，按固定间隔采样，并保留时序序列。
5. step 级性能：每个 step 的任务完成数、吞吐量、平均任务耗时以及对应原始 rollout 指标。

## 边界约束

该工具的边界约束如下：

1. 不修改 [trinity/explorer/explorer.py](/root/Trinity-RFT/trinity/explorer/explorer.py) 或其他主流程模块。
2. 不把资源采集逻辑塞进现有 monitor 框架。
3. 不将训练效果评估类的 benchmark 或 eval 指标混入性能结果。
4. 不做 warmup、对照实验或多次重复取均值。
5. 第一版默认运行环境具备可用 GPU，不额外兼容无 GPU 场景。
6. 第一版优先支持单次本地运行和结果落盘。

## 设计概览

整体思路是把 Explorer perf 拆成两条完全独立的链路：

1. 运行链路：启动 Explorer，统计启动耗时和总运行时间。
2. 观测链路：在外部独立采集系统资源，并从 TensorBoard 文件读取 step metrics。

其中：

1. 资源数据来自 perf 下的独立采样模块。
2. 吞吐量和任务平均完成时间来自 Explorer monitor 产出的 step metrics。
3. 汇总逻辑在 perf 脚本内完成，不侵入 Trinity 现有实现。

## 推荐目录结构

建议按如下方式组织代码：

```text
perf/
  scripts/
    explorer/
      README.md
      example.yaml
trinity/
  perf/
    __init__.py
    stage_perf.py
    tensorboard_metrics.py
    report_utils.py
    resource_backends.py
    resource_sampler.py
```

各文件职责建议如下：

1. `trinity/perf/resource_backends.py`
   封装资源采集后端，例如 `psutil` 和 `pynvml`。
2. `trinity/perf/resource_sampler.py`
   提供独立资源采样器，支持启动、停止、导出原始样本和聚合统计。
3. `trinity/perf/report_utils.py`
   提供时间序列聚合、百分位数计算和统一 JSON 序列化能力。
4. `trinity/perf/stage_perf.py`
  负责 Explorer perf 的单次运行编排和结果落盘。
5. `perf/scripts/explorer/example.yaml`
   提供最小可运行的 Trinity Explorer 配置样例。

这种拆分方式的核心目的是让资源采集模块天然进入 `trinity` 命名空间，后续 trainer perf 可以直接复用 `trinity.perf.*`。

## 运行流程草案

Explorer perf 脚本建议按以下阶段执行：

1. 读取 Trinity 配置文件并校验 `mode: explore`。
2. 校验 `monitor.monitor_type == tensorboard`。
3. 初始化资源采样器并启动后台采样。
4. 创建 Explorer actor。
5. 单独计时 `prepare.remote()`，得到启动耗时。
6. 执行 `sync_weight.remote()`。
7. 执行 `explore.remote()` 直到运行结束。
8. 停止资源采样。
9. 解析 TensorBoard 本地文件，提取 step 级 metrics。
10. 聚合资源指标和 Explorer step 指标。
11. 输出 JSON 结果到指定路径。

这里的“启动耗时”定义为：

1. 从 perf 脚本开始创建 Explorer actor。
2. 到 `prepare.remote()` 成功返回为止。

这样可以覆盖模型准备、rollout coordinator 准备等初始化成本，同时保持对主流程零侵入。

## 指标来源草案

### 资源指标

资源指标由 perf 目录下的独立采样模块提供，建议采样字段如下：

1. `timestamp`
2. `cpu_percent`
3. `memory_rss_mb`
4. `memory_percent`
5. `gpu_metrics`

其中 `gpu_metrics` 建议按卡记录，例如：

1. `gpu_id`
2. `gpu_util_percent`
3. `gpu_memory_used_mb`
4. `gpu_memory_total_mb`

第一版建议优先支持整机级采样，不强制做按 Ray actor 或 PID 树聚合。

资源序列的展示目标如下：

1. CPU 只保留一条时间线。
2. GPU 为每张卡分别保留一条时间线。
3. 结果格式优先为后续折线图绘制服务，而不是做离线聚合统计。

### Explorer 运行指标

Explorer 运行指标优先从 TensorBoard 事件文件提取，原因如下：

1. 不需要修改 Explorer 主流程。
2. Explorer 已有 monitor 写本地标量文件。
3. step 级指标能够自然复用，不需要重新推导内部状态。

因此，第一版建议明确要求 monitor 使用 `tensorboard`。

## 吞吐量与平均任务耗时口径

建议统一采用以下统计口径：

1. step 吞吐量：`finished_task_count / step_time_sec * 60`
2. step 平均任务耗时：`step_time_sec / finished_task_count`
3. 全局吞吐量：`sum(finished_task_count) / sum(step_time_sec) * 60`
4. 全局平均任务耗时：`sum(step_time_sec) / sum(finished_task_count)`

实现时优先直接读取 TensorBoard 中已有的 step 级时间类指标。如果不同配置场景下字段名存在差异，建议在 perf 代码中维护字段映射表，而不要把具体字段名散落在业务逻辑中。

## 输出结果结构草案

结果文件建议输出为一个 JSON 文档，结构如下：

```json
{
  "run_meta": {},
  "timing": {},
  "resource_timeline": [],
  "step_metrics": [],
  "global_metrics": {},
  "artifacts": {},
  "status": {}
}
```

各字段建议含义如下：

### `run_meta`

记录一次 perf 运行的基础信息：

1. config 路径
2. explorer 名称
3. 采样间隔
4. 启动时间
5. hostname
6. pid

### `timing`

记录关键耗时：

1. `startup_time_sec`
2. `execution_time_sec`
3. `total_time_sec`

### `resource_timeline`

记录原始采样序列，用于后续可视化或 trainer perf 复用。

建议至少包含以下结构：

1. `timestamp`
2. `cpu_percent`
3. `memory_rss_mb`
4. `gpu_metrics`

其中 `gpu_metrics` 为数组，每个元素对应一张卡，例如：

1. `gpu_id`
2. `gpu_util_percent`
3. `gpu_memory_used_mb`

结果组织应优先满足以下可视化需求：

1. CPU 一条折线。
2. GPU utilization 按卡多条折线。
3. GPU memory used 按卡多条折线。

### `step_metrics`

每个 step 一条记录，建议包含：

1. `step`
2. `finished_task_count`
3. `throughput_task_per_min`
4. `avg_task_time_sec`
5. `raw_metrics`

### `global_metrics`

记录全局性能指标，建议至少包含：

1. `total_finished_task_count`
2. `overall_throughput_task_per_min`
3. `overall_avg_task_time_sec`

### `artifacts`

记录排障和追踪所需路径：

1. `checkpoint_job_dir`
2. `tensorboard_dir`
3. `log_dir`
4. `output_json`

### `status`

记录运行状态：

1. 是否成功完成。
2. 异常信息。
3. 是否拿到 GPU 指标。

## 命令行接口草案

当前建议的命令行形式如下：

## 使用方法

```
python -m trinity.cli.launcher perf --module explorer --config <path_to_config> --output-path <path_to_output> [--monitor-interval <interval_in_seconds>]
```

建议支持以下参数：

1. `--config`
   Trinity 配置文件路径，要求符合 Trinity 配置规范，且模式为 `explore`。
2. `--output-path`
   结果 JSON 输出路径。
3. `--monitor-interval`
   资源采样间隔，默认 5 秒。
4. `--timeout`
   整次 perf 运行的超时时间，可选。
5. `--total-steps`
  覆盖配置中的 Explorer 总步数，默认 5。
6. `--module`
  当前固定为 `explorer`，为后续扩展 trainer perf 预留统一入口。

## 配置要求

该工具依赖以下配置约束：

1. `mode` 必须为 `explore`。
2. `monitor.monitor_type` 必须为 `tensorboard`。
3. Explorer 本身应能在当前环境下正常启动和运行。

如果 monitor 不是 `tensorboard`，建议 perf 工具直接报错退出，而不是在运行时偷偷覆盖用户配置。

## 示例结果草案

下面给出一个结果结构示意：

```json
{
  "run_meta": {
    "config_path": "perf/scripts/explorer/example.yaml",
    "monitor_interval_sec": 5
  },
  "timing": {
    "startup_time_sec": 32.5,
    "execution_time_sec": 640.2,
    "total_time_sec": 672.7
  },
  "resource_timeline": [
    {
      "timestamp": 1710000000.0,
      "cpu_percent": 71.2,
      "memory_rss_mb": 18342.0,
      "gpu_metrics": [
        {
          "gpu_id": 0,
          "gpu_util_percent": 84.0,
          "gpu_memory_used_mb": 22134.0
        },
        {
          "gpu_id": 1,
          "gpu_util_percent": 79.0,
          "gpu_memory_used_mb": 21980.0
        }
      ]
    }
  ],
  "step_metrics": [
    {
      "step": 1,
      "finished_task_count": 64,
      "throughput_task_per_min": 384.0,
      "avg_task_time_sec": 0.156,
      "raw_metrics": {}
    }
  ],
  "global_metrics": {
    "total_finished_task_count": 1024,
    "overall_throughput_task_per_min": 401.7,
    "overall_avg_task_time_sec": 0.149
  }
}
```

## 实现清单

下面是建议的实现顺序：

1. 定义结果 JSON schema 和字段口径。
2. 实现 `trinity/perf/resource_backends.py`。
3. 实现 `trinity/perf/resource_sampler.py`。
4. 实现 `trinity/perf/report_utils.py`。
5. 在 `trinity/perf/stage_perf.py` 中完成单次运行编排。
6. 在 `trinity/perf/stage_perf.py` 中实现 TensorBoard 指标解析。
7. 补充 `perf/scripts/explorer/example.yaml`。
8. 补充测试和文档示例。

## 测试建议

第一版建议优先补以下测试：

1. 资源采样器可以稳定输出 CPU 单线时序和按卡 GPU 时序。
3. TensorBoard 解析逻辑可以正确提取 step 级 metrics。
4. 全局吞吐量和平均任务耗时计算口径正确。
5. Explorer perf 运行失败时仍能输出可诊断的状态字段。

## 已知取舍

第一版的取舍如下：

1. 优先做整机级资源观测，不强制追踪 Ray actor 子进程。
2. 强依赖 `tensorboard` monitor，不额外兼容 wandb 或 mlflow。
3. 默认要求运行环境具备 GPU，不额外兼容无 GPU 场景。
4. 资源结果优先保留时序序列，不在第一版中输出 `mean/max/min/p50/p95` 聚合统计。
5. 只做单次运行和汇总，不做 warmup 和多次重复实验。
6. 资源采样和 TensorBoard 指标解析均在 perf 层完成，不入侵主流程。

## 后续扩展方向

该设计后续可以自然扩展到：

1. trainer perf 复用资源采样模块。
2. 自动生成 Markdown 报告。
3. 增加 PID 级或进程树级资源观测。
4. 支持更多 monitor 后端的指标解析。
