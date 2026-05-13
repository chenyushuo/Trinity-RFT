"""otel_init.py — OpenTelemetry 初始化与工具函数。

只在 --enable_otel 开启时调用 init_otel()，其他情况下 trace_span() 为 no-op，
对业务代码完全透明。

端点配置来自 setup_otel.sh（ARMS 阿里云链路追踪）。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------
_DEFAULT_SERVICE_NAME = "qwenpaw-run"

# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------
_otel_enabled: bool = False
_tracer = None  # opentelemetry.trace.Tracer | None
_attributes = {}  # dict，记录全局属性（如 task_id）


def init_otel(
    service_name: str | None = None,
    endpoint: str | None = None,
    license_key: str | None = None,
    attributes: dict | None = None,
) -> bool:
    """初始化 OpenTelemetry SDK，配置 OTLP HTTP exporter。

    参数均可通过环境变量覆盖：
      OTEL_SERVICE_NAME      — 服务名称
      OTEL_EXPORTER_OTLP_ENDPOINT — 上报端点
      OTEL_ARMS_LICENSE_KEY  — ARMS license key（作为 HTTP header 下发）

    Returns:
        True if initialization succeeded, False otherwise.
    """
    global _tracer, _otel_enabled

    svc = service_name or os.environ.get("OTEL_SERVICE_NAME", _DEFAULT_SERVICE_NAME)
    ep = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    key = license_key or os.environ.get("OTEL_ARMS_LICENSE_KEY")
    arms_project = os.environ.get("OTEL_ARMS_PROJECT")
    cms_workspace = os.environ.get("OTEL_CMS_WORKSPACE")

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import (
            DEPLOYMENT_ENVIRONMENT,
            HOST_NAME,
            SERVICE_NAME,
            SERVICE_VERSION,
            Resource,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(
            attributes={
                SERVICE_NAME: svc,
                SERVICE_VERSION: "v0",
                DEPLOYMENT_ENVIRONMENT: "test",
                HOST_NAME: "sandbox",
                "acs.cms.workspace": cms_workspace,
            }
        )

        headers = {
            "x-arms-license-key": key,
            "x-arms-project": arms_project,
            "x-cms-workspace": cms_workspace,
        }
        # 使用HTTP协议上报
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{ep}/v1/traces", headers=headers)
        )

        trace_provider = TracerProvider(resource=resource, active_span_processor=span_processor)
        trace.set_tracer_provider(trace_provider)

        _otel_enabled = True
        if attributes:
            _attributes.update(attributes)
        log.info("OpenTelemetry 初始化成功，service=%s, endpoint=%s", svc, ep)
        return True

    except ImportError as e:
        log.warning("OpenTelemetry 包未安装，跳过 otel 初始化: %s", e)
        return False
    except Exception as e:
        log.warning("OpenTelemetry 初始化失败: %s", e)
        return False


def is_enabled() -> bool:
    """返回当前 OTel 是否已成功初始化。"""
    return _otel_enabled


@contextmanager
def trace_span(
    name: str,
    attributes: dict | None = None,
) -> Generator[None, None, None]:
    """上下文管理器：otel 启用时创建 span 并附加属性，否则为 no-op。

    异常会被自动记录到 span 并重新抛出。

    Usage::

        with trace_span("call_agent", {"task_id": task_id}):
            call_agent(...)
    """
    if not _otel_enabled:
        yield
        return

    try:
        from opentelemetry import trace

        _tracer = trace.get_tracer(__name__)
        merged_attributes = _attributes.copy()
        if attributes:
            merged_attributes.update(attributes)
        attributes = merged_attributes
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            try:
                yield
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise
    except ImportError:
        yield
