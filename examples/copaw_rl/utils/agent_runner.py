#!/usr/bin/env python3

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import requests
from setup_provider import (
    config_builtin_provider,
    config_provider,
    put_tool_guard_settings,
)

RL_PROVIDER_NAME = "rl-server"
_STOP_ENDPOINTS = (
    "/api/console/chat/stop",
    "/api/agent/console/chat/stop",
    "/api/agents/default/console/chat/stop",
)
_STOP_WAIT_SECONDS = 15.0


@dataclass
class _StreamLogState:
    step_idx: int = 0
    step_has_output: bool = False
    tool_event_count: int = 0


@dataclass
class CallAgentResult:
    timed_out: bool = False
    stop_succeeded: bool = False


def _candidate_chat_list_endpoints(url: str) -> tuple[str, ...]:
    return (
        f"{url}/api/chats",
        f"{url}/api/agents/default/chats",
    )


def _resolve_chat_id(url: str, session_id: str, user_id: str, log: logging.Logger) -> str | None:
    for chats_url in _candidate_chat_list_endpoints(url):
        try:
            response = requests.get(chats_url, timeout=10)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            chats = response.json()
        except (requests.RequestException, ValueError) as exc:
            log.debug("failed to list chats from %s: %s", chats_url, exc)
            continue

        if not isinstance(chats, list):
            continue

        for chat in chats:
            if not isinstance(chat, dict):
                continue
            if chat.get("session_id") != session_id:
                continue
            if user_id and chat.get("user_id") not in {None, "", user_id}:
                continue

            chat_id = chat.get("id")
            if isinstance(chat_id, str) and chat_id:
                log.info("resolved session=%s to chat_id=%s via %s", session_id, chat_id, chats_url)
                return chat_id
    return None


def _stop_targets(session_id: str, chat_id: str | None) -> list[str]:
    targets = [session_id]
    if chat_id and chat_id not in targets:
        targets.append(chat_id)
    return targets


def _shorten_for_log(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
        text = text.strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _append_event(
    events: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    kind: str,
    text: Any = "",
    name: Any = "",
    event_input: Any = "",
) -> None:
    text_s = _shorten_for_log(text)
    name_s = _shorten_for_log(name)
    input_s = _shorten_for_log(event_input)
    key = (kind, name_s, text_s + input_s)
    if key in seen:
        return
    seen.add(key)
    events.append(
        {
            "kind": kind,
            "text": text_s,
            "name": name_s,
            "input": input_s,
        }
    )


def _safe_json_loads(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return raw
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def _collect_type_specific_events(
    node: dict[str, Any],
    events: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> None:
    node_type = node.get("type")
    if node_type == "thinking":
        _append_event(
            events,
            seen,
            "thinking",
            text=node.get("thinking") or node.get("text") or node.get("content"),
        )
        return

    if node_type == "reasoning":
        reasoning_content = node.get("content")
        if isinstance(reasoning_content, list):
            for rc in reasoning_content:
                if isinstance(rc, dict) and rc.get("type") == "text":
                    _append_event(events, seen, "thinking", text=rc.get("text"))
            return
        _append_event(events, seen, "thinking", text=node.get("text") or node.get("message"))
        return

    if node_type == "text":
        _append_event(events, seen, "text", text=node.get("text") or node.get("content"))
        return

    if node_type == "tool_use":
        _append_event(
            events, seen, "tool_use", name=node.get("name"), event_input=node.get("input")
        )
        return

    if node_type == "tool_result":
        _append_event(events, seen, "tool_result", text=node.get("content") or node.get("output"))
        return

    if node_type == "plugin_call":
        data_obj = node.get("data") if isinstance(node.get("data"), dict) else {}
        raw_args = data_obj.get("arguments", node.get("arguments"))
        _append_event(
            events,
            seen,
            "tool_use",
            name=data_obj.get("name") or node.get("name"),
            event_input=_safe_json_loads(raw_args),
        )
        return

    if node_type == "plugin_call_output":
        data_obj = node.get("data") if isinstance(node.get("data"), dict) else {}
        output_val = data_obj.get("output", node.get("output"))
        _append_event(events, seen, "tool_result", text=_safe_json_loads(output_val))
        return

    if node_type == "data":
        data_obj = node.get("data") if isinstance(node.get("data"), dict) else {}
        if "name" in data_obj and "arguments" in data_obj:  # type: ignore
            _append_event(
                events,
                seen,
                "tool_use",
                name=data_obj.get("name"),
                event_input=data_obj.get("arguments"),
            )
        if "output" in data_obj:  # type: ignore
            _append_event(events, seen, "tool_result", text=data_obj.get("output"))


def _collect_common_events(
    node: dict[str, Any],
    events: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> None:
    if isinstance(node.get("thinking"), str):
        _append_event(events, seen, "thinking", text=node.get("thinking"))
    if isinstance(node.get("reasoning_content"), str):
        _append_event(events, seen, "thinking", text=node.get("reasoning_content"))

    tool_calls = node.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            event_input = call.get("input")
            function = call.get("function")
            if isinstance(function, dict):
                name = name or function.get("name")
                raw_args = function.get("arguments")
                if raw_args:
                    try:
                        event_input = json.loads(raw_args)
                    except (TypeError, json.JSONDecodeError):
                        event_input = raw_args
            _append_event(events, seen, "tool_use", name=name, event_input=event_input)

    node_type = node.get("type")
    text_val = node.get("text")
    if isinstance(text_val, str) and node_type not in {"text", "thinking"}:
        _append_event(events, seen, "text", text=text_val)


def _walk_stream_node(
    node: Any,
    events: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> None:
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            _walk_stream_node(item, events, seen)
        return
    if isinstance(node, str):
        text = node.strip()
        if text:
            _append_event(events, seen, "text", text=text)
        return
    if not isinstance(node, dict):
        return

    _collect_type_specific_events(node, events, seen)
    _collect_common_events(node, events, seen)

    for key in ("delta", "message", "content", "output", "data"):
        nested = node.get(key)
        if isinstance(nested, (dict, list)):
            _walk_stream_node(nested, events, seen)


def _iter_stream_events(payload_obj: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    _walk_stream_node(payload_obj, events, seen)
    return events


def _build_content(user_input: str | list) -> list:
    if isinstance(user_input, str):
        return [{"type": "text", "text": user_input, "status": "created"}]
    return user_input


def _build_payload(content: list, session_id: str, user_id: str) -> dict[str, Any]:
    return {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": content,
            }
        ],
        "session_id": session_id,
        "user_id": user_id,
        "channel": "console",
        "stream": True,
    }


def _load_generate_kwargs(log: logging.Logger) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {}
    gen_kwargs_str = os.environ.get("AUTO_EVAL_GENERATE_KWARGS")
    if not gen_kwargs_str:
        return generate_kwargs

    try:
        parsed = json.loads(gen_kwargs_str)
    except json.JSONDecodeError:
        log.warning("AUTO_EVAL_GENERATE_KWARGS 不是合法 JSON，已忽略: %s", gen_kwargs_str)
        return generate_kwargs

    if isinstance(parsed, dict):
        generate_kwargs = parsed
    return generate_kwargs


def _configure_provider(
    *,
    log: logging.Logger,
    url: str,
    provider_name: str,
    provider_base_url: str | None,
    provider_api_key: str,
    provider_model_id: str | None,
    generate_kwargs: dict[str, Any],
) -> None:
    result = put_tool_guard_settings(url)
    log.info("put_tool_guard_settings: %s", result)

    if provider_name == RL_PROVIDER_NAME:
        result = config_provider(
            qwenpaw_url=url,
            provider_name=provider_name,
            provider_base_url=provider_base_url,
            provider_model_id=provider_model_id,
            provider_api_key=provider_api_key,
            provider_model_name="rl-model",
            generate_kwargs=generate_kwargs,
        )
        log.info("Provider configured and model activated successfully: %s", result)
        return

    config_builtin_provider(
        qwenpaw_url=url,
        provider_name=provider_name,
        provider_model_id=provider_model_id,
        provider_api_key=provider_api_key,
        provider_base_url=provider_base_url,
        generate_kwargs=generate_kwargs,
    )


def _parse_sse_line(raw_line: str, log: logging.Logger) -> dict[str, Any] | None:
    if not raw_line:
        return None

    text = raw_line.strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if text == "[DONE]":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("failed to parse stream chunk: %r", text)
        return None


def _extract_new_events(
    stream_events: list[dict[str, Any]], emitted_event_keys: set[tuple[str, str, str, str]]
) -> list[dict[str, Any]]:
    new_events: list[dict[str, Any]] = []
    for event in stream_events:
        event_key = (
            event.get("kind", ""),
            event.get("name", ""),
            event.get("text", ""),
            event.get("input", ""),
        )
        if event_key in emitted_event_keys:
            continue
        emitted_event_keys.add(event_key)
        new_events.append(event)
    return new_events


def _update_step(state: _StreamLogState, kind: str) -> None:
    if kind not in {"thinking", "text", "tool_use"}:
        return
    if not state.step_has_output:
        state.step_idx = 1
    else:
        state.step_idx += 1
    state.step_has_output = True


def _log_event(log: logging.Logger, event: dict[str, Any], state: _StreamLogState) -> None:
    kind = event["kind"]
    _update_step(state, kind)

    if kind == "thinking" and event["text"]:
        log.info("[step %d] [thinking] %s", state.step_idx, event["text"])
        return

    if kind == "tool_use":
        state.tool_event_count += 1
        if event["input"]:
            log.info(
                "[step %d] [tool] %s(%s)",
                state.step_idx,
                event["name"] or "unknown",
                event["input"],
            )
        else:
            log.info("[step %d] [tool] %s", state.step_idx, event["name"] or "unknown")
        return

    if kind == "tool_result" and event["text"]:
        if state.step_idx == 0:
            state.step_idx = 1
            state.step_has_output = True
        log.info("[step %d] [tool_result] %s", state.step_idx, event["text"])
        return

    if kind == "text" and event["text"]:
        log.info("[step %d] [text] %s", state.step_idx, event["text"])


def _consume_stream(response: requests.Response, log: logging.Logger) -> _StreamLogState:
    state = _StreamLogState()
    emitted_event_keys: set[tuple[str, str, str, str]] = set()

    for raw_line in response.iter_lines(decode_unicode=True):
        data = _parse_sse_line(raw_line, log)
        if data is None:
            continue

        stream_events = _iter_stream_events(data)
        if not stream_events:
            continue

        new_events = _extract_new_events(stream_events, emitted_event_keys)
        if not new_events:
            continue

        for event in new_events:
            _log_event(log, event, state)

    return state


def _request_stop(url: str, session_id: str, user_id: str, log: logging.Logger) -> bool:
    chat_id = _resolve_chat_id(url, session_id, user_id, log)
    stop_attempts: list[tuple[str, str, bool]] = []
    for target in _stop_targets(session_id, chat_id):
        params = {"chat_id": target}
        for endpoint in _STOP_ENDPOINTS:
            stop_url = f"{url}{endpoint}"
            try:
                response = requests.post(stop_url, params=params, timeout=10)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json() if response.content else {}
                stopped = bool(payload.get("stopped"))
                stop_attempts.append((endpoint, target, stopped))
                log.info(
                    "sent stop signal for session=%s target=%s endpoint=%s stopped=%s",
                    session_id,
                    target,
                    endpoint,
                    stopped,
                )
                if stopped:
                    return True
            except requests.RequestException as exc:
                log.warning(
                    "failed to send stop signal for session=%s target=%s via %s: %s",
                    session_id,
                    target,
                    endpoint,
                    exc,
                )
            except ValueError as exc:
                log.warning(
                    "stop endpoint returned non-json response for session=%s target=%s via %s: %s",
                    session_id,
                    target,
                    endpoint,
                    exc,
                )

    if stop_attempts:
        log.warning("all stop attempts returned stopped=false for session=%s", session_id)
    return False


def call_agent(
    url: str,
    user_input: str | list,
    session_id: str,
    user_id: str,
    provider_name: str,
    provider_base_url: str | None,
    provider_api_key: str,
    provider_model_id: str | None,
    timeout_seconds: float | None = None,
) -> CallAgentResult:
    log = logging.getLogger(__name__)

    content = _build_content(user_input)
    log.info("Calling agent with input: %s", str(content))

    payload = _build_payload(content, session_id=session_id, user_id=user_id)

    headers = {"Referer": f"{url}/chat", "content-type": "application/json"}

    # auto_eval.py → sandbox_utils.launch_run_py 注入的每请求 sampling kwargs
    # （已分流：OpenAI 标准字段在顶层，vLLM 私有字段在 extra_body 里），
    # 写进 provider 配置后 CoPaw 调 OpenAI 兼容接口时会作为 kwargs 下发。
    generate_kwargs = _load_generate_kwargs(log)

    _configure_provider(
        log=log,
        url=url,
        provider_name=provider_name,
        provider_base_url=provider_base_url,
        provider_api_key=provider_api_key,
        provider_model_id=provider_model_id,
        generate_kwargs=generate_kwargs,
    )

    worker_state: dict[str, Any] = {"response": None, "state": None, "exception": None}

    def _run_stream() -> None:
        response: requests.Response | None = None
        try:
            response = requests.post(
                f"{url}/api/agent/process",
                json=payload,
                headers=headers,
                stream=True,
            )
            worker_state["response"] = response
            response.raise_for_status()
            worker_state["state"] = _consume_stream(response, log)
        except Exception as exc:  # noqa: BLE001
            worker_state["exception"] = exc
        finally:
            if response is not None:
                response.close()

    worker = threading.Thread(target=_run_stream, name="call-agent-stream", daemon=True)
    worker.start()

    timed_out = False
    stop_succeeded = False
    if timeout_seconds and timeout_seconds > 0:
        worker.join(timeout_seconds)
        if worker.is_alive():
            timed_out = True
            log.warning(
                "call_agent timed out after %.2f seconds, sending stop signal for session=%s",
                timeout_seconds,
                session_id,
            )
            stop_succeeded = _request_stop(url, session_id, user_id, log)
            worker.join(_STOP_WAIT_SECONDS)
            if worker.is_alive():
                response = worker_state.get("response")
                if response is not None:
                    response.close()
                worker.join(2)
    else:
        worker.join()

    stream_exc = worker_state.get("exception")
    if stream_exc is not None:
        if timed_out:
            log.warning("call_agent stream exited after timeout: %s", stream_exc)
        else:
            raise stream_exc

    state = worker_state.get("state") or _StreamLogState()
    if state.tool_event_count == 0:
        log.info(
            "stream did not include tool events; tool calls will be visible from session trajectory"
        )
    if timed_out and not stop_succeeded:
        log.warning(
            "call_agent timed out but no stop endpoint confirmed interruption for session=%s",
            session_id,
        )
    log.info(
        "stream completed, total_steps=%d, tool_events=%d", state.step_idx, state.tool_event_count
    )
    return CallAgentResult(timed_out=timed_out, stop_succeeded=stop_succeeded)
