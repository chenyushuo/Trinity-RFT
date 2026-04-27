"""Unified tool-output context shared by all graders (regardless of task domain).

Design (no domain-based branching):
  1. Each tool call → dispatched to its cleaner (browser snapshot / curl HTML / raw)
  2. Cross-tool content fingerprint dedup (same page scraped twice, same shell repeated)
  3. Single packing function (``_pack_entries_reverse_fill``); reverse-fill on overflow
     since later calls are usually more specific and closer to the final answer

Differences across graders are *only* in how they consume the context:
  - correctness / search_relevance: 100k reverse-fill → single grader pass
  - hallucination: total length ≤ 30k → single grader (shares the same cleaner pipeline);
    > 30k → MapReduce (raw entries kept for chunking)

This module depends only on ``_core`` (for ``logger``-style imports it grabs locally).
"""

import json
import logging
import os
import re
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truncated tool-result recovery (QwenPaw / CoPaw markers)
# ---------------------------------------------------------------------------

# QwenPaw agents/tools/utils.py — 截断提示含该标记与 file_path= 续读路径
_TRUNCATION_NOTICE_MARKER = "<<<TRUNCATED>>>"
_FILE_PATH_IN_NOTICE_RE = re.compile(r"file_path=(/\S+)")


def _tool_message_content_to_str(content: Any) -> str:
    """将 trajectory 里 tool 消息的 content 统一为字符串（str 或 text 块列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text":
                parts.append(blk.get("text", "") or "")
            elif isinstance(blk.get("text"), str):
                parts.append(blk["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _recover_truncated_output(output: str) -> str:
    """检测 QwenPaw / CoPaw 截断通知，尝试从磁盘读取完整文件内容。

    QwenPaw（agents/tools/utils.py）在超长输出末尾追加::
      <<<TRUNCATED>>>\\nThe output above was truncated.\\n...\\
      call `read_file` with file_path=<绝对路径> start_line=...

    完整内容保存在该路径指向的文件中（含 tool_result 落盘 hash 文件、或 read_file
    正在读取的任意工作区文件）。评测容器内路径仍以 /app/working 为根。

    仅当能在本地打开文件且读出非空内容时才替换；否则保留原始（含摘要）字符串。
    """
    if not output:
        return output
    if "The output above was truncated" not in output and _TRUNCATION_NOTICE_MARKER not in output:
        return output

    seen: set[str] = set()
    for m in _FILE_PATH_IN_NOTICE_RE.finditer(output):
        raw = m.group(1)
        fpath = raw.rstrip(".,);'\"")
        if not fpath.startswith("/app/working/") or fpath in seen:
            continue
        seen.add(fpath)
        try:
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                full = f.read()
            if full:
                logger.info("从磁盘恢复截断 tool_result: %s (%d chars)", fpath, len(full))
                return full
        except OSError:
            continue
    return output


def _extract_tool_call_results(
    session: dict,
    tool_names: set[str] | None = None,
) -> list[dict[str, str]]:
    """从 _model_trajectory 提取指定工具的调用及其返回结果。

    对被框架截断的 tool result，自动尝试从磁盘读取完整文件恢复。

    Returns:
        [{"name": "tool_name", "input": "...", "output": "..."}]
    """
    traj = session.get("agent", {}).get("_model_trajectory", [])
    calls: list[dict[str, str]] = []

    for entry in traj:
        response = entry.get("response", [])
        if not isinstance(response, list):
            continue
        for item in response:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            name = item.get("name", "")
            if tool_names and name not in tool_names:
                continue
            calls.append({
                "id": item.get("id", ""),
                "name": name,
                "input": json.dumps(item.get("input", {}), ensure_ascii=False)[:5000],
                "output": "",
            })

    for entry in traj:
        for msg in entry.get("messages", []):
            if msg.get("role") != "tool":
                continue
            tid = msg.get("tool_call_id", "")
            text = _tool_message_content_to_str(msg.get("content", ""))
            if text:
                for c in calls:
                    if c["id"] == tid and not c["output"]:
                        c["output"] = _recover_truncated_output(text)

    return calls


# ---------------------------------------------------------------------------
# Browser snapshot cleaner
# ---------------------------------------------------------------------------

_BROWSER_SKIP_ACTIONS = {
    "open", "click", "wait_for", "stop", "navigate",
    "handle_dialog", "start", "type", "fill", "scroll",
}

_SNAPSHOT_STRIP_RES: list[re.Pattern] = [
    re.compile(r"^\s*- /url:.*$", re.MULTILINE),
    re.compile(r"^\s*- img\b.*$", re.MULTILINE),
    re.compile(
        r"^\s*- (?:document|iframe|banner|complementary|navigation"
        r"|button|searchbox)\b[^\"]*$",
        re.MULTILINE,
    ),
    re.compile(r"^\s*- (?:list|listitem):?\s*$", re.MULTILINE),
    re.compile(r"\s*\[ref=e\d+\]"),
    re.compile(r"\s*\[nth=\d+\]"),
    re.compile(r"\s*\[level=\d+\]")
]
_SNAPSHOT_BLANK_LINES_RE = re.compile(r"\n{2,}")


def _clean_browser_output(output: str) -> str:
    """清洗 browser_use snapshot，去除 DOM 噪音只保留有效文本。

    过滤: URL 行、img、结构标签、ref/nth/level 标签、
    短 link/text（导航栏）、连续空行。

    对被框架截断导致 JSON 格式破坏的 snapshot，用字符串匹配提取内容后再清洗。
    """
    text: str | None = None
    try:
        data = json.loads(output)
        if isinstance(data, dict) and "snapshot" in data:
            text = data["snapshot"]
        else:
            return output
    except (json.JSONDecodeError, TypeError):
        marker = '"snapshot": "'
        idx = output.find(marker)
        if idx >= 0:
            raw = output[idx + len(marker):]
            text = raw.replace("\\n", "\n").replace('\\"', '"')
        else:
            return output

    for pat in _SNAPSHOT_STRIP_RES:
        text = pat.sub("", text)
    text = _SNAPSHOT_BLANK_LINES_RE.sub("\n", text)
    return text.strip()


def _is_browser_action_skip(call: dict) -> bool:
    """判断 browser_use 调用是否为不产生有效内容的操作型 action。"""
    if call["name"] != "browser_use":
        return False
    try:
        inp = json.loads(call["input"]) if isinstance(call["input"], str) else call["input"]
    except (json.JSONDecodeError, TypeError):
        return False
    return inp.get("action", "") in _BROWSER_SKIP_ACTIONS


def _content_fingerprint(text: str) -> str:
    """取清洗后内容中段的一小块作为指纹，避免导航栏干扰。"""
    if not text:
        return ""
    mid = len(text) // 3
    return text[mid : mid + 600]


# ---------------------------------------------------------------------------
# execute_shell_command + curl/wget as search fallback
# ---------------------------------------------------------------------------
#
# 部分 site 反爬严格（Cloudflare 等），browser_use 被拦后 Agent 会回退到
# `execute_shell_command` + `curl`/`wget` 抓取 HTML。把它视作「搜索」工具：
# 当 shell 命令包含 curl/wget 且含 http(s):// URL 时识别，并对返回 HTML 剥噪。

_SHELL_HTTP_RE = re.compile(
    r"\b(curl|wget|fetch)\b[^\n]*\bhttps?://", re.IGNORECASE,
)
_HTML_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _is_search_shell_call(call: dict) -> bool:
    """识别 `execute_shell_command` 是否在做网页抓取 (curl/wget + http URL)。"""
    if call.get("name") != "execute_shell_command":
        return False
    inp = call.get("input", "")
    if isinstance(inp, dict):
        inp = json.dumps(inp, ensure_ascii=False)
    if not isinstance(inp, str):
        return False
    return bool(_SHELL_HTTP_RE.search(inp))


def _clean_curl_output(output: str) -> str:
    """剥离 curl/wget 抓回的 HTML：移除 script/style、标签，折叠空行。

    返回纯文本主体，过短时保留原样（极端情况下 HTML 为压缩单行）。
    """
    if not output:
        return ""
    text = _HTML_SCRIPT_STYLE_RE.sub("", output)
    text = _HTML_TAG_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()
    return text if len(text) >= 50 else output


# ---------------------------------------------------------------------------
# Unified tool-output pipeline
# ---------------------------------------------------------------------------

def _clean_tool_output(call: dict) -> str:
    """根据工具类型派发到对应 cleaner，返回清洗后的文本（空字符串=本调用应跳过）。

    - browser_use: 跳过 open/click 等 action 调用；snapshot 走 _clean_browser_output
    - execute_shell_command: 仅当是 curl/wget 类网页抓取时清洗 HTML；其他保持原样
    - 其他工具: 原样输出
    """
    name = call.get("name", "")
    output = call.get("output", "") or ""
    if not output.strip():
        return ""

    if name == "browser_use":
        if _is_browser_action_skip(call):
            return ""
        cleaned = _clean_browser_output(output)
        return cleaned if cleaned else output

    if name == "execute_shell_command" and _is_search_shell_call(call):
        cleaned = _clean_curl_output(output)
        return cleaned if cleaned else output

    return output


def build_unified_entries(session: dict) -> list[tuple[str, str]]:
    """收集 session 中所有工具调用的输出，按统一规则清洗 + 去重。

    返回 [(header, content), ...] 的有序列表（按调用时间）。
    后续可由：
      - `_pack_entries_reverse_fill` 拼成单次 grader 用的 context；
      - `_evaluate_hallucination_mapreduce` 直接消费做 chunking。
    """
    calls = _extract_tool_call_results(session, None)
    if not calls:
        return []

    seen_fingerprints: list[str] = []
    entries: list[tuple[str, str]] = []
    for i, c in enumerate(calls, 1):
        cleaned = _clean_tool_output(c)
        if not cleaned:
            continue
        # 跨工具内容去重：同一页面被反复抓取或重复 shell 输出只保留一次
        fp = _content_fingerprint(cleaned)
        if fp and any(fp == prev for prev in seen_fingerprints):
            continue
        if fp:
            seen_fingerprints.append(fp)
        header = f"[Tool {i}] {c.get('name','')}: {(c.get('input','') or '')[:2000]}"
        entries.append((header, cleaned))
    return entries


def _pack_entries_reverse_fill(
    entries: list[tuple[str, str]],
    max_chars: int = 100_000,
    *,
    with_date_header: bool = True,
) -> str:
    """把 (header, content) entries 打包成 grader context 字符串。

    超预算时从后往前填充——尾部的工具调用通常更接近最终回答、更具体。
    `with_date_header=True` 会在最前加入 `[评测日期: YYYY-MM-DD]` 防止 grader 用过时的
    时间假设否定真实数据。
    """
    if not entries:
        return ""

    header_line = f"[评测日期: {date.today().isoformat()}]" if with_date_header else ""
    budget = max_chars - (len(header_line) if header_line else 0)
    total = sum(len(h) + 1 + len(o) for h, o in entries)

    if total <= budget:
        body = [f"{h}\n{o}" for h, o in entries]
    else:
        selected: list[str] = []
        used = 0
        for h, o in reversed(entries):
            entry_text = f"{h}\n{o}"
            if used + len(entry_text) <= budget:
                selected.append(entry_text)
                used += len(entry_text)
            else:
                remaining = budget - used
                if remaining > len(h) + 100:
                    selected.append(f"{h}\n{o[:remaining - len(h) - 1]}")
                break
        selected.reverse()
        body = selected

    if header_line:
        return "\n---\n".join([header_line, *body])
    return "\n---\n".join(body)


def _get_context(session: dict) -> str:
    """提取 grader 用的工具上下文文本。

    所有 grader 共享同一份「过程中所有工具的统一上下文」：
    收集所有工具调用 → 统一清洗/去重 → 100k 反向填充。
    """
    return _pack_entries_reverse_fill(build_unified_entries(session), 100_000)


# ---------------------------------------------------------------------------
# Agent capability context (注入给 hallucination grader 防止误判内置技能)
# ---------------------------------------------------------------------------

def extract_agent_capability_context(session: dict, max_chars: int = 3000) -> str:
    """从 session 中提取 Agent 的能力上下文（skill 列表、工具说明等）。

    用于注入到幻觉 grader 的 context 中，让 grader 理解这是一个有工具和技能的
    CoPaw Agent，而非普通 LLM 聊天，避免将 Agent 提到的真实技能误判为"幻觉"。
    """
    preamble = (
        "[Agent 系统能力说明]\n"
        "被评估的回答来自一个 CoPaw Agent，它具备以下能力：\n"
        "- 可通过工具直接访问用户本地文件系统（读写文件、执行 shell 命令）\n"
        "- 可使用浏览器工具访问网页\n"
        "- 拥有以下已安装的 Skills（技能），Agent 提到这些技能时不应视为幻觉：\n"
    )

    skill_lines: list[str] = []
    _skill_pattern = re.compile(
        r"## ([\w_-]+)\n((?:(?!\n## ).)*?)Check\s+\"[^\"]*?/skills/[^\"]*?/SKILL\.md\"",
        re.DOTALL,
    )

    def _extract_skills_from_text(text: str) -> None:
        for m in _skill_pattern.finditer(text):
            name = m.group(1).strip()
            desc_raw = m.group(2).strip()
            first_line = desc_raw.split("\n")[0].strip()[:200]
            skill_lines.append(f"  - {name}: {first_line}")

    traj = session.get("agent", {}).get("_model_trajectory", [])
    if traj:
        msgs = traj[0].get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if isinstance(content, str):
                _extract_skills_from_text(content)

    if not skill_lines:
        sys_prompt = session.get("agent", {}).get("_sys_prompt", "")
        if sys_prompt:
            _extract_skills_from_text(sys_prompt)

    if skill_lines:
        result = preamble + "\n".join(skill_lines[:30])
    else:
        result = (
            "[Agent 系统能力说明]\n"
            "被评估的回答来自一个 CoPaw Agent，它具备以下能力：\n"
            "- 可通过工具直接访问用户本地文件系统（读写文件、执行 shell 命令）\n"
            "- 可使用浏览器工具访问网页\n"
            "- 可能拥有已安装的 Skills（技能），Agent 提到的技能可能真实存在于其系统中\n"
            "评估幻觉时请注意：Agent 声称能访问文件、执行命令、使用已安装的技能，"
            "这些都是其真实能力，不应被视为幻觉。"
        )

    return result[:max_chars]
