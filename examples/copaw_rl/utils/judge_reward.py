"""Reward aggregation and penalty logic for copaw_rl judge (no LLM dependencies)."""

from __future__ import annotations

import logging
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraderScoreEntry:
    evaluator: str
    normalized: float
    include_in_score: bool = True
    high_variance: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class RewardPolicy:
    """RL reward 聚合与惩罚策略（可通过环境变量微调部分参数）。"""

    trajectory_blend_alpha: float = 0.2
    trajectory_base_floor: float = 0.8
    use_search_geometric_mean: bool = True
    baseline_subtract: float = 0.0
    hard_terminated_cap: float = 0.0
    empty_final_answer_cap: float = 0.10
    step_soft_limit: int = 50
    step_hard_limit: int = 100
    step_penalty_mid_cap: float = 0.15
    step_penalty_per_extra_mid: float = 0.003
    step_penalty_hard_base: float = 0.15
    step_penalty_per_extra_hard: float = 0.003
    step_penalty_cap: float = 0.30
    high_variance_multiplier: float = 0.85
    outcome_disagreement_std_threshold: float = 0.25
    outcome_disagreement_multiplier: float = 0.80


PROCESS_EVALUATORS = frozenset({"trajectory"})


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def load_reward_policy() -> RewardPolicy:
    """从环境变量加载 reward 策略，未设置时使用保守默认值。"""
    return RewardPolicy(
        trajectory_blend_alpha=_env_float("JUDGE_TRAJ_BLEND_ALPHA", 0.2),
        trajectory_base_floor=_env_float("JUDGE_TRAJ_BASE_FLOOR", 0.8),
        baseline_subtract=_env_float("JUDGE_REWARD_BASELINE_SUBTRACT", 0.0),
        step_soft_limit=int(_env_float("JUDGE_STEP_SOFT_LIMIT", 50)),
        step_hard_limit=int(_env_float("JUDGE_STEP_HARD_LIMIT", 100)),
    )


DEFAULT_REWARD_POLICY = load_reward_policy()


def extract_final_text_from_session(session: Mapping[str, Any]) -> str:
    trajectory = session.get("agent", {}).get("_model_trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        return ""
    last_entry = trajectory[-1]
    if not isinstance(last_entry, Mapping):
        return ""
    response = last_entry.get("response", [])
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text", ""))
    return ""


def is_hard_terminated_session(session: Mapping[str, Any]) -> bool:
    """会话未完成：无最终文本，且 _model_trajectory 显示停在 tool 链上。

    只读 ``agent._model_trajectory`` 最后一条 entry（judge / RL 的主数据源），
    不再单独查 ``memory.content``，避免两处格式不一致、重复判定。

    命中条件（满足任一）：
      - 最后一步 ``response`` 含 ``tool_use`` 且无 ``text``（调用发出后会话结束）
      - 最后一步 ``messages`` 以 ``role=tool`` 结尾（工具已返回，模型未续推理）
    """
    if extract_final_text_from_session(session).strip():
        return False

    trajectory = session.get("agent", {}).get("_model_trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        return False
    last_entry = trajectory[-1]
    if not isinstance(last_entry, Mapping):
        return False

    messages = last_entry.get("messages", [])
    if isinstance(messages, list) and messages:
        last_msg = messages[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "tool":
            return True

    response = last_entry.get("response", [])
    if not isinstance(response, list):
        return False
    has_tool_use = any(
        isinstance(item, dict) and item.get("type") == "tool_use" for item in response
    )
    text_parts = [
        str(item.get("text", ""))
        for item in response
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return has_tool_use and not "\n".join(text_parts).strip()


def count_trajectory_steps(session: Mapping[str, Any]) -> int:
    trajectory = session.get("agent", {}).get("_model_trajectory", [])
    return len(trajectory) if isinstance(trajectory, list) else 0


def weighted_mean(entries: list[GraderScoreEntry]) -> float:
    if not entries:
        return 0.0
    total_weight = sum(entry.weight for entry in entries)
    if total_weight <= 0:
        return sum(entry.normalized for entry in entries) / len(entries)
    return sum(entry.normalized * entry.weight for entry in entries) / total_weight


def compute_step_penalty(steps: int, policy: RewardPolicy = DEFAULT_REWARD_POLICY) -> float:
    """按步数档位计算惩罚比例（0~step_penalty_cap）。

    - steps ≤ 50：不罚
    - 50 < steps ≤ 100：线性累加，最多扣 15%
    - steps > 100：在 15% 基础上继续累加，总上限 30%
    """
    if steps <= policy.step_soft_limit:
        return 0.0
    if steps <= policy.step_hard_limit:
        extra = steps - policy.step_soft_limit
        return min(
            policy.step_penalty_mid_cap,
            policy.step_penalty_per_extra_mid * extra,
        )
    extra = steps - policy.step_hard_limit
    return min(
        policy.step_penalty_cap,
        policy.step_penalty_hard_base + policy.step_penalty_per_extra_hard * extra,
    )


def aggregate_grader_scores(
    entries: list[GraderScoreEntry],
    *,
    domain: str,
    policy: RewardPolicy = DEFAULT_REWARD_POLICY,
) -> tuple[float, str]:
    """聚合各 grader 分数：outcome 主导，trajectory 仅做乘法微调。"""
    scoring = [entry for entry in entries if entry.include_in_score]
    if not scoring:
        return 0.0, "no_scoring_entries"

    outcome = [entry for entry in scoring if entry.evaluator not in PROCESS_EVALUATORS]
    process = [entry for entry in scoring if entry.evaluator in PROCESS_EVALUATORS]

    if outcome:
        if policy.use_search_geometric_mean and domain == "search" and len(outcome) >= 2:
            values = [entry.normalized for entry in outcome]
            if any(value <= 0.0 for value in values):
                base = 0.0
            else:
                base = math.prod(values) ** (1.0 / len(values))
            agg_mode = "search_geometric_mean"
        else:
            base = weighted_mean(outcome)
            agg_mode = "outcome_weighted_mean"

        if process:
            traj = weighted_mean(process)
            floor = policy.trajectory_base_floor
            alpha = policy.trajectory_blend_alpha
            final = base * (floor + alpha * traj)
            detail = (
                f"base={base:.4f}({agg_mode}), traj={traj:.4f}, "
                f"blend={floor:.2f}+{alpha:.2f}*traj→{final:.4f}"
            )
        else:
            final = base
            detail = f"base={base:.4f}({agg_mode})"
    else:
        final = weighted_mean(process)
        detail = f"process_only={final:.4f}"

    return final, detail


def finalize_reward(
    raw_score: float,
    session: Mapping[str, Any],
    *,
    has_answer: bool,
    outcome_scores: list[float],
    any_high_variance: bool,
    policy: RewardPolicy = DEFAULT_REWARD_POLICY,
) -> tuple[float, str]:
    """在 grader 聚合分之上应用确定性/稳定性惩罚与 baseline 校准。"""
    multiplier = 1.0
    cap: Optional[float] = None
    penalty_lines: list[str] = []

    if is_hard_terminated_session(session):
        cap = policy.hard_terminated_cap
        penalty_lines.append(f"hard_terminated→cap={policy.hard_terminated_cap:.2f}")

    if not has_answer:
        empty_cap = policy.empty_final_answer_cap
        cap = empty_cap if cap is None else min(cap, empty_cap)
        penalty_lines.append(f"empty_final_answer→cap={empty_cap:.2f}")

    steps = count_trajectory_steps(session)
    step_penalty = compute_step_penalty(steps, policy)
    if step_penalty > 0.0:
        multiplier *= 1.0 - step_penalty
        tier = (
            f">{policy.step_hard_limit}"
            if steps > policy.step_hard_limit
            else f"{policy.step_soft_limit}-{policy.step_hard_limit}"
        )
        penalty_lines.append(
            f"steps={steps}({tier}), step_penalty×{1.0 - step_penalty:.2f}"
        )

    if any_high_variance:
        multiplier *= policy.high_variance_multiplier
        penalty_lines.append(f"high_variance×{policy.high_variance_multiplier:.2f}")

    if len(outcome_scores) >= 2:
        std = statistics.pstdev(outcome_scores)
        if std > policy.outcome_disagreement_std_threshold:
            multiplier *= policy.outcome_disagreement_multiplier
            penalty_lines.append(
                f"outcome_disagreement_std={std:.2f}×{policy.outcome_disagreement_multiplier:.2f}"
            )

    score = raw_score * multiplier
    if cap is not None:
        score = min(score, cap)

    if policy.baseline_subtract > 0.0:
        score = max(0.0, score - policy.baseline_subtract)
        penalty_lines.append(f"baseline_subtract={policy.baseline_subtract:.2f}")

    detail = f"raw={raw_score:.4f}, mult={multiplier:.4f}, final={score:.4f}"
    if penalty_lines:
        detail += " [" + "; ".join(penalty_lines) + "]"
    return score, detail
