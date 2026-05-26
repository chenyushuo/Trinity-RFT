"""Unit tests for judge reward aggregation and penalties."""

from __future__ import annotations

import unittest

from judge_reward import (
    GraderScoreEntry,
    RewardPolicy,
    aggregate_grader_scores,
    compute_step_penalty,
    finalize_reward,
)


class AggregateGraderScoresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RewardPolicy(baseline_subtract=0.0)

    def test_outcome_with_trajectory_uses_multiplicative_blend(self) -> None:
        entries = [
            GraderScoreEntry("correctness", 0.75),
            GraderScoreEntry("trajectory", 1.0),
        ]
        score, detail = aggregate_grader_scores(
            entries,
            domain="gui",
            policy=self.policy,
        )
        expected = 0.75 * (0.8 + 0.2 * 1.0)
        self.assertAlmostEqual(score, expected, places=4)
        self.assertIn("blend=", detail)

    def test_search_domain_uses_geometric_mean(self) -> None:
        entries = [
            GraderScoreEntry("search_hallucination", 0.80),
            GraderScoreEntry("search_relevance", 0.50),
            GraderScoreEntry("trajectory", 1.0),
        ]
        score, detail = aggregate_grader_scores(
            entries,
            domain="search",
            policy=self.policy,
        )
        base = (0.80 * 0.50) ** 0.5
        expected = base * (0.8 + 0.2 * 1.0)
        self.assertAlmostEqual(score, expected, places=4)
        self.assertIn("search_geometric_mean", detail)

    def test_process_only_domain_keeps_trajectory_mean(self) -> None:
        entries = [GraderScoreEntry("trajectory", 0.80)]
        score, detail = aggregate_grader_scores(
            entries,
            domain="bootstrap",
            policy=self.policy,
        )
        self.assertAlmostEqual(score, 0.80, places=4)
        self.assertIn("process_only", detail)


class FinalizeRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RewardPolicy()
        self.session = {
            "agent": {
                "_model_trajectory": [
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "response": [{"type": "text", "text": "done"}],
                    }
                ],
                "memory": {"content": []},
            }
        }

    def test_step_penalty_tiers(self) -> None:
        policy = RewardPolicy()
        self.assertEqual(compute_step_penalty(50, policy), 0.0)
        self.assertAlmostEqual(compute_step_penalty(75, policy), 0.075, places=4)
        self.assertAlmostEqual(compute_step_penalty(100, policy), 0.15, places=4)
        self.assertAlmostEqual(compute_step_penalty(125, policy), 0.225, places=4)
        self.assertAlmostEqual(compute_step_penalty(150, policy), 0.30, places=4)

    def test_empty_final_answer_caps_reward(self) -> None:
        empty_session = {
            "agent": {
                "_model_trajectory": [{"messages": [], "response": []}],
                "memory": {"content": []},
            }
        }
        score, detail = finalize_reward(
            0.80,
            empty_session,
            has_answer=False,
            outcome_scores=[0.80],
            any_high_variance=False,
            policy=self.policy,
        )
        self.assertLessEqual(score, 0.10)
        self.assertIn("empty_final_answer", detail)

    def test_messages_end_with_tool_is_hard_terminated(self) -> None:
        truncated_session = {
            "agent": {
                "_model_trajectory": [
                    {
                        "messages": [
                            {"role": "user", "content": "go"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": "{}"},
                                    }
                                ],
                            },
                            {"role": "tool", "content": "file content"},
                        ],
                        "response": [],
                    }
                ],
                "memory": {"content": []},
            }
        }
        score, detail = finalize_reward(
            0.90,
            truncated_session,
            has_answer=False,
            outcome_scores=[0.90],
            any_high_variance=False,
            policy=self.policy,
        )
        self.assertEqual(score, 0.0)
        self.assertIn("hard_terminated", detail)

    def test_hard_terminated_zeros_reward(self) -> None:
        hard_session = {
            "agent": {
                "_model_trajectory": [
                    {
                        "messages": [],
                        "response": [{"type": "tool_use", "name": "browser_use", "input": {}}],
                    }
                ],
                "memory": {"content": []},
            }
        }
        score, detail = finalize_reward(
            0.90,
            hard_session,
            has_answer=False,
            outcome_scores=[0.90],
            any_high_variance=False,
            policy=self.policy,
        )
        self.assertEqual(score, 0.0)
        self.assertIn("hard_terminated", detail)


if __name__ == "__main__":
    unittest.main()
