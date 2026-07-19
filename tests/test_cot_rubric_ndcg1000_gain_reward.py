from __future__ import annotations

import json
import math
import os
import unittest
from unittest import mock

from manu_src.api_info.rubric_target_relevance_api import (
    TargetRelevanceJudgeAPIClient,
)

from manu_src.scripts.train.cot_rubric_ndcg1000_gain_reward import (
    _api_score_norm,
    _apply_group_ndcg_only_fallback,
    _combine_group_rewards,
    _joint_gain,
    _parse_api_keys,
)
from rubric_cot_pipeline.judge_api import JudgeAPIResult
from rubric_cot_pipeline.prompts import build_judge_messages


class _RecordingJudgeClient:
    def __init__(self) -> None:
        self.target_items: list[str] = []
        self.target_usages: list[str] = []

    def score(
        self,
        user_history: str,
        cot: str,
        target_item: str = "",
        target_usage: str = "leakage",
    ) -> JudgeAPIResult:
        self.target_items.append(target_item)
        self.target_usages.append(target_usage)
        return JudgeAPIResult(
            score={"score_norm": 0.8},
            raw='{"score_norm": 0.8}',
            provider="recording",
        )


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class CotRubricNdcg1000GainRewardTest(unittest.TestCase):
    def test_missing_rubric_score_falls_back_for_entire_group(self) -> None:
        scores, fallback_mask, fallback_groups = _apply_group_ndcg_only_fallback(
            [0.8, None, 0.6, 0.9, 0.2, 0.4, 0.6, 0.8],
            {("example", 0): [0, 1, 2, 3], ("example", 1): [4, 5, 6, 7]},
        )
        self.assertEqual(scores[:4], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(scores[4:], [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(fallback_mask, [True] * 4 + [False] * 4)
        self.assertEqual(fallback_groups, 1)

    def test_tokenverse_rubric_request_disables_reasoning(self) -> None:
        captured_payloads: list[dict] = []

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "preference_grounding": 4,
                                        "taste_specificity": 4,
                                        "transitional_reasoning": 4,
                                        "discriminative_framing": 4,
                                        "conciseness": 4,
                                    }
                                ),
                                "reasoning_content": "",
                            }
                        }
                    ]
                }
            )

        client = TargetRelevanceJudgeAPIClient(
            provider="ks_tokenverse",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="glm-5-2",
            timeout=1,
            max_retries=0,
        )
        with mock.patch.dict(
            os.environ,
            {
                "COT_RUBRIC_NDCG_GAIN_API_THINKING": "disabled",
                "COT_RUBRIC_NDCG_GAIN_API_MAX_TOKENS": "128",
            },
        ), mock.patch(
            "manu_src.api_info.rubric_target_relevance_api.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = client.score("history", "candidate", "target")

        self.assertIsNotNone(result.score)
        payload = captured_payloads[0]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 128)

    def test_rubric_parser_does_not_use_reasoning_content(self) -> None:
        response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "preference_grounding": 5,
                                    "taste_specificity": 5,
                                    "transitional_reasoning": 5,
                                    "discriminative_framing": 5,
                                    "conciseness": 5,
                                }
                            ),
                            "reasoning_content": json.dumps(
                                {
                                    "preference_grounding": 5,
                                    "taste_specificity": 5,
                                    "transitional_reasoning": 5,
                                    "discriminative_framing": 5,
                                    "conciseness": 5,
                                }
                            ),
                        }
                    }
                ]
            }
        )
        client = TargetRelevanceJudgeAPIClient(
            provider="ks_tokenverse",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="glm-5-2",
            timeout=1,
            max_retries=0,
        )
        with mock.patch(
            "manu_src.api_info.rubric_target_relevance_api.urllib.request.urlopen",
            return_value=response,
        ):
            result = client.score("history", "candidate", "target")

        self.assertIsNone(result.score)

    def test_api_judge_receives_target_information(self) -> None:
        client = _RecordingJudgeClient()
        score, _, provider = _api_score_norm(
            client,
            "history",
            "generated cot",
            "held-out target item",
        )
        self.assertEqual(client.target_items, ["held-out target item"])
        self.assertEqual(client.target_usages, ["relevance"])
        self.assertAlmostEqual(score or 0.0, 0.8)
        self.assertEqual(provider, "recording")

    def test_target_relevance_prompt_does_not_apply_leakage_caps(self) -> None:
        messages = build_judge_messages(
            "history",
            "generated cot",
            "next item",
            target_usage="relevance",
        )
        prompt = messages[1]["content"]
        self.assertIn("evaluating the predicted interest direction", prompt)
        self.assertNotIn("leakage check", prompt)
        self.assertNotIn("at most 2", prompt)
        self.assertNotIn("Do not reward similarity", prompt)
        self.assertNotIn("direct leakage wording", prompt)
        self.assertIn("captures the interest direction", prompt)

    def test_api_key_pool_accepts_json_and_delimited_values(self) -> None:
        self.assertEqual(_parse_api_keys('["key-a", "key-b", "key-a"]'), ["key-a", "key-b"])
        self.assertEqual(_parse_api_keys("key-a,key-b\nkey-a"), ["key-a", "key-b"])

    def test_positive_gain_is_weighted_by_rubric_quality(self) -> None:
        self.assertAlmostEqual(
            _joint_gain(
                0.8,
                0.1,
                quality_power=1.0,
                negative_gain_weight=1.0,
            ),
            0.08,
        )

    def test_negative_gain_is_not_attenuated_by_low_quality(self) -> None:
        low_quality = _joint_gain(
            0.2,
            -0.1,
            quality_power=1.0,
            negative_gain_weight=1.0,
        )
        high_quality = _joint_gain(
            0.9,
            -0.1,
            quality_power=1.0,
            negative_gain_weight=1.0,
        )
        self.assertAlmostEqual(low_quality, -0.1)
        self.assertAlmostEqual(high_quality, -0.1)

    def test_quality_power_only_changes_positive_gain(self) -> None:
        self.assertAlmostEqual(
            _joint_gain(
                0.5,
                0.2,
                quality_power=2.0,
                negative_gain_weight=1.0,
            ),
            0.05,
        )
        self.assertAlmostEqual(
            _joint_gain(
                0.5,
                -0.2,
                quality_power=2.0,
                negative_gain_weight=1.0,
            ),
            -0.2,
        )

    def test_group_reward_uses_configured_point_six_point_four_weights(self) -> None:
        similarities = [-8.0, -7.0, -6.0, -5.0]
        joint_gains = [-0.1, 0.0, 0.03, 0.08]
        rewards, similarity_z, joint_z = _combine_group_rewards(
            similarities,
            joint_gains,
            similarity_weight=0.6,
            joint_weight=0.4,
            epsilon=1e-6,
        )
        self.assertAlmostEqual(sum(rewards), 0.0, places=6)
        self.assertEqual(rewards[-1], max(rewards))
        for reward, sim_value, joint_value in zip(rewards, similarity_z, joint_z):
            self.assertTrue(math.isfinite(reward))
            self.assertAlmostEqual(reward, 0.6 * sim_value + 0.4 * joint_value)

    def test_constant_joint_component_contributes_zero(self) -> None:
        rewards, similarity_z, joint_z = _combine_group_rewards(
            [-8.0, -7.0, -6.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],
            similarity_weight=0.6,
            joint_weight=0.4,
            epsilon=1e-6,
        )
        self.assertEqual(joint_z, [0.0, 0.0, 0.0, 0.0])
        for reward, sim_value in zip(rewards, similarity_z):
            self.assertAlmostEqual(reward, 0.6 * sim_value)


if __name__ == "__main__":
    unittest.main()
