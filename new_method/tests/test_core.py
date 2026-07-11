from __future__ import annotations

import math
import unittest

from new_method.core import (
    append_think,
    classify_gain,
    delta_log_rank,
    extract_think,
    history_from_row,
    ndcg_at_rank,
    rank_bucket,
)


class CoreTest(unittest.TestCase):
    def test_extract_think_ignores_answer(self) -> None:
        text = "<think>supported preference</think><answer>target-like tail</answer>"
        think, tagged = extract_think(text)
        self.assertTrue(tagged)
        self.assertEqual(think, "supported preference")
        self.assertNotIn("target-like", append_think("history", think))

    def test_rank_metrics(self) -> None:
        gain = delta_log_rank(100, 20)
        self.assertTrue(math.isclose(gain, math.log(101) - math.log(21)))
        self.assertEqual(ndcg_at_rank(1, 20), 1.0)
        self.assertEqual(ndcg_at_rank(21, 20), 0.0)
        self.assertEqual(rank_bucket(20), "1-20")
        self.assertEqual(rank_bucket(21), "21-100")
        self.assertEqual(rank_bucket(1001), "1000+")

    def test_base_query_precedes_augmented_query(self) -> None:
        row = {
            "base_query": "history only",
            "query": "history only\nRecommendation reasoning:\nold cot",
        }
        self.assertEqual(history_from_row(row), "history only")

    def test_classify_gain_requires_rank_and_margin_agreement(self) -> None:
        base = {
            "baseline_rank": 100,
            "cot_rank": 40,
            "delta_log_rank": 0.5,
            "delta_margin": 0.2,
            "format_ok": True,
            "history_truncated_tokens": 0,
        }
        label, failures = classify_gain(
            base,
            min_good_log_rank=0.4,
            min_good_margin=0.0,
            min_bad_log_rank=-0.4,
            min_bad_margin=0.0,
        )
        self.assertEqual(label, "good")
        self.assertEqual(failures, [])

        inconsistent = {**base, "delta_margin": -0.1}
        label, _ = classify_gain(
            inconsistent,
            min_good_log_rank=0.4,
            min_good_margin=0.0,
            min_bad_log_rank=-0.4,
            min_bad_margin=0.0,
        )
        self.assertEqual(label, "neutral")

    def test_classify_gain_rejects_truncation(self) -> None:
        row = {
            "baseline_rank": 100,
            "cot_rank": 10,
            "delta_log_rank": 1.0,
            "delta_margin": 1.0,
            "cot_think": "reason [TRUNCATED]",
        }
        label, failures = classify_gain(
            row,
            min_good_log_rank=0.4,
            min_good_margin=0.0,
            min_bad_log_rank=-0.4,
            min_bad_margin=0.0,
        )
        self.assertEqual(label, "rejected")
        self.assertIn("cot_think_truncated_marker", failures)


if __name__ == "__main__":
    unittest.main()
