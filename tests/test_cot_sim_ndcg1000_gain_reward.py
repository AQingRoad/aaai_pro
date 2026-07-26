from __future__ import annotations

import math
import unittest

from manu_src.scripts.train.cot_sim_ndcg1000_gain_reward import (
    _combine_group_rewards,
    _resolve_cached_references,
)


class CotSimNdcg1000GainRewardTest(unittest.TestCase):
    def test_reward_uses_similarity_and_new_minus_reference_gain(self) -> None:
        similarities = [-8.0, -7.0, -6.0, -5.0]
        gains = [-0.10, 0.00, 0.03, 0.08]
        rewards, similarity_z, gain_z = _combine_group_rewards(
            similarities,
            gains,
            similarity_weight=0.6,
            gain_weight=0.4,
            epsilon=1e-6,
        )
        self.assertAlmostEqual(sum(rewards), 0.0, places=6)
        self.assertEqual(rewards[-1], max(rewards))
        for reward, sim_value, gain_value in zip(rewards, similarity_z, gain_z):
            self.assertTrue(math.isfinite(reward))
            self.assertAlmostEqual(reward, 0.6 * sim_value + 0.4 * gain_value)

    def test_constant_gain_component_contributes_zero(self) -> None:
        rewards, similarity_z, gain_z = _combine_group_rewards(
            [-8.0, -7.0, -6.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],
            similarity_weight=0.6,
            gain_weight=0.4,
            epsilon=1e-6,
        )
        self.assertEqual(gain_z, [0.0, 0.0, 0.0, 0.0])
        for reward, sim_value in zip(rewards, similarity_z):
            self.assertAlmostEqual(reward, 0.6 * sim_value)

    def test_cached_reference_is_broadcast_inside_each_group(self) -> None:
        ndcgs, ranks, sources = _resolve_cached_references(
            grouped_indices={
                ("example_id", "a"): [0, 1, 2, 3],
                ("example_id", "b"): [4, 5, 6, 7],
            },
            reference_ndcg_values=[0.5] * 4 + [None] * 4,
            reference_rank_values=[1] * 4 + [10] * 4,
            ndcg_k=1000,
        )
        self.assertEqual(ndcgs[:4], [0.5] * 4)
        self.assertEqual(ranks[:4], [1] * 4)
        self.assertEqual(ranks[4:], [10] * 4)
        self.assertAlmostEqual(ndcgs[4], 1.0 / math.log2(11.0))
        self.assertEqual(sources, {"cached_metadata": 1, "cached_rank": 1})

    def test_reference_metadata_must_be_consistent_inside_group(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "differs inside"):
            _resolve_cached_references(
                grouped_indices={("example_id", "a"): [0, 1, 2, 3]},
                reference_ndcg_values=[0.1, 0.1, 0.2, 0.1],
                reference_rank_values=[10, 10, 10, 10],
                ndcg_k=1000,
            )


if __name__ == "__main__":
    unittest.main()
