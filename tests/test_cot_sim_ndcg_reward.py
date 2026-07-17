from __future__ import annotations

import math
import unittest

from manu_src.scripts.train.cot_sim_ndcg_reward import (
    _combine_group_rewards,
    _group_zscore,
    _ndcg_at_rank,
    _pairwise_conflict_rate,
)


class CotSimNdcgRewardTest(unittest.TestCase):
    def test_ndcg_at_rank_uses_hard_k_cutoff(self) -> None:
        self.assertAlmostEqual(_ndcg_at_rank(1, 100), 1.0)
        self.assertAlmostEqual(_ndcg_at_rank(100, 100), 1.0 / math.log2(101.0))
        self.assertEqual(_ndcg_at_rank(101, 100), 0.0)

    def test_constant_group_component_contributes_zero(self) -> None:
        self.assertEqual(
            _group_zscore([-3.0, -3.0, -3.0, -3.0]),
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_combined_reward_uses_only_similarity_and_ndcg(self) -> None:
        similarities = [-8.0, -7.0, -6.0, -5.0]
        ndcgs = [0.0, 0.0, 0.0, 0.5]
        rewards, similarity_z, ndcg_z = _combine_group_rewards(
            similarities,
            ndcgs,
            similarity_weight=0.8,
            ndcg_weight=0.2,
        )
        self.assertEqual(len(rewards), 4)
        self.assertEqual(len(similarity_z), 4)
        self.assertEqual(len(ndcg_z), 4)
        self.assertAlmostEqual(sum(rewards), 0.0, places=6)
        self.assertEqual(rewards[-1], max(rewards))
        for reward, sim_value, ndcg_value in zip(rewards, similarity_z, ndcg_z):
            self.assertAlmostEqual(reward, 0.8 * sim_value + 0.2 * ndcg_value)

    def test_pairwise_conflict_rate_ignores_ties(self) -> None:
        self.assertEqual(_pairwise_conflict_rate([1.0, 2.0], [0.2, 0.1]), 1.0)
        self.assertEqual(_pairwise_conflict_rate([1.0, 2.0], [0.1, 0.2]), 0.0)
        self.assertEqual(_pairwise_conflict_rate([1.0, 2.0], [0.0, 0.0]), 0.0)

if __name__ == "__main__":
    unittest.main()
