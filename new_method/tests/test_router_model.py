import unittest

import torch

from new_method.router_model import route_metrics, select_threshold, user_group_split


class RouterModelTest(unittest.TestCase):
    def test_user_group_split_keeps_users_together(self):
        users = ["a", "a", "b", "b", "c", "d", "e", "f"]
        train, valid = user_group_split(users, valid_fraction=0.5, seed=42)
        for user in set(users):
            indices = [index for index, value in enumerate(users) if value == user]
            assignments = {bool(valid[index]) for index in indices}
            self.assertEqual(len(assignments), 1)
        self.assertFalse(bool((train & valid).any()))

    def test_threshold_respects_budget_and_prefers_positive_gain(self):
        probabilities = torch.tensor([0.9, 0.8, 0.7, 0.6])
        gains = torch.tensor([0.2, 0.1, -0.5, 1.0])
        selected = select_threshold(probabilities, gains, max_trigger_rate=0.5)
        route = probabilities > float(selected["threshold"])
        self.assertEqual(int(route.sum()), 2)
        self.assertAlmostEqual(float(gains[route].sum()), 0.3, places=6)

    def test_route_metrics_uses_frozen_threshold(self):
        metrics = route_metrics(
            torch.tensor([0.9, 0.2]),
            threshold=0.5,
            labels=torch.tensor([1.0, 0.0]),
            baseline_ndcg=torch.tensor([0.0, 1.0]),
            cot_ndcg=torch.tensor([0.5, 0.0]),
            baseline_rank=torch.tensor([30, 1]),
            cot_rank=torch.tensor([3, 30]),
        )
        self.assertAlmostEqual(metrics["routed_ndcg"], 0.75)
        self.assertAlmostEqual(metrics["oracle_recovery"], 1.0)
        self.assertEqual(metrics["trigger_count"], 1)


if __name__ == "__main__":
    unittest.main()
