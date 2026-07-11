from __future__ import annotations

import torch
import unittest

from new_method.paired_loss import paired_cot_loss


class PairedLossTest(unittest.TestCase):
    def test_paired_loss_orders_good_history_bad(self) -> None:
        history_logits = torch.tensor(
            [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]],
            requires_grad=True,
        )
        good_logits = torch.tensor([[5.0, 0.0, -1.0]], requires_grad=True)
        bad_logits = torch.tensor([[0.0, 3.0, -1.0]], requires_grad=True)
        query_ids = torch.tensor([10, 20])
        document_ids = torch.tensor([10, 20, -1])
        mask = torch.tensor([True, False])

        output = paired_cot_loss(
            history_logits=history_logits,
            query_target_ids=query_ids,
            document_target_ids=document_ids,
            good_logits=good_logits,
            good_mask=mask,
            bad_logits=bad_logits,
            bad_mask=mask,
            alpha=1.0,
            beta=0.2,
            gamma=0.2,
        )
        self.assertTrue(bool(torch.isfinite(output.loss)))
        self.assertEqual(output.good_order_accuracy.item(), 1.0)
        self.assertEqual(output.bad_order_accuracy.item(), 1.0)
        output.loss.backward()
        self.assertIsNotNone(history_logits.grad)
        self.assertIsNotNone(good_logits.grad)
        self.assertIsNotNone(bad_logits.grad)

    def test_history_only_rows_are_supported(self) -> None:
        history_logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]],
            requires_grad=True,
        )
        query_ids = torch.tensor([1, 2])
        document_ids = torch.tensor([1, 2])
        empty_mask = torch.tensor([False, False])
        output = paired_cot_loss(
            history_logits=history_logits,
            query_target_ids=query_ids,
            document_target_ids=document_ids,
            good_mask=empty_mask,
            bad_mask=empty_mask,
        )
        self.assertEqual(output.loss_good_retrieval.item(), 0.0)
        self.assertEqual(output.loss_good_order.item(), 0.0)
        self.assertEqual(output.loss_bad_order.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
