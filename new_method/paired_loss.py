from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PairedLossOutput:
    loss: torch.Tensor
    loss_history: torch.Tensor
    loss_good_retrieval: torch.Tensor
    loss_good_order: torch.Tensor
    loss_bad_order: torch.Tensor
    history_accuracy: torch.Tensor
    good_accuracy: torch.Tensor
    good_order_accuracy: torch.Tensor
    bad_order_accuracy: torch.Tensor
    history_margin_mean: torch.Tensor
    good_margin_mean: torch.Tensor
    bad_margin_mean: torch.Tensor


def positive_mask(
    query_target_ids: torch.Tensor,
    document_target_ids: torch.Tensor,
) -> torch.Tensor:
    mask = query_target_ids[:, None].eq(document_target_ids[None, :])
    mask &= document_target_ids[None, :].ge(0)
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("Every query must have at least one positive document")
    return mask


def multi_positive_info_nce(
    logits: torch.Tensor,
    query_target_ids: torch.Tensor,
    document_target_ids: torch.Tensor,
) -> torch.Tensor:
    mask = positive_mask(query_target_ids, document_target_ids)
    positive_logits = logits.masked_fill(~mask, -torch.inf)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def retrieval_margin(
    logits: torch.Tensor,
    query_target_ids: torch.Tensor,
    document_target_ids: torch.Tensor,
) -> torch.Tensor:
    pos_mask = positive_mask(query_target_ids, document_target_ids)
    neg_mask = ~pos_mask
    if not bool(neg_mask.any(dim=1).all()):
        raise RuntimeError("Every query must have at least one negative document")
    positive_score = torch.logsumexp(logits.masked_fill(~pos_mask, -torch.inf), dim=1)
    negative_score = torch.logsumexp(logits.masked_fill(~neg_mask, -torch.inf), dim=1)
    return positive_score - negative_score


def retrieval_accuracy(
    logits: torch.Tensor,
    query_target_ids: torch.Tensor,
    document_target_ids: torch.Tensor,
) -> torch.Tensor:
    predicted_ids = document_target_ids[logits.argmax(dim=1)]
    return predicted_ids.eq(query_target_ids).float().mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def paired_cot_loss(
    *,
    history_logits: torch.Tensor,
    query_target_ids: torch.Tensor,
    document_target_ids: torch.Tensor,
    good_logits: torch.Tensor | None = None,
    good_mask: torch.Tensor | None = None,
    bad_logits: torch.Tensor | None = None,
    bad_mask: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.2,
    gamma: float = 0.2,
    good_order_margin: float = 0.0,
    bad_order_margin: float = 0.0,
) -> PairedLossOutput:
    device = history_logits.device
    row_count = history_logits.shape[0]
    if query_target_ids.shape != (row_count,):
        raise ValueError("query_target_ids must match history_logits rows")

    history_loss = multi_positive_info_nce(history_logits, query_target_ids, document_target_ids)
    history_margin = retrieval_margin(history_logits, query_target_ids, document_target_ids)
    history_acc = retrieval_accuracy(history_logits, query_target_ids, document_target_ids)

    zero = history_loss * 0.0
    good_loss = zero
    good_order_loss = zero
    bad_order_loss = zero
    good_acc = zero
    good_order_acc = zero
    bad_order_acc = zero
    good_margin_full = torch.full((row_count,), torch.nan, device=device)
    bad_margin_full = torch.full((row_count,), torch.nan, device=device)

    if good_mask is None:
        good_mask = torch.zeros(row_count, dtype=torch.bool, device=device)
    else:
        good_mask = good_mask.to(device=device, dtype=torch.bool)
    if bad_mask is None:
        bad_mask = torch.zeros(row_count, dtype=torch.bool, device=device)
    else:
        bad_mask = bad_mask.to(device=device, dtype=torch.bool)

    if bool(good_mask.any()):
        if good_logits is None or good_logits.shape[0] != int(good_mask.sum().item()):
            raise ValueError("good_logits rows must equal good_mask true count")
        good_ids = query_target_ids[good_mask]
        good_loss = multi_positive_info_nce(good_logits, good_ids, document_target_ids)
        good_margin = retrieval_margin(good_logits, good_ids, document_target_ids)
        good_margin_full[good_mask] = good_margin
        good_order_loss = F.relu(
            good_order_margin - (good_margin - history_margin[good_mask])
        ).mean()
        good_acc = retrieval_accuracy(good_logits, good_ids, document_target_ids)
        good_order_acc = (good_margin > history_margin[good_mask]).float().mean()

    if bool(bad_mask.any()):
        if bad_logits is None or bad_logits.shape[0] != int(bad_mask.sum().item()):
            raise ValueError("bad_logits rows must equal bad_mask true count")
        bad_ids = query_target_ids[bad_mask]
        bad_margin = retrieval_margin(bad_logits, bad_ids, document_target_ids)
        bad_margin_full[bad_mask] = bad_margin
        bad_order_loss = F.relu(
            bad_order_margin - (history_margin[bad_mask] - bad_margin)
        ).mean()
        bad_order_acc = (history_margin[bad_mask] > bad_margin).float().mean()

    total = history_loss + alpha * good_loss + beta * good_order_loss + gamma * bad_order_loss
    return PairedLossOutput(
        loss=total,
        loss_history=history_loss,
        loss_good_retrieval=good_loss,
        loss_good_order=good_order_loss,
        loss_bad_order=bad_order_loss,
        history_accuracy=history_acc,
        good_accuracy=good_acc,
        good_order_accuracy=good_order_acc,
        bad_order_accuracy=bad_order_acc,
        history_margin_mean=history_margin.mean(),
        good_margin_mean=_masked_mean(torch.nan_to_num(good_margin_full), good_mask),
        bad_margin_mean=_masked_mean(torch.nan_to_num(bad_margin_full), bad_mask),
    )
