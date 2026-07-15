#!/usr/bin/env python3
"""测试 embedding 评测的 SFT 风格 query 压缩。"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "manu_src" / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "manu_src" / "scripts" / "pre_datas"))
sys.path.insert(0, str(ROOT / "manu_src" / "scripts" / "prompts"))

from build_sft_from_teacher_cot import remove_item_field  # noqa: E402
from fit_embedding_query import (  # noqa: E402
    fit_embedding_query,
    query_token_count,
)


class WhitespaceTokenizer:
    """提供压缩函数所需的最小 tokenizer 接口。"""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def __call__(self, text, **_) -> dict:
        if isinstance(text, list):
            return {"input_ids": [self._tokens(value) for value in text]}
        return {"input_ids": self._tokens(text)}

    def encode(self, text: str, **_) -> list[str]:
        return self._tokens(text)

    @staticmethod
    def decode(tokens: list[str], **_) -> str:
        return " ".join(tokens)


def format_query(query: str) -> str:
    return f"Instruct: retrieve next item\nQuery: {query}"


class FitEmbeddingQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def test_query_within_budget_is_unchanged(self) -> None:
        query = "History below.\n1. Title: A; Categories: Rock"
        fitted, audit = fit_embedding_query(
            self.tokenizer, query, 100, format_query
        )
        self.assertEqual(fitted, query)
        self.assertFalse(audit["compression_applied"])

    def test_remove_details_before_description(self) -> None:
        item_one = (
            "1. Title: A; Description: useful description remains; "
            "Details: one two three four five six seven eight"
        )
        item_two = "2. Title: B; Categories: Rock"
        query = f"History below.\n{item_one}\n{item_two}"
        expected = f"History below.\n{remove_item_field(item_one, 'details')}\n{item_two}"
        max_length = query_token_count(self.tokenizer, expected, format_query)

        fitted, audit = fit_embedding_query(
            self.tokenizer, query, max_length, format_query
        )
        self.assertEqual(fitted, expected)
        self.assertEqual(audit["details_removed_item_numbers"], [1])
        self.assertEqual(audit["description_removed_item_numbers"], [])
        self.assertEqual(audit["removed_history_item_numbers"], [])

    def test_preserve_referenced_item_and_complete_reasoning_suffix(self) -> None:
        item_one = "1. Title: Keep Me; Categories: Rock"
        item_two = (
            "2. Title: Remove Me; Description: a b c d e f; "
            "Details: g h i j k l"
        )
        suffix = (
            "\n\nRecommendation reasoning:\n"
            "<think>Item 1 is the retained evidence.</think>\n"
            "<answer>Rock album</answer>"
        )
        query = f"History below.\n{item_one}\n{item_two}{suffix}"
        expected = f"History below.\n{item_one}{suffix}"
        max_length = query_token_count(self.tokenizer, expected, format_query)

        fitted, audit = fit_embedding_query(
            self.tokenizer, query, max_length, format_query
        )
        self.assertEqual(fitted, expected)
        self.assertTrue(fitted.endswith(suffix))
        self.assertEqual(audit["protected_cot_item_numbers"], [1])
        self.assertEqual(audit["removed_history_item_numbers"], [2])
        self.assertTrue(audit["reasoning_suffix_preserved"])

    def test_shorten_only_remaining_item_tail(self) -> None:
        words = " ".join(f"word{index}" for index in range(40))
        suffix = "\n\nRecommendation reasoning:\n<answer>Rock</answer>"
        query = f"History below.\n1. Title: Long Item; Notes: {words}{suffix}"

        fitted, audit = fit_embedding_query(
            self.tokenizer, query, 24, format_query
        )
        self.assertLessEqual(
            query_token_count(self.tokenizer, fitted, format_query), 24
        )
        self.assertGreater(audit["oldest_retained_item_tail_tokens_removed"], 0)
        self.assertTrue(fitted.endswith(suffix))


if __name__ == "__main__":
    unittest.main()
