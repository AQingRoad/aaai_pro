from __future__ import annotations

import unittest

from new_method.make_cot_perturbations import variant_texts


class PerturbationTest(unittest.TestCase):
    def test_variants_do_not_need_target_fields(self) -> None:
        row = {
            "base_query": "history",
            "cot": "<think>First reason. Second reason.</think><answer>Preference summary.</answer>",
        }
        variants = dict(
            variant_texts(
                row,
                (
                    "original",
                    "think_plus_answer",
                    "answer_only",
                    "reverse_sentences",
                    "repeat_tail",
                ),
            )
        )
        self.assertEqual(variants["original"], "First reason. Second reason.")
        self.assertIn("Preference summary.", variants["think_plus_answer"])
        self.assertEqual(variants["answer_only"], "Preference summary.")
        self.assertTrue(variants["reverse_sentences"].startswith("Second reason."))
        self.assertGreater(variants["repeat_tail"].count("Second reason."), 1)


if __name__ == "__main__":
    unittest.main()
