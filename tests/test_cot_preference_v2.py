from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rubric_cot_pipeline.preference_scorer_v2 import build_preference_features


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


JUDGE = load_module("judge_cot_preference_groups_v2", ROOT / "scripts/cot/judge_cot_preference_groups_v2.py")


class PreferenceV2Test(unittest.TestCase):
    def test_build_groups_aligns_rollouts_and_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            completions = directory / "completions.jsonl"
            components = directory / "components.jsonl"
            source = directory / "source.jsonl"
            output = directory / "groups.jsonl"
            example_id = "CDs_and_Vinyl:train:1:user"
            completion_texts = [
                f"<think>Evidence {index}</think><answer>Profile {index}</answer>" for index in range(4)
            ]
            completions.write_text(
                json.dumps(
                    {
                        "step": ["1"] * 4,
                        "completion": completion_texts,
                        "advantages": [0.3, 0.1, -0.1, -0.3],
                        "ReferenceSoftNdcgReward": [0.4, 0.3, 0.2, 0.1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            components.write_text(
                json.dumps(
                    {
                        "call_index": 1,
                        "items": [
                            {
                                "idx": index,
                                "example_id": example_id,
                                "new_rank": 10 + index,
                                "reference_rank": 20,
                                "q_new": 0.4 - index * 0.1,
                                "q_ref": 0.2,
                            }
                            for index in range(4)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source.write_text(
                json.dumps(
                    {
                        "example_id": example_id,
                        "split": "train",
                        "category": "CDs_and_Vinyl",
                        "user_history": "1. Example; Store: Artist; Categories: Rock",
                        "history_item_ids": [1],
                        "history_item_count": 1,
                        "target_item_id": 2,
                        "baseline_rank": 20,
                        "cot_rank": 8,
                        "candidate_id": f"{example_id}-0",
                        "cot": "<think>Grounded evidence</think><answer>Rock profile</answer>",
                        "judge_used_target": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/datasets/build_cot_preference_groups_v2.py"),
                    "--completions",
                    str(completions),
                    "--components",
                    str(components),
                    "--source-scored",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["example_id"], example_id)
            self.assertEqual(len(row["candidates"]), 6)
            self.assertEqual({candidate["source"] for candidate in row["candidates"]}, {
                "policy_rollout",
                "external_glm",
                "synthetic_corruption",
            })
            self.assertEqual(row["fold"], 0)

    def test_listwise_judgment_requires_complete_ranking(self) -> None:
        mapping = {"C1": "a", "C2": "b"}
        valid = {
            "ranking": ["C2", "C1"],
            "scores": {
                token: {
                    "history_grounding": 4,
                    "preference_specificity": 4,
                    "transition_reasoning": 4,
                    "discriminative_constraints": 4,
                    "factual_support": 4,
                    "conciseness": 4,
                    "format_valid": True,
                }
                for token in mapping
            },
        }
        normalized = JUDGE.normalized_judgment(valid, mapping)
        self.assertEqual(normalized["ranking"], ["b", "a"])
        invalid = dict(valid)
        invalid["ranking"] = ["C1"]
        with self.assertRaises(ValueError):
            JUDGE.normalized_judgment(invalid, mapping)

    def test_feature_shape_and_values(self) -> None:
        history = torch.tensor([[1.0, 2.0]])
        joint = torch.tensor([[3.0, 5.0]])
        features = build_preference_features(history, joint)
        self.assertEqual(tuple(features.shape), (1, 6))
        self.assertTrue(torch.equal(features, torch.tensor([[3.0, 5.0, 2.0, 3.0, 3.0, 10.0]])))


if __name__ == "__main__":
    unittest.main()
