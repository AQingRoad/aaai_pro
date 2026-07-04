#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RREC_ROOT = ROOT / "external/RRec"
QWEN25_3B_PATH = "/home/user/models_hf/Qwen2.5-3B-Instruct"

sys.path.insert(0, str(RREC_ROOT))

from prompters import rrec_prompter  # noqa: E402


def get_item_by_id(item_dset: Any, item_id: int) -> dict[str, Any]:
    if 0 <= item_id < len(item_dset) and int(item_dset[item_id]["item_id"]) == item_id:
        return item_dset[item_id]
    one_based_index = item_id - 1
    if 0 <= one_based_index < len(item_dset) and int(item_dset[one_based_index]["item_id"]) == item_id:
        return item_dset[one_based_index]
    raise IndexError(f"Cannot find item_id={item_id} in item_info")


def get_item_info_str_zero_aware(sequence: dict[str, Any], item_dset: Any, just_title: bool = False) -> str:
    if just_title:
        return sequence["item_title"] if "item_title" in sequence else sequence["title"]
    if "title" in sequence:
        item_title = sequence["title"]
        item_info = sequence
    else:
        item_id = int(sequence["seq_labels"])
        item_title = sequence["item_title"]
        item_info = get_item_by_id(item_dset, item_id)
        assert item_title == item_info["title"], f"item_title: {item_title}, item_info title: {item_info['title']}"

    description = item_info["description"]
    description = "" if len(description) == 0 else " ".join(description[::-1])
    words = description.split()
    if len(words) > rrec_prompter.DESCRIPTION_MAX_LEN:
        description = " ".join(words[: rrec_prompter.DESCRIPTION_MAX_LEN]) + "..."

    return (
        f"Title: {item_title}\n"
        f"User Rating: {item_info['average_rating']}\n"
        f"Number of Buyers: {item_info['rating_number']}\n"
        f"Description: {description}"
    )


def main_train(**kwargs: Any) -> None:
    import train as rrec_train

    rrec_prompter.get_item_info_str = get_item_info_str_zero_aware
    rrec_train.model_names["Qwen2.5-3B-Instruct"] = QWEN25_3B_PATH
    rrec_train.train(**kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main_train)
