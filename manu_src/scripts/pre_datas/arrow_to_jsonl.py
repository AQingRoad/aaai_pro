#!/usr/bin/env python3
"""把 Hugging Face DatasetDict 中的数据 split 和 item_info 转为 JSONL。"""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


# 源数据使用 valid，输出文件按用户习惯命名为 val.jsonl；item_info 原名保留。
SPLIT_FILES = {
    "train": "train.jsonl",
    "valid": "val.jsonl",
    "test": "test.jsonl",
    "item_info": "item_info.jsonl",
}


def convert(input_dir: Path, output_dir: Path) -> None:
    """逐个 split 写出 JSONL，不修改字段、样本顺序或字段值。"""
    # load_from_disk 直接读取包含 dataset_dict.json 的 Arrow 数据目录。
    dataset = load_from_disk(str(input_dir))

    # 三个 split 缺少任何一个都停止，避免生成不完整的数据目录。
    missing = [split for split in SPLIT_FILES if split not in dataset]
    if missing:
        raise ValueError(f"输入数据缺少 split: {', '.join(missing)}")

    # 输出目录不存在时自动创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 Arrow 中的原始顺序写出每条样本。
    for split, filename in SPLIT_FILES.items():
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as file:
            for row in dataset[split]:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 输出行数和路径，便于转换后核对。
        print(json.dumps({"split": split, "rows": len(dataset[split]), "output": str(output_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 Arrow DatasetDict 转为 train、val、test 和 item_info JSONL。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="包含 dataset_dict.json 和各 split Arrow 文件的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="train.jsonl、val.jsonl、test.jsonl 和 item_info.jsonl 的输出目录",
    )
    args = parser.parse_args()

    # 在读取数据前检查路径，给出比 load_from_disk 更直接的错误信息。
    if not (args.input_dir / "dataset_dict.json").is_file():
        parser.error(f"输入目录缺少 dataset_dict.json: {args.input_dir}")

    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
