#!/usr/bin/env python3
"""从原始 JSONL 构建 title_store_categories 口径的训练、验证和测试数据。"""

import argparse
import json
import re
from pathlib import Path

from format_positive import clean, format_details, format_positive, limit_description, texts


# 输入文件名、输出文件名和写入样本的 split 标签。
SPLITS = {
    "train.jsonl": ("train.jsonl", "train"),
    "val.jsonl": ("val.jsonl", "valid"),
    "test.jsonl": ("test.jsonl", "test"),
}

# 当前构建过程不主动截断 query，出现该标记说明输入数据口径异常。
FORBIDDEN_QUERY_MARKERS = ("[TRUNCATED]",)
# 清理显式 Amazon ASIN 和裸露的 B0 开头十位 ASIN。
ASIN_RE = re.compile(r"(?:Amazon\s+)?ASIN\s*[:#]?\s*[A-Z0-9]{10}|\bB0[0-9A-Z]{8}\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL，并忽略空行。"""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_time_delta(history_timestamp: int, target_timestamp: int) -> str:
    """复现 RRec 的相对时间格式，表示历史交互距目标交互过去多久。"""
    # 原始时间戳单位为毫秒；按秒计算差值可以避免本地时区影响。
    delta_seconds = (int(target_timestamp) - int(history_timestamp)) / 1000
    if delta_seconds < 0:
        raise ValueError("history_timestamp 晚于目标 timestamp")

    days = int(delta_seconds // 86400)
    remainder = delta_seconds % 86400
    hours = int(remainder // 3600)
    minutes = (remainder % 3600) / 60

    # 与 external/RRec/prompters/rrec_prompter.py 的 format_timedelta 保持一致。
    if days > 0:
        hours_with_minutes = hours + minutes / 60
        return f"{days}d {hours_with_minutes:.1f}h ago"
    if hours > 0:
        return f"{hours}h {minutes:.1f}min ago"
    return f"{minutes:.1f}min ago"


def history_item_text(item: dict, fallback_title: str) -> str:
    """只保留历史物品的 Title、Store/artist/format 和 Categories。"""
    # 先分别清理 item_info 标题和 split 自带标题，避免纯空格被当成有效值。
    title = clean(item.get("title")) or clean(fallback_title)
    if not title:
        raise ValueError("历史物品缺少标题和回退标识")

    # 标题也添加字段标签，与 positive 的字段边界保持一致。
    parts = [f"Title: {title}"]
    if store := clean(item.get("store")):
        parts.append(f"Store/artist/format: {store}")
    # 保留 item_info 中全部类别层级，不设置最大层数。
    if categories := " > ".join(texts(item.get("categories"))):
        parts.append(f"Categories: {categories}")
    # 拼接该历史物品的全部 Description 段落，再限制为最多 256 个字符。
    if description := limit_description(item.get("description")):
        parts.append(f"Description: {description}")
    # 解析并拼接该历史物品的全部 Details 键值，再限制为最多 256 个字符。
    if details := format_details(item.get("details")):
        parts.append(f"Details: {details}")

    # 防止 item metadata 中的 ASIN 进入 history query。
    return clean(ASIN_RE.sub("", "; ".join(parts)))


def build_query(row: dict, items: dict[int, dict]) -> tuple[str, list[int]]:
    """按原始交互顺序构造用户历史文本。"""
    history_ids = [int(item_id) for item_id in row.get("history_item_id", [])]
    fallback_titles = row.get("history_item_title", [])
    history_timestamps = [int(timestamp) for timestamp in row.get("history_timestamp", [])]
    target_timestamp = int(row["timestamp"])
    if len(history_timestamps) != len(history_ids):
        raise ValueError("history_timestamp 与 history_item_id 数量不一致")
    lines = ["This user's Amazon CDs and Vinyl interaction history over time is listed below."]

    for position, item_id in enumerate(history_ids, 1):
        if item_id not in items:
            raise ValueError(f"history item_id={item_id} 不在 item_info 中")
        # 两处标题都为空时使用稳定的 item_id 标识，项目中只有 item_id=2134 命中该情况。
        fallback = fallback_titles[position - 1] if position <= len(fallback_titles) else ""
        fallback = clean(fallback) or f"item_{item_id}"
        # RRec 使用相对目标交互的时间差，让不同日期范围的样本共享同一时间尺度。
        time_delta = format_time_delta(history_timestamps[position - 1], target_timestamp)
        lines.append(f"{position}. Time: {time_delta}; {history_item_text(items[item_id], fallback)}")

    query = "\n".join(lines)
    # Description 正文可能自然包含评分或“Details”等单词，不能按全文关键词误判。
    if any(marker in query for marker in FORBIDDEN_QUERY_MARKERS) or ASIN_RE.search(query):
        raise ValueError("history query 含截断标记或 ASIN")
    return query, history_ids


def build_row(row: dict, items: dict[int, dict], split: str) -> dict:
    """构建一条 embedding 训练 pair。"""
    target_id = int(row["item_id"])
    if target_id not in items:
        raise ValueError(f"target item_id={target_id} 不在 item_info 中")

    query, history_ids = build_query(row, items)
    target_title = clean(items[target_id].get("title") or row.get("item_title"))
    history_timestamps = [int(timestamp) for timestamp in row.get("history_timestamp", [])]
    target_timestamp = int(row["timestamp"])
    # 组合 split、interaction_id 和 user_id，得到跨 split 可追踪的唯一样本 ID。
    example_id = f"CDs_and_Vinyl:{split}:{row.get('interaction_id', '')}:{row.get('user_id', '')}"

    return {
        "example_id": example_id,
        "query": query,
        "positive": format_positive(items[target_id], target_title),
        "category": "CDs_and_Vinyl",
        "split": split,
        "user_id": row.get("user_id", ""),
        "interaction_id": row.get("interaction_id", ""),
        "target_item_id": target_id,
        "target_item_title": target_title,
        "history_item_ids": history_ids,
        # 保留 RRec 原始字段名，便于与源 split 和参考实现直接对齐。
        "history_timestamp": history_timestamps,
        "timestamp": target_timestamp,
        "history_item_count": len(history_ids),
        "ablation_name": "time_title_store_categories_description_details",
        "query_fields": ["relative_time", "title", "store", "categories", "description", "details"],
    }


def build_split(input_path: Path, output_path: Path, items: dict[int, dict], split: str) -> None:
    """转换一个 split，并输出最小审计信息。"""
    source_rows = read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in source_rows:
            output = build_row(row, items, split)
            file.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(json.dumps({"split": split, "rows": len(source_rows), "output": str(output_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 title_store_categories 训练、验证和测试 JSONL。")
    parser.add_argument("--input-dir", type=Path, required=True, help="包含 train、val、test、item_info JSONL 的目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="处理后 train、val、test JSONL 的输出目录")
    args = parser.parse_args()

    item_info_path = args.input_dir / "item_info.jsonl"
    if not item_info_path.is_file():
        parser.error(f"缺少 item_info.jsonl: {item_info_path}")
    items = {int(item["item_id"]): item for item in read_jsonl(item_info_path)}

    for source_name, (output_name, split) in SPLITS.items():
        input_path = args.input_dir / source_name
        if not input_path.is_file():
            parser.error(f"缺少输入文件: {input_path}")
        build_split(input_path, args.output_dir / output_name, items, split)


if __name__ == "__main__":
    main()
