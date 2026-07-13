#!/usr/bin/env python3
"""给 JSONL 的 positive 添加清晰字段标签，query、history 和 CoT 保持原样。"""

# argparse 用于读取命令行中的输入、item_info 和输出路径。
import argparse
# json 用于解析和写出 JSONL 中的每一行。
import json
# re 用于把连续空白字符压缩成一个空格。
import re
# Path 统一处理文件路径和输出目录。
from pathlib import Path


# query 和 positive 中每个 item 的 Description 最多保留 256 个字符。
DESCRIPTION_MAX_CHARS = 256
# query 和 positive 中每个 item 的 Details 最多保留 256 个字符。
DETAILS_MAX_CHARS = 256


# 清理单个 metadata 字段，避免换行和连续空格破坏 positive 格式。
def clean(value) -> str:
    # 空值先转为空字符串，再把所有连续空白替换成一个空格。
    return re.sub(r"\s+", " ", str(value or "")).strip()


# 把字符串或列表字段统一转换为清理后的字符串列表；limit<=0 表示全部保留。
def texts(value, limit: int = 0) -> list[str]:
    # 单个字符串包装成列表，原本为列表时直接使用。
    values = value if isinstance(value, list) else [value]
    # limit 为正数时限制数量，否则使用全部原始值。
    selected = values[:limit] if limit > 0 else values
    # 清理每个值，并删除空字符串。
    return [text for item in selected if (text := clean(item))]


# 对清理后的 Description 做统一字符级限制。
def limit_description(value, max_chars: int = DESCRIPTION_MAX_CHARS) -> str:
    # 先把全部 Description 段落按原顺序拼接成一个字符串。
    description = " ".join(texts(value))
    # rstrip 删除截断点前可能残留的空格，最终长度不会超过 max_chars。
    return description[:max_chars].rstrip()


def format_details(value, max_chars: int = DETAILS_MAX_CHARS) -> str:
    """解析 details，按原始顺序拼接全部键值，再限制字符长度。"""
    # 当前 item_info 将 details 保存为 JSON 字符串，先恢复为字典。
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # 非 JSON 字符串仍保留清理后的原文，避免静默丢失信息。
            return clean(value)[:max_chars].rstrip()

    # 标准 details 使用字典；key=value 可以保留字段名称和对应内容。
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            key_text = clean(key)
            value_text = clean(item)
            if key_text and value_text:
                parts.append(f"{key_text}={value_text}")
        return "; ".join(parts)[:max_chars].rstrip()

    # 兼容少量可能使用列表保存 details 的数据。
    if isinstance(value, list):
        return "; ".join(texts(value))[:max_chars].rstrip()
    return clean(value)[:max_chars].rstrip()


# 根据 item_info 中的目标物品 metadata 构造带标签的 positive。
def format_positive(item: dict, fallback_title: str = "") -> str:
    """保留完整目标信息，只增加字段名，不做字符截断。"""
    # 优先读取 item_info 标题，缺失时使用样本中的 target_item_title。
    title = clean(item.get("title") or fallback_title)
    # 标题是 positive 的必需字段，缺失时立即停止，避免写出错误数据。
    if not title:
        raise ValueError("目标物品缺少标题")

    # Title 始终放在 positive 的第一部分，明确目标物品名称。
    parts = [f"Title: {title}"]
    # main_category 存在时保留，并添加清晰字段名。
    if main_category := clean(item.get("main_category")):
        parts.append(f"Main category: {main_category}")
    # store 通常包含艺术家、创作者或载体格式信息。
    if store := clean(item.get("store")):
        parts.append(f"Store/artist/format: {store}")
    # 类别路径最多读取 6 层，并使用 > 保留层级关系。
    if categories := " > ".join(texts(item.get("categories"), 6)):
        parts.append(f"Categories: {categories}")
    # 商品特征最多保留 8 项，内容不做字符截断。
    if features := "; ".join(texts(item.get("features"), 8)):
        parts.append(f"Features: {features}")
    # 拼接全部 Description 段落，再统一限制为最多 256 个字符。
    if description := limit_description(item.get("description")):
        parts.append(f"Description: {description}")
    # Details 中的全部键值先按原始顺序拼接，再限制为最多 256 个字符。
    if details := format_details(item.get("details")):
        parts.append(f"Details: {details}")
    # 使用分号分隔字段，让 Title、类别、格式和描述的边界清晰可见。
    return "; ".join(parts)


# 一次性读取 item_info，并建立 item_id 到 metadata 的索引。
def read_item_info(path: Path) -> dict[int, dict]:
    # 使用 UTF-8 打开 JSONL，保证非英文字符正常读取。
    with path.open(encoding="utf-8") as file:
        # 解析每个非空行，并用整数 item_id 作为字典键。
        return {
            int(item["item_id"]): item
            for line in file
            if line.strip() and (item := json.loads(line)).get("item_id") is not None
        }


# 逐行处理输入 JSONL，只替换 positive 字段。
def process(input_path: Path, item_info_path: Path, output_path: Path) -> None:
    # 加载目标物品索引，后续按 target_item_id 查询 metadata。
    items = read_item_info(item_info_path)
    # 输出目录不存在时自动创建，parents=True 同时创建上级目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 记录实际写出的非空样本数。
    count = 0

    # 同时打开输入和输出文件，输出文件统一使用 UTF-8。
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        # 从 1 开始记录行号，报错时能定位原始样本。
        for line_number, line in enumerate(source, 1):
            # 跳过输入文件中的空行。
            if not line.strip():
                continue
            # 将当前 JSONL 行解析为字典。
            row = json.loads(line)
            # target_item_id 强制转为整数，与 item_info 索引保持一致。
            item_id = int(row["target_item_id"])
            # item_info 缺少目标物品时停止，避免沿用旧 positive。
            if item_id not in items:
                raise ValueError(f"第 {line_number} 行的 target_item_id={item_id} 不在 item_info 中")
            # 重建 positive；除 positive 外，row 中所有字段保持不变。
            row["positive"] = format_positive(items[item_id], row.get("target_item_title", ""))
            # ensure_ascii=False 让中文和其它非 ASCII 字符直接写入文件。
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            # 当前样本成功写出后更新计数。
            count += 1

    # 在终端输出处理行数和结果路径，便于核对任务是否完整。
    print(json.dumps({"rows": count, "output": str(output_path)}, ensure_ascii=False))


# 定义命令行入口和三个必需路径参数。
def main() -> None:
    # 创建参数解析器，并说明脚本只重建 positive。
    parser = argparse.ArgumentParser(description="重建带字段标签的 positive，其他字段保持不变。")
    # --input 指向需要处理的 history-only 或 history+CoT JSONL。
    parser.add_argument("--input", type=Path, required=True, help="待处理 JSONL")
    # --item-info 提供 target_item_id 对应的完整物品 metadata。
    parser.add_argument("--item-info", type=Path, required=True, help="包含目标物品 metadata 的 JSONL")
    # --output 指向新文件，禁止覆盖输入文件以保护原始数据。
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL，请勿与输入路径相同")
    # 解析用户传入的命令行参数。
    args = parser.parse_args()

    # 输入输出路径相同时直接报错，避免原文件在读取前被清空。
    if args.input.resolve() == args.output.resolve():
        parser.error("--output 不能覆盖 --input")
    # 参数检查通过后开始逐行处理。
    process(args.input, args.item_info, args.output)


# 只有直接运行当前脚本时才进入 main，作为模块导入时不会自动处理数据。
if __name__ == "__main__":
    main()
