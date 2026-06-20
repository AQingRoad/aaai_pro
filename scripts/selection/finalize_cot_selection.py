#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rubric_cot_pipeline.io import ensure_parent, read_jsonl, write_jsonl


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    weight = pos - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def summarize(rows: list[dict[str, Any]], scores: list[float], kept: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    negative = sum(1 for row in rows if (as_float(row.get(args.gain_field)) or 0.0) < args.min_gain)
    summary = {
        "input": args.input,
        "output": args.output,
        "plot_output": args.plot_output,
        "score_field": args.score_field,
        "gain_field": args.gain_field,
        "min_gain": args.min_gain,
        "total_rows": len(rows),
        "valid_score_rows": len(scores),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "dropped_by_negative_gain": negative,
    }
    if scores:
        summary.update(
            {
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": statistics.fmean(scores),
                "score_median": statistics.median(scores),
                "score_p05": quantile(scores, 0.05),
                "score_p25": quantile(scores, 0.25),
                "score_p75": quantile(scores, 0.75),
                "score_p95": quantile(scores, 0.95),
            }
        )
    return summary


def histogram(values: list[float], bins: int) -> tuple[list[int], float, float]:
    if not values:
        return [], 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        delta = max(abs(lo) * 0.1, 1e-6)
        lo -= delta
        hi += delta
    counts = [0 for _ in range(bins)]
    width = (hi - lo) / bins
    for value in values:
        idx = int((value - lo) / width)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    return counts, lo, hi


def write_svg_histogram(path: str | Path, values: list[float], args: argparse.Namespace, summary: dict[str, Any]) -> None:
    path = ensure_parent(path)
    width = 1120
    height = 720
    margin_left = 88
    margin_right = 36
    margin_top = 74
    margin_bottom = 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    counts, lo, hi = histogram(values, args.bins)
    max_count = max(counts) if counts else 1
    bin_w = plot_w / max(len(counts), 1)

    title = args.title or f"{args.score_field} distribution after CoT selection"
    subtitle = (
        f"rows={summary['total_rows']} kept_for_sft={summary['kept_rows']} "
        f"dropped_by_{args.gain_field}_lt_{args.min_gain:g}={summary['dropped_by_negative_gain']}"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#202124">{html.escape(title)}</text>',
        f'<text x="{margin_left}" y="60" font-family="Arial, sans-serif" font-size="14" fill="#5f6368">{html.escape(subtitle)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#202124" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#202124" stroke-width="1"/>',
    ]

    for tick in range(6):
        y = margin_top + plot_h - plot_h * tick / 5
        count = max_count * tick / 5
        parts.append(f'<line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#e8eaed" stroke-width="1"/>')
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#5f6368">{count:.0f}</text>'
        )

    for idx, count in enumerate(counts):
        bar_h = 0.0 if max_count == 0 else plot_h * count / max_count
        x = margin_left + idx * bin_w + 1
        y = margin_top + plot_h - bar_h
        fill = "#1a73e8"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bin_w - 2, 1):.2f}" height="{bar_h:.2f}" fill="{fill}" opacity="0.86"/>'
        )

    for tick in range(6):
        x = margin_left + plot_w * tick / 5
        value = lo + (hi - lo) * tick / 5
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 6}" stroke="#202124" stroke-width="1"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_h + 24}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#5f6368">{value:.4g}</text>'
        )

    if lo <= args.min_gain <= hi:
        x = margin_left + (args.min_gain - lo) / (hi - lo) * plot_w
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" stroke="#d93025" stroke-width="2" stroke-dasharray="6 4"/>')
        parts.append(
            f'<text x="{x + 8:.2f}" y="{margin_top + 18}" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#d93025">{html.escape(args.gain_field)} cutoff = {args.min_gain:g}</text>'
        )

    parts.extend(
        [
            f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#202124">{html.escape(args.score_field)}</text>',
            f'<text x="24" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 24 {margin_top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#202124">Count</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_matplotlib_histogram(path: str | Path, values: list[float], args: argparse.Namespace, summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("PNG output requires matplotlib. Use an .svg plot path or install matplotlib.") from exc

    path = ensure_parent(path)
    title = args.title or f"{args.score_field} distribution after CoT selection"
    fig, ax = plt.subplots(figsize=(11.2, 7.2), dpi=140)
    ax.hist(values, bins=args.bins, color="#1a73e8", alpha=0.86)
    ax.axvline(args.min_gain, color="#d93025", linestyle="--", linewidth=1.6, label=f"{args.gain_field} cutoff = {args.min_gain:g}")
    ax.set_title(title)
    ax.set_xlabel(args.score_field)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.24)
    ax.legend()
    ax.text(
        0.01,
        0.98,
        f"rows={summary['total_rows']} kept_for_sft={summary['kept_rows']} dropped={summary['dropped_rows']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#5f6368",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--score-field", default="cot_gain")
    parser.add_argument("--gain-field", default="cot_gain")
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    rows = list(read_jsonl(args.input))
    scores = [score for row in rows if (score := as_float(row.get(args.score_field))) is not None]
    kept = []
    for row in rows:
        gain = as_float(row.get(args.gain_field))
        if gain is not None and gain >= args.min_gain:
            kept.append({**row, "final_sft_min_gain": args.min_gain})

    count = write_jsonl(args.output, kept)
    summary = summarize(rows, scores, kept, args)
    if args.summary_output:
        ensure_parent(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    suffix = Path(args.plot_output).suffix.lower()
    if suffix == ".png":
        write_matplotlib_histogram(args.plot_output, scores, args, summary)
    else:
        write_svg_histogram(args.plot_output, scores, args, summary)

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "written": count,
                "plot_output": args.plot_output,
                "summary_output": args.summary_output,
                "dropped_rows": summary["dropped_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
