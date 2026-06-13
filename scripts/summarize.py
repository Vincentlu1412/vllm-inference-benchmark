from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUP_KEYS = [
    "backend",
    "model",
    "case",
    "batch_size",
    "prompt_tokens_per_request",
    "prompt_tokens_target",
    "max_new_tokens",
    "concurrency",
    "temperature",
    "top_p",
]


METRIC_KEYS = [
    "requests_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "latency_ms_p50",
    "latency_ms_p95",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "e2e_ms_p50",
    "e2e_ms_p95",
]


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_records(records: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(item) for item in GROUP_KEYS if record.get(item) is not None)
        grouped[key].append(record)
    return grouped


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    first = items[0]
    for key in GROUP_KEYS:
        if first.get(key) is not None:
            summary[key] = first[key]
    for key in METRIC_KEYS:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        if values:
            summary[f"{key}_median"] = statistics.median(values)
    summary["runs"] = len(items)
    return summary


def to_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "backend",
        "model",
        "case",
        "batch_size",
        "concurrency",
        "prompt_tokens_per_request",
        "prompt_tokens_target",
        "max_new_tokens",
        "requests_per_s_median",
        "output_tokens_per_s_median",
        "latency_ms_p95_median",
        "ttft_ms_p95_median",
        "tpot_ms_p95_median",
        "e2e_ms_p95_median",
        "runs",
    ]
    present_columns = [column for column in columns if any(row.get(column) is not None for row in rows)]
    lines = ["# Benchmark Summary", ""]
    lines.append("| " + " | ".join(present_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(present_columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in present_columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("results/summary.md"))
    args = parser.parse_args()

    records = load_records(args.inputs)
    rows = [summarize_group(items) for items in group_records(records).values()]
    rows.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    markdown = to_markdown(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
