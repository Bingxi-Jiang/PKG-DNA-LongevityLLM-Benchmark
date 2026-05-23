"""Small file IO helpers shared by builders and evaluators."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_split_records(records: list[dict], output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "test"]:
        split_records = [record for record in records if record["split"] == split]
        path = Path(output_dir) / f"{stem}_{split}.jsonl"
        write_jsonl(path, split_records)

        labels = Counter(record["messages"][-1]["content"] for record in split_records)
        print(f"  {path} - {len(split_records)} rows, labels: {dict(labels)}")
