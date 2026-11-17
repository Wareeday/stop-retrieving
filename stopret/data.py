"""Loading the multi-hop QA subsets used in this project.

The three datasets (HotpotQA, 2WikiMultihopQA, MuSiQue) come from the 1,000-question
subsets that the HippoRAG authors published together with a pooled passage corpus for
each one. Every question comes with its gold supporting passages, which is all we need
to decide whether a set of retrieved passages is "sufficient".

Source: https://github.com/OSU-NLP-Group/HippoRAG/tree/main/reproduce/dataset
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    gold_ids: list[int]  # indices into Dataset.titles / Dataset.texts


@dataclass
class Dataset:
    name: str
    titles: list[str]
    texts: list[str]
    examples: list[Example] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)


def download(data_dir: str | Path = "data/raw", datasets=DATASETS) -> None:
    """Download the question and corpus files (about 43 MB in total) if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        for suffix in ("", "_corpus"):
            target = data_dir / f"{name}{suffix}.json"
            if target.exists() and target.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{name}{suffix}.json"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, target)


def _gold_by_title(example: dict, title_to_id: dict[str, int]) -> list[int]:
    titles = {title for title, _sent_idx in example["supporting_facts"]}
    return sorted(title_to_id[t] for t in titles)


def load(name: str, data_dir: str | Path = "data/raw") -> Dataset:
    """Load one dataset and map each question's gold passages to corpus indices."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    data_dir = Path(data_dir)
    with open(data_dir / f"{name}.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(data_dir / f"{name}_corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    titles = [p["title"] for p in corpus]
    texts = [p["text"] for p in corpus]
    ds = Dataset(name=name, titles=titles, texts=texts)

    if name == "musique":
        # MuSiQue reuses titles for different paragraphs, so match on (title, text).
        key_to_id = {(t, x): i for i, (t, x) in enumerate(zip(titles, texts))}
        for q in questions:
            gold = sorted(
                key_to_id[(p["title"], p["paragraph_text"])]
                for p in q["paragraphs"]
                if p["is_supporting"]
            )
            ds.examples.append(Example(q["id"], q["question"], q["answer"], gold))
    else:
        title_to_id = {t: i for i, t in enumerate(titles)}
        if len(title_to_id) != len(titles):
            raise ValueError(f"{name}: duplicate titles, cannot match gold by title")
        for q in questions:
            ds.examples.append(
                Example(q["_id"], q["question"], q["answer"], _gold_by_title(q, title_to_id))
            )
    return ds
