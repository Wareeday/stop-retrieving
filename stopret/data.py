"""Loading the multi-hop QA subsets used in this project.

The three datasets (HotpotQA, 2WikiMultihopQA, MuSiQue) come from the 1,000-question
subsets that the HippoRAG authors published together with a pooled passage corpus for
each one. Every question comes with its gold supporting passages, which is all we need
to decide whether a set of retrieved passages is "sufficient".

Source: https://github.com/OSU-NLP-Group/HippoRAG/tree/main/reproduce/dataset (MIT licence for
the repository; the datasets themselves keep their own licences, see the README).

The download is pinned to one commit of that repository and every file is checked against a
SHA-256 hash, so the data behind the reported numbers cannot change silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DATA_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
BASE_URL = (f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{DATA_COMMIT}"
            "/reproduce/dataset")
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SHA256 = {
    "hotpotqa.json": "3ad9c0bcbf93f41d7004ca6007049c904d2605046b314a2f7ecfb379c64cba6d",
    "hotpotqa_corpus.json": "9333647b922382776cd2cb02893b390d77984df85a91bfa8be411284978aca7d",
    "2wikimultihopqa.json": "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4",
    "2wikimultihopqa_corpus.json": "9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a",
    "musique.json": "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
    "musique_corpus.json": "73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
}


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


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filenames(datasets) -> list[str]:
    return [f"{name}{suffix}.json" for name in datasets for suffix in ("", "_corpus")]


def verify(data_dir: str | Path = "data/raw", datasets=DATASETS) -> list[str]:
    """Names of files that are missing or whose SHA-256 does not match (empty list = all good)."""
    data_dir = Path(data_dir)
    problems = []
    for filename in _filenames(datasets):
        path = data_dir / filename
        if not path.exists() or sha256_of(path) != SHA256[filename]:
            problems.append(filename)
    return problems


def download(data_dir: str | Path = "data/raw", datasets=DATASETS, force: bool = False) -> None:
    """Download the question and corpus files (about 43 MB in total).

    Files that are already present and match their hash are kept. Everything else is fetched
    to a temporary file, hash-checked, and only then moved into place, so an interrupted
    download can never leave a half-written file behind."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in _filenames(datasets):
        target = data_dir / filename
        if not force and target.exists() and sha256_of(target) == SHA256[filename]:
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"downloading {url}", flush=True)
        partial = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            actual = sha256_of(partial)
            if actual != SHA256[filename]:
                raise ValueError(
                    f"{filename}: SHA-256 mismatch (expected {SHA256[filename][:12]}..., got "
                    f"{actual[:12]}...). The download is corrupt or the source changed.")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)


def _gold_by_title(example: dict, title_to_id: dict[str, int]) -> list[int]:
    titles = {title for title, _sent_idx in example["supporting_facts"]}
    return sorted(title_to_id[t] for t in titles)


def load(name: str, data_dir: str | Path = "data/raw") -> Dataset:
    """Load one dataset and map each question's gold passages to corpus indices."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    data_dir = Path(data_dir)
    for filename in (f"{name}.json", f"{name}_corpus.json"):
        if not (data_dir / filename).exists():
            raise FileNotFoundError(
                f"{data_dir / filename} not found. Run `python -m stopret download` first.")
    with open(data_dir / f"{name}.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(data_dir / f"{name}_corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    titles = [p["title"] for p in corpus]
    texts = [p["text"] for p in corpus]
    ds = Dataset(name=name, titles=titles, texts=texts)

    if name == "musique":
        # MuSiQue reuses titles for different paragraphs, so match on (title, text).
        key_to_id = {(t, x): i for i, (t, x) in enumerate(zip(titles, texts, strict=True))}
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
