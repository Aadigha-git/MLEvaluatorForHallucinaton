"""BM25 over Pyserini's prebuilt `wikipedia-dpr-100w` index (docs/PREREGISTRATION.md).

The query is the question (the claim for FEVER), never the generated answer, so both generators
get the same passages. Uses a project-local JDK 21; the system Java is left alone.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class JavaNotFound(RuntimeError):
    pass


def configure_java(jdk_glob: str, root: str | Path = ".") -> str:
    """Point pyjnius at the project-local JDK 21. Must run before pyserini is imported."""
    homes = sorted(glob.glob(str(Path(root) / jdk_glob)))
    if not homes:
        raise JavaNotFound(f"no JDK matching {jdk_glob}")
    home = str(Path(homes[-1]).resolve())
    release = (Path(home) / "release").read_text()
    if 'JAVA_VERSION="21' not in release:
        raise JavaNotFound(f"{home} is not JDK 21")
    os.environ["JAVA_HOME"] = home
    return home


@dataclass(frozen=True)
class Passage:
    docid: str
    score: float
    title: str
    text: str

    def formatted(self) -> str:
        """F1: retrieved passages enter features as "{title}: {text}"."""
        return f"{self.title}: {self.text}"


def parse_dpr_contents(contents: str) -> tuple[str, str]:
    """Wiki-DPR documents store `"Title"\\ntext`; returns (title, text)."""
    first, _, rest = contents.partition("\n")
    title = first.strip()
    if len(title) >= 2 and title[0] == title[-1] == '"':
        title = title[1:-1]
    return title, rest.strip()


class BM25Retriever:
    def __init__(self, index_dir: str | Path, k1: float, b: float, k: int):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1, b)
        self.k = k
        self.num_docs = self.searcher.num_docs

    def search(self, query: str) -> tuple[list[Passage], float]:
        t0 = time.perf_counter()
        hits = self.searcher.search(query, k=self.k)
        passages = []
        for hit in hits:
            raw = json.loads(self.searcher.doc(hit.docid).raw())
            title, text = parse_dpr_contents(raw["contents"])
            passages.append(Passage(hit.docid, round(float(hit.score), 6), title, text))
        return passages, time.perf_counter() - t0


def retrieval_row(question_id: str, query: str, passages: list[Passage], seconds: float) -> dict:
    return {
        "question_id": question_id,
        "query": query,
        "passages": [asdict(p) for p in passages],
        "retrieval_s": round(seconds, 6),
    }


def passages_from_row(row: dict) -> list[Passage]:
    return [Passage(**p) for p in row["passages"]]
