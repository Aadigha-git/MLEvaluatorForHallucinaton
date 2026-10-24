"""Answer a question with Model A and score the answer with the frozen Step 7 artifact (demo only).

Everything is the frozen pipeline: the `qa-v1` prompt with primary-answer settings, BM25 retrieval of
the top 5 Wiki-DPR passages with the question as query, `features-v1`, and the artifact's F1
threshold, calibrator, and policy thresholds. Chatbot spend is capped from a call log.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from hdetect.cleaning import clean_answer
from hdetect.config import load_config
from hdetect.data.io import read_jsonl
from hdetect.features.pipeline import build_extractor, build_retriever, load_feature_config
from hdetect.generation.client import OpenAICompatibleClient
from hdetect.generation.config import load_generation_config
from hdetect.generation.prompts import PROMPTS, messages
from hdetect.service.explain import contributions, explain
from hdetect.service.schema import now
from hdetect.training.calibration import calibrated
from hdetect.training.inputs import load_dev_inputs
from hdetect.training.models import score as clf_score


class BudgetExceeded(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band_of(p: float, t_low: float, t_high: float) -> str:
    return "accept" if p < t_low else ("escalate" if p < t_high else "flag")


@dataclass
class Answer:
    text: str
    model: str | None
    cost_usd: float
    latency_s: float
    finish_reason: str | None


class Scorer:
    def __init__(self, root: Path, demo_config: str = "configs/demo.yaml") -> None:
        self.root = root
        self.dcfg = yaml.safe_load((root / demo_config).read_text())
        cfg = load_config(root / "configs/default.yaml")
        self.fcfg = load_feature_config(root / "configs/features.yaml")
        gcfg = load_generation_config(root / "configs/generation.yaml")
        tcfg = yaml.safe_load((root / "configs/training.yaml").read_text())

        c = self.dcfg["classifier"]
        path = root / c["artifact"]
        manifest = json.loads((root / "data/models/step7/manifest.json").read_text())
        expected = {Path(a["path"]).name: a["sha256"] for a in manifest["artifacts"]}
        self.artifact_sha256 = sha256(path)
        if expected.get(path.name) != self.artifact_sha256:
            raise RuntimeError(f"{path} does not match the Step 7 manifest")
        self.art = joblib.load(path)
        if (self.art["tier"], self.art["condition"], self.art["seed"]) != (c["tier"], c["condition"], c["seed"]):
            raise RuntimeError("artifact does not match configs/demo.yaml")
        self.features = tuple(self.art["features"])

        dev = load_dev_inputs(cfg, self.fcfg, tcfg["paths"], val_conditions=(c["condition"], "none"))
        self.medians = {f: float(dev.val[c["condition"]].X[f].median()) for f in self.features}

        cb = self.dcfg["chatbot"]
        self.spec = gcfg.generators[cb["generator"]]
        self.template = PROMPTS[cb["prompt_version"]]
        self.max_tokens = gcfg.max_tokens[cb["prompt_version"]]
        self.sampling = gcfg.primary
        self.api = gcfg.api
        self.client = None
        self.call_log = root / cb["call_log"]
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        self.spent = sum(r.get("cost_usd") or 0.0 for r in read_jsonl(self.call_log)) if self.call_log.exists() else 0.0

        self.retriever = build_retriever(self.fcfg)
        self.extractor = build_extractor(self.fcfg, root / "data/demo/feature_cache.sqlite")

    def versions(self) -> dict[str, Any]:
        a = self.art
        return {
            "generator": {"name": self.spec.name, "model_id": self.spec.model_id,
                          "prompt_version": self.dcfg["chatbot"]["prompt_version"]},
            "classifier": {"artifact_sha256": self.artifact_sha256, "artifact_version": a["artifact_version"],
                           "family": a["family"], "seed": a["seed"], "f1_threshold": a["f1_threshold"]["value"],
                           "feature_medians_val_id": self.medians},
            "calibration": {"method": a["calibration_method"]},
            "policy": {"t_low": a["policy"]["t_low"], "t_high": a["policy"]["t_high"],
                       "feasible": a["policy"]["feasible"]},
        }

    def answer(self, question_id: str, question: str) -> Answer:
        prompt = self.template.format(question=question)
        bound = self.spec.cost_usd(len(prompt), self.max_tokens)
        cap = self.dcfg["chatbot"]["budget_usd"]
        if self.spent + bound > cap:
            raise BudgetExceeded(f"demo chatbot spend ${self.spent:.4f} of ${cap:.2f}; next call may cost ${bound:.5f}")
        if self.client is None:
            self.client = OpenAICompatibleClient(self.api)
        t0 = time.perf_counter()
        row = {"question_id": question_id, "started_at": now(), "model_id": self.spec.model_id}
        try:
            c = self.client.complete(model=self.spec.model_id, messages=messages(prompt),
                                     temperature=self.sampling.temperature, top_p=self.sampling.top_p,
                                     max_tokens=self.max_tokens, seed=None)
        except Exception as e:
            row.update(status="error", error=type(e).__name__, latency_s=time.perf_counter() - t0, cost_usd=0.0)
            self._log(row)
            raise
        cost = self.spec.cost_usd(c.prompt_tokens or len(prompt), c.completion_tokens or self.max_tokens)
        row.update(status="ok", latency_s=round(time.perf_counter() - t0, 4), cost_usd=cost, model=c.model,
                   response_id=c.response_id, finish_reason=c.finish_reason, prompt_tokens=c.prompt_tokens,
                   completion_tokens=c.completion_tokens)
        self._log(row)
        self.spent += cost
        return Answer(c.text, c.model, cost, row["latency_s"], c.finish_reason)

    def _log(self, row: dict) -> None:
        with self.call_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def score(self, question: str, answer: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        text = clean_answer(answer)
        if not text:
            return {"error": "empty answer after cleaning"}
        passages, retrieval_s = self.retriever.search(question)
        formatted = [p.formatted() for p in passages]
        x = {**self.extractor.free(text, [])["features"],
             **self.extractor.evidence(text, question, formatted)["features"]}
        X = np.ascontiguousarray([[x[f] for f in self.features]], dtype=float)
        s = float(clf_score(self.art["estimator"], X)[0])
        p = float(calibrated(self.art["calibrator"], X)[0])
        pol = self.art["policy"]
        band = band_of(p, pol["t_low"], pol["t_high"])
        deltas = contributions(x, self.features, self.medians,
                               lambda M: clf_score(self.art["estimator"], np.ascontiguousarray(M)))
        return {
            "score": s, "p_hallucinated": p, "band": band,
            "predicted_hallucinated": bool(s >= self.art["f1_threshold"]["value"]),
            "explanation": explain(band, x, deltas, self.medians, self.dcfg["explanation"]["top_k"]),
            "features": {f: x[f] for f in self.features},
            "evidence": [{"title": q.title, "text": q.text} for q in passages],
            "retrieval_s": retrieval_s, "scoring_s": time.perf_counter() - t0, "error": None,
        }
