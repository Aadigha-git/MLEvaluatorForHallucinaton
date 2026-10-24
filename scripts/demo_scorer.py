"""Demo scoring service: consume question events, answer with Model A, score, publish verdict events.

`--once "question"` answers and scores one question without Kafka and prints the verdict event
(useful for a smoke test and as the example payload for the dashboard). Otherwise it consumes
`questions.v1` and produces `verdicts.v1` (configs/demo.yaml) until interrupted or the chatbot
budget is reached. Offsets are committed only after a verdict is published.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from hdetect.judging.clients import load_dotenv
from hdetect.service.schema import encode, now, parse_question, verdict_event
from hdetect.service.scorer import BudgetExceeded, Scorer

ROOT = Path(__file__).resolve().parents[1]


def handle(scorer: Scorer, q: dict) -> dict:
    ts = {"produced": q["produced_at"], "received": now()}
    base = dict(question_id=q["question_id"], question=q["question"], source=q["source"],
                reference_answer=q.get("reference_answer"), tier=scorer.dcfg["classifier"]["tier"],
                condition=scorer.dcfg["classifier"]["condition"], **scorer.versions())
    empty = dict(answer=None, p_hallucinated=None, score=None, predicted_hallucinated=None, band=None,
                 explanation=None, features=None, evidence=None, cost_usd=None)
    try:
        a = scorer.answer(q["question_id"], q["question"])
    except BudgetExceeded:
        raise
    except Exception as e:
        return verdict_event(**base, **empty, latency_s={}, timestamps=ts, error=f"chatbot: {type(e).__name__}")
    ts["answered"] = now()
    r = scorer.score(q["question"], a.text)
    ts["scored"] = now()
    latency = {"chatbot_s": a.latency_s, "retrieval_s": r.get("retrieval_s"), "scoring_s": r.get("scoring_s")}
    if r["error"]:
        return verdict_event(**base, **{**empty, "answer": a.text, "cost_usd": a.cost_usd}, latency_s=latency,
                             timestamps=ts, error=r["error"])
    fields = {k: r[k] for k in ("p_hallucinated", "score", "predicted_hallucinated", "band", "explanation",
                                "features", "evidence")}
    return verdict_event(**base, answer=a.text, **fields, cost_usd=a.cost_usd, latency_s=latency, timestamps=ts,
                         error=None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", metavar="QUESTION", help="score one question without Kafka and print the event")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")
    scorer = Scorer(ROOT)
    if args.once:
        q = {"schema_version": "demo-event-v1", "question_id": f"once-{uuid.uuid4().hex[:8]}",
             "question": args.once, "source": "manual", "produced_at": now()}
        print(json.dumps(handle(scorer, q), indent=1, ensure_ascii=False))
        return

    from confluent_kafka import Consumer, Producer

    k = scorer.dcfg["kafka"]
    consumer = Consumer({"bootstrap.servers": k["bootstrap_servers"], "group.id": k["scorer_group_id"],
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    producer = Producer({"bootstrap.servers": k["bootstrap_servers"]})
    consumer.subscribe([k["questions_topic"]])
    print(f"scoring {k['questions_topic']} -> {k['verdicts_topic']}; chatbot spend so far ${scorer.spent:.4f}",
          flush=True)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"kafka error: {msg.error()}", file=sys.stderr, flush=True)
                continue
            q = parse_question(msg.value())
            event = handle(scorer, q)
            producer.produce(k["verdicts_topic"], key=q["question_id"].encode(), value=encode(event))
            producer.flush()
            consumer.commit(message=msg, asynchronous=False)
            print(f"{q['question_id']}: {event['band'] or event['error']} (spent ${scorer.spent:.4f})", flush=True)
    except BudgetExceeded as e:
        print(f"stopped: {e}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
