"""Build a Markdown leaderboard from saved JSON files in ../results/."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="ndcg@10", help="Headline metric (default: ndcg@10)")
    ap.add_argument("--include-limited", action="store_true", help="Include smoke-test (limited) runs")
    ap.add_argument("--tag", default="", help="Only include JSONs ending with __<tag>.json")
    a = ap.parse_args()

    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    tasks_seen: list[str] = []

    for p in sorted(RESULTS.glob("*.json")):
        if a.tag and not p.name.endswith(f"__{a.tag}.json"):
            continue
        rec = json.loads(p.read_text())
        if rec.get("limited") and not a.include_limited:
            continue
        task = rec["task"]
        if ":" not in task:
            task = f"coir:{task}"
        by_model[rec["model"]][task] = rec
        if task not in tasks_seen:
            tasks_seen.append(task)

    if not by_model:
        print("No matching results found.")
        return

    tasks = sorted(tasks_seen, key=lambda t: (0 if t.startswith("beir:") else 1, t))
    header = ["Model"] + tasks + ["General", "Code", "Overall", "Avg Q/D terms", "Lat ms"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    def sort_key(item):
        vals = [r["metrics"][a.metric] for r in item[1].values() if a.metric in r.get("metrics", {})]
        return -(float(np.mean(vals)) if vals else -1.0)

    for model, tmap in sorted(by_model.items(), key=sort_key):
        cols = [model]
        gen, code, all_v, q_terms, d_terms, lats = [], [], [], [], [], []
        for t in tasks:
            r = tmap.get(t)
            if r and a.metric in r.get("metrics", {}):
                v = r["metrics"][a.metric]
                cols.append(f"{v:.4f}")
                all_v.append(v)
                (gen if t.startswith("beir:") else code).append(v)
                if "avg_query_terms" in r:
                    q_terms.append(r["avg_query_terms"])
                if "avg_doc_terms" in r:
                    d_terms.append(r["avg_doc_terms"])
                if "query_latency_ms_bs1" in r:
                    lats.append(r["query_latency_ms_bs1"])
            else:
                cols.append("—")
        cols.append(f"{np.mean(gen):.4f}" if gen else "—")
        cols.append(f"{np.mean(code):.4f}" if code else "—")
        cols.append(f"{np.mean(all_v):.4f}" if all_v else "—")
        cols.append(f"{np.mean(q_terms):.0f}/{np.mean(d_terms):.0f}" if q_terms and d_terms else "—")
        cols.append(f"{np.mean(lats):.1f}" if lats else "—")
        print("| " + " | ".join(cols) + " |")


if __name__ == "__main__":
    main()
