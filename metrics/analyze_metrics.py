#!/usr/bin/env python3
"""Local parser for Φ-Lite scientific dumps and legacy jsonl.

Reads ledger.jsonl, logs.jsonl, state.json (and metrics.jsonl if present).
Writes metrics/report.json and metrics/events.jsonl.
No training. Stdlib + optional numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, deque
from pathlib import Path

import os

MET = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PHILITE_ROOT", MET.parent))
DET = json.loads((MET / "detectors.json").read_text(encoding="utf-8"))
TH = DET["thresholds"]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"_parse_error": True, "raw": line[:200]})
    return out


def l2(xs):
    return math.sqrt(sum(float(x) * float(x) for x in xs))


def entropy_from_actions(window):
    if not window:
        return 0.0
    c = Counter(window)
    n = len(window)
    h = 0.0
    for v in c.values():
        p = v / n
        h -= p * math.log(p + 1e-12)
    return h


def main():
    logs = load_jsonl(ROOT / "logs.jsonl")
    ledger = load_jsonl(ROOT / "ledger.jsonl")
    state = {}
    sp = ROOT / "state.json"
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
    weights_path = ROOT / "weights.json"
    wstat = None
    if weights_path.exists():
        st = weights_path.stat()
        wstat = {"size": st.st_size, "mtime": st.st_mtime}

    events = []
    actions, cycles, errors = [], [], []
    for rec in logs:
        if rec.get("_parse_error"):
            events.append({"type": "IO_FAULT", "detail": "malformed log line"})
            continue
        if "error" in rec:
            errors.append(rec)
            events.append({"type": "IO_FAULT", "t": rec.get("t"), "detail": rec.get("error")})
            continue
        actions.append(rec.get("action"))
        cycles.append(rec.get("cycle"))
        if rec.get("action") == "PROMPT_REPLY":
            events.append(
                {
                    "type": "PROMPT_CONSUMED",
                    "cycle": rec.get("cycle"),
                    "ts": rec.get("ts"),
                    "text": rec.get("text"),
                }
            )
        if rec.get("action") == "REPAIR":
            events.append({"type": "REPAIR_NOOP", "cycle": rec.get("cycle"), "note": "text-only repair in v1"})

    act_n = len(actions)
    counts = Counter(actions)
    top, topn = (counts.most_common(1)[0] if counts else ("NONE", 0))
    share = (topn / act_n) if act_n else 0.0
    if share >= TH["mode_collapse_share"]:
        events.append(
            {
                "type": "MODE_COLLAPSE",
                "action": top,
                "share": round(share, 4),
                "n": act_n,
            }
        )

    if cycles:
        for i in range(1, len(cycles)):
            if cycles[i] is None or cycles[i - 1] is None:
                continue
            if cycles[i] < cycles[i - 1] + TH["cycle_reset_delta"]:
                events.append(
                    {
                        "type": "CYCLE_RESET",
                        "from": cycles[i - 1],
                        "to": cycles[i],
                    }
                )

    # ledger novelty / regime
    vecs, ts = [], []
    for rec in ledger:
        if "b" in rec and isinstance(rec["b"], list):
            vecs.append(rec["b"])
            ts.append(rec.get("t"))
    novelty_scores = []
    if vecs:
        dim = len(vecs[0])
        centroid = [sum(v[j] for v in vecs) / len(vecs) for j in range(dim)]
        for i, v in enumerate(vecs):
            d = l2([v[j] - centroid[j] for j in range(dim)])
            novelty_scores.append(d)
        mu = statistics.fmean(novelty_scores)
        sd = statistics.pstdev(novelty_scores) or 1e-9
        for i, d in enumerate(novelty_scores):
            z = (d - mu) / sd
            if z >= TH["novelty_z"]:
                events.append({"type": "NOVEL_BOUNDARY", "i": i, "z": round(z, 3), "t": ts[i]})
        dts = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if ts[i] and ts[i - 1]]
        for i, dt in enumerate(dts, start=1):
            if dt >= TH["dt_gap_s"]:
                events.append({"type": "DT_GAP", "i": i, "dt_s": round(dt, 3)})

        # crude regime: first half vs second half mean vector L2
        mid = len(vecs) // 2
        if mid > 8:
            m1 = [sum(v[j] for v in vecs[:mid]) / mid for j in range(dim)]
            m2 = [sum(v[j] for v in vecs[mid:]) / (len(vecs) - mid) for j in range(dim)]
            shift = l2([a - b for a, b in zip(m1, m2)])
            if shift > 0.35:
                events.append({"type": "REGIME_SHIFT", "centroid_l2": round(shift, 4)})

    # horizon stick from log text if present
    horizons = []
    for rec in logs:
        txt = rec.get("text") or ""
        if "Horizon=" in txt:
            try:
                horizons.append(int(txt.split("Horizon=")[1].split(",")[0]))
            except Exception:
                pass
    if horizons:
        hc = Counter(horizons)
        hs, hn = hc.most_common(1)[0]
        if hn / len(horizons) >= TH["horizon_stick_share"]:
            events.append({"type": "HORIZON_STICK", "H": hs, "share": round(hn / len(horizons), 4)})

    # state discontinuity vs last ledger energy
    if state.get("h"):
        hl2 = l2(state["h"])
        sl2 = l2(state.get("s") or [0])
    else:
        hl2 = sl2 = None

    emergence_hits = [e["type"] for e in events]
    rare = {"NOVEL_BOUNDARY", "REGIME_SHIFT", "ACTION_INNOVATION", "ENTROPY_DROP", "WEIGHT_MUTATION"}
    if len(rare.intersection(emergence_hits)) >= TH["emergence_min_signals"]:
        events.append(
            {
                "type": "EMERGENCE_CANDIDATE",
                "signals": sorted(rare.intersection(emergence_hits)),
                "note": "candidate only; requires learning + non-template policy to confirm",
            }
        )

    report = {
        "schema": "phi-lite-scientific-metrics/1.0",
        "n_logs": act_n,
        "n_errors": len(errors),
        "n_ledger": len(vecs),
        "action_counts": dict(counts),
        "dominant_action": top,
        "dominant_share": round(share, 4),
        "action_entropy_full": round(entropy_from_actions(actions), 4),
        "cycle_min": min(cycles) if cycles else None,
        "cycle_max": max(cycles) if cycles else None,
        "state_cycle": state.get("cycle"),
        "state_last_action": state.get("last_action"),
        "h_l2": None if hl2 is None else round(hl2, 4),
        "s_l2": None if sl2 is None else round(sl2, 4),
        "ledger_mean_novelty": None
        if not novelty_scores
        else round(statistics.fmean(novelty_scores), 4),
        "weights": wstat,
        "event_counts": dict(Counter(e["type"] for e in events)),
        "audit_pointers": {
            k: DET["map_to_audit"].get(k, [])
            for k in set(e["type"] for e in events)
        },
        "enhancement_priority": [
            "online weight update + dump WEIGHT_MUTATION with signed diff",
            "persist c, attn, H_pi, argmax_margin each cycle in metrics.jsonl",
            "stop full-vector jsonl or rotate to cap EIO",
            "break PLAN lock: cost on repeated action + log ACTION_INNOVATION",
            "horizon must change with novelty_z and be written to metrics",
            "REPAIR must mutate E or W and emit before/after hash",
        ],
        "stage_estimate": {
            "label": "scaffold_closed_loop",
            "agi_emergence": 1.5,
            "note": "Do not promote stage without WEIGHT_MUTATION plus falling prediction error.",
        },
    }

    (MET / "events.jsonl").write_text(
        "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events),
        encoding="utf-8",
    )
    (MET / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(events), "report": str(MET / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
