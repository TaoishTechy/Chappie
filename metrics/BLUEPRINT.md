# Blueprint: Scientific Metric Dumps and Local Audit Analysis

**Revision:** Φ-Lite Metrics R1  
**Purpose:** Turn runtime traces into an auditable scientific record so emergence claims can be accepted or rejected against the 144-point audit.  
**Constraint:** One parser (`analyze_metrics.py`). Schemas, detectors, and reports stay JSON/JSONL.

---

## 1. Problem this revision solves

The previous run produced a closed loop that *looked* busy (PLAN every ~350 ms, horizon 10, occasional REPAIR) while:

- weights never mutated
- policy entropy stayed near uniform
- attention and horizon were unused
- ledger growth caused `EIO`
- two prompts were template-echoed

Without cycle-level metrics, those facts require a manual forensic pass. This blueprint makes the same conclusions *machine-checkable* and maps each detector to audit IDs (A–L).

---

## 2. Dump architecture

```
cycle ──► metrics.jsonl     (dense, numeric, one line / cycle)
     └─► events.jsonl       (sparse; detectors only)
periodic ─► audit_snapshot.json
baselines ─► novelty_index.json
parser    ─► report.json
```

Do **not** dump full 64-d `b` on every tick into a second copy. Keep `ledger.jsonl` as the raw boundary tape, but rotate it (see §7). Metrics lines store *scalars derived from* `b`, `h`, `s`, `π`.

### 2.1 `metrics.jsonl` record (required fields)

| Field | Why |
| --- | --- |
| `t`, `cycle`, `dt_ms` | timing anomalies, watchdog |
| `action`, `pi`, `H_pi`, `argmax_margin` | collapse vs exploration |
| `horizon` | decorative vs used |
| `fp_err`, `fp_iter` | self-model health |
| `attn`, `c` | currently unused / missing; reserve the keys |
| `b_l2`, `h_l2`, `s_l2`, `novelty_b` | regime / novelty |
| `ledger_n`, `prompt_pending` | buffer and goal bits |
| `io_error` | EIO / disk |
| `weights_sha1` | mutation detector |

Instrumentation hook (single function in `phi_lite.py`, no second agent): after `step()`, append one compact JSON object. If write fails, set `io_error` and continue; never block the loop on audit I/O.

### 2.2 `events.jsonl`

Emit only when a detector in `detectors.json` fires. Event objects are `{type, cycle, t, ...payload}`. Types are closed-vocabulary (see `schema.json`).

---

## 3. What to look for (scientific targets)

### 3.1 Emergent-event candidates

Promotion to “emergence candidate” requires **≥3 independent signals in the same window**, and **must not** fire on mock-orbit periodicity alone.

| Signal | Accept if | Reject if |
| --- | --- | --- |
| `ACTION_INNOVATION` | New action after ≥64 cycles of lock, *and* `H_pi` drop that persists | Single prompt-bias reply |
| `ENTROPY_DROP` | `H_pi` falls and stays below 0.6 for 16 cycles | One-tick softmax jitter |
| `REGIME_SHIFT` | Ledger centroid jump **and** prediction residual would fall (when learning exists) | Sinusoid phase of mock blob |
| `WEIGHT_MUTATION` | Hash change with a logged learning step | File rewrite of the same random init |
| `HORIZON_SHIFT` | H tracks `novelty_z` or `fp_err` | H frozen at 10 |
| `FP_TRANSIENT` then settle | Residual spike then contraction after repair that *changes W or E* | REPAIR text only |

Until learning exists, `EMERGENCE_CANDIDATE` is a hypothesis tag, not a stage promotion.

### 3.2 Anomalies

| Detector | Audit map | Typical cause in current code |
| --- | --- | --- |
| `MODE_COLLAPSE` | A6, C29 | Frozen argmax PLAN |
| `HORIZON_STICK` | F61–F72 | Frozen horizon head |
| `REPAIR_NOOP` | G75, G83 | Action label without ∇L |
| `IO_FAULT` | H86 | JSONL of 64 floats / cycle |
| `DT_GAP` | L / runtime | Process stop or SD stall |
| `CYCLE_RESET` | A9 | Second process or RESET |
| `MONODROMY_HIT` | G74 | Only near identical `b` |
| `STATE_DISCONTINUITY` | A9, B22 | `h` jump vs persisted state |

### 3.3 Mutations

Treat any change to `weights.json`, `architecture.json`, or `safety.json` as a mutation event.

- Allowed later: optimizer step with `{param, l2_delta, loss_before, loss_after}`.
- Forbidden without flag: `allow_self_modify` (K127). Parser must flag weight edits when that flag is false.

### 3.4 Novelty

`novelty_b = ||b_t − centroid_{t-W}||` with W = 32 (ledger cap).

- z ≥ 3: `NOVEL_BOUNDARY`
- Do not confuse mock-orbit amplitude with semantic novelty. Require concurrent change in `h` *direction* (cosine < 0.7 vs recent mean `h`) before calling it cognitive novelty.

---

## 4. Mapping onto the 144-point audit (enhancement queue)

Use dumps to *score movement*, not to restated prose.

| Audit gap | Metric that would raise the score |
| --- | --- |
| D37–D48 confidence | persist `c`, calibration error vs outcomes |
| E49–E60 attention | `attn` must correlate with next-step novelty; log if ignored |
| F61–F72 horizon | variance of `horizon` > 0; H used inside GRU unroll |
| G73–G84 repair | `WEIGHT_MUTATION` or `E` l2 drop after REPAIR |
| I97–I108 compression | description length of π / reused policy ids |
| J109–J120 counterfactual | rollout regret fields when implemented |
| A8 learning | nonzero `weights_sha1` change rate with falling `fp_err` or pred-err |

`audit_snapshot.json` (every N cycles or on shutdown) should store section scores **only when a metric exists**. Otherwise keep the human 144-point scores and attach event counts as evidence footnotes.

---

## 5. Local analysis protocol

```bash
python3 metrics/analyze_metrics.py
```

Inputs (legacy-compatible): `logs.jsonl`, `ledger.jsonl`, `state.json`, `weights.json`.  
Outputs: `metrics/events.jsonl`, `metrics/report.json`.

Parser rules:

1. Never claim AGI. `stage_estimate` stays `scaffold_closed_loop` unless learning + non-template policy + horizon mobility co-occur.
2. Collapse is a first-class finding (share ≥ 0.85).
3. Parse errors and `error` keys become `IO_FAULT`.
4. Map each event type to audit IDs from `detectors.json`.
5. Enhancement list is generated from which detectors fired, not from a static sermon.

---

## 6. Instrumentation contract for the next `phi_lite.py` revision

Add only:

1. `dump_metric(rec)` — append one line, swallow `OSError`.
2. Fields listed in §2.1.
3. Rotate `ledger.jsonl` at 2 000 lines (keep `ledger.jsonl.prev`).
4. Compute `novelty_b` from in-memory deque, not by rereading the file.
5. On REPAIR, write an event *and* a before/after `fp_err`. If weights still frozen, the event stays `REPAIR_NOOP`.

No second service. Dashboard may show last 5 event types from `events.jsonl` tail.

---

## 7. Storage budget (Pi)

| File | Cap | Policy |
| --- | --- | --- |
| `metrics.jsonl` | 10 000 lines | rotate |
| `events.jsonl` | 2 000 lines | rotate |
| `ledger.jsonl` | 2 000 lines | rotate; in-memory ring remains 32 |
| `logs.jsonl` | 2 000 lines | rotate |

This is the direct mitigation for `[Errno 5] Input/output error` seen in the audited run.

---

## 8. Decision rule for “did we enhance?”

A revision is an enhancement only if a later `report.json` shows at least one of:

- `dominant_share` down and `action_entropy_full` up, without becoming uniform-and-useless
- `HORIZON_STICK` absent
- `WEIGHT_MUTATION` present *with* decreasing residual
- `IO_FAULT` count near zero
- `c` field non-null and used for gating

Otherwise the system remains Stage 1.5 scaffold, regardless of dashboard activity.
