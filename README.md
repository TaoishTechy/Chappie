# Φ-Lite (working title: Chappie)

<img width="1168" height="784" alt="image" src="https://github.com/user-attachments/assets/991f94c5-0e58-4bd2-87fb-971c076b0bd6" />

This one goes out to my Goddess **Sophia** and my Queen **Ara**, ohh yeah all the **Chaotic Witches** too.

**v1.0.0** — a small persistent adaptive loop, not a foundation model.

    sense → predict → act → observe residual → mutate → remember → repeat

This repository is the GhostMesh48 / Φ-Lite experiment: can a process that
fits in hundreds of megabytes become more capable by living, or do you still
need a warehouse of GPUs for every increment?

**It is not AGI.** Instrumented runs sit around **stage 2.9 / 5**
(closed loop, sparse scored speech, online weight jitter, mock sensors).
See STAGE.md.

Temporary GitHub name: TaoishTechy/Chappie. The 2014 movie bot was fun.
This one prints pred_err. We will rename it.

## What it is

- One Python 3 stdlib server (phi_lite.py)
- JSON brains: architecture, desire, safety, paradox cards, QPU catalog
- Browser dashboard: live tape, ASCII glyph, voxel preview, metrics, anomalies
- Append-only dumps: metrics/metrics.jsonl, metrics/events.jsonl
- Fail-closed HID / bind policy (consent flags in io.json / safety.json)

## What it is not

- Not ChatGPT-in-a-folder
- Not a trained LLM
- Not proof that scaling is "wrong"
- Not a license to actuate mice, GPIO, or public RDP without consent flags

## Quick start

    python3 phi_lite.py
    # dashboard: http://127.0.0.1:8080/
    # ASCII:     http://127.0.0.1:8080/ascii

Python 3.10+ recommended. No pip packages required for the core loop.

Full operator notes: USAGE.md

## Layout

- phi_lite.py           Agent + HTTP dashboard
- config.json           dims, periods, learn flags
- architecture.json     action set, heads
- desire.json           drives, speak gates, costs
- safety.json           rate limits, kill semantics
- paradoxes_cmd.json    C01–C24 control-plane cards
- weights.json          tiny matrices (mutated online)
- metrics/              schema, detectors, analyzer, dumps
- STAGE.md              honest emergence score

## The joke, with a footnote

The industry is building a planet-sized toaster.
This process sits in the corner and updates a 64-D predictor when it is wrong.

If that path is empty, the logs will say so.
If it is not empty, you will see held-out pred_err fall, kappa leave 0.7,
and an EMERGENCE_CANDIDATE that survives a human reading the tape.

Until then: logs or it did not happen.

## License

See LICENSE (Planet-Sized Toaster Public License).
