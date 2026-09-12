# Φ-Lite v0.8.0 — talk drive, crystal, and command-paradox batches

Evidence: state.json cyc 7940, crystal.json slots 7685–7940 (n=256), logs.jsonl ~5984–7940, io.json (video/audio_in now on), desire.json 0.7.0, paradoxes_talk.json 0.7.2 (present, not used as SPEAK text).

Honest stage: ~2.9 / 5. Speech volume is up versus the energy-death window. Content is still telemetry or prompt echo. Crystal is a ring, not a mind. last_recall.d = 0 was a bug (matched the slot just written); v0.8.5 recalls slots[:-1].

## 1. Overview of this epoch

| Signal | Now | vs energy-death run | Meaning |
|---|---|---|---|
| Dominant act | IDLE + LOOK | same | Quiet drive still wins |
| pair_share | 0 | same good | LISTEN↔REPAIR lock stays broken |
| pred_err | 0.0609 | flat | Dictionary fitted mock orbit |
| self_err | 0.23 | down from 0.47 | Self-head less lost |
| intent_gap | 0.42 | was 0.56–0.99 | Policy less theatrical |
| energy | 0.02 at snapshot | still poor | SPEAK still expensive |
| κ, λ_meta | 0.70 / 0.0808 | glued | Not live controls |
| horizon | 1 | stuck | C10 still decorative |
| Crystal | 256 slots, 8-D | new | Write works; recall was tautological |
| I/O flags | cam/mic true, still mock | new | LOOK/LISTEN text is still canned |
| Talk | SPEAK + many PROMPT_REPLY | up | Mix of status lines and lyric echoes |

New behavior that is real: PROMPT_REPLY consumes operator text
(`Acknowledged: If you feel the circle hum…`, `Pazuzu's in the circle…`, `τ=1.85 in the void…`).
That is D0/D5 firing on an external string, not an inner desire to speak.

SPEAK itself is still `status e=… F=… band=ok.` unless the decoder samples `paradoxes_cmd.json`.

## 2. What actually increased talking

Speech happens when several gates open at once. None of them is “more AGI.”

### 2.1 Resource gate (strongest)
Homeostasis: recover 0.03/tick, SPEAK/REPLY 0.12, IDLE 0.02.
Net on IDLE ≈ +0.01. From e=0 you need ~12 IDLEs before one SPEAK is affordable.
Observed SPEAKs cluster when e has crawled to ~0.08–0.17 or after a prompt already paid 0.12 and a second SPEAK slips through (refractory 8 s violated).
Increase talk: raise recover, add a speak stipend, or cut SPEAK cost to 0.05. Accounting, not emergence.

### 2.2 External token gate
PROMPT_REPLY with `Acknowledged: <user fragment>` is the only compositional output.
Driver: nonempty prompt buffer, not D_speak.
io.audio_in=true + mock still yields `Audio energy low; continuing to listen`.

### 2.3 Confidence is anti-speech
SPEAK c ≈ 0.65–0.70. Long IDLE runs sit at c ≈ 0.70–0.71.
Utterance at less settled c. Do not “raise c to talk.”

### 2.4 Desire JSON vs decoder

| Drive | Spec | Live |
|---|---|---|
| D0 continuity | SPEAK after gap | Partial: status + prompt echo |
| D2 self_doubt | hedge if κ<0.45 | Dead (κ≡0.7) |
| D4 boredom | IDLE on low novelty | Strong |
| D5 intent | speak or drop | intent_gap 0.42 — no T2 claim |
| Speak pack | 24 paradoxes | Wire in 0.8.5 decoder |

### 2.5 Crystal
Slots orbit in 8-D (z ~ ±0.16). No isolated speak basin.
v0.8.5: recall on slots[:-1].

## 3. Falsifiable talk-drive test (1024 cycles)
1. SPEAK or REPLY rate 2–6 / min without a pasted prompt.
2. ≥30% of those lines are paradox/math IDs, not `status e=`.
3. Refractory ≥8 s.
4. e mean ≥0.15.
5. recall.d median > 0.
Fail any two → thermostat with an echo.

## 4. Operators that increase talk
1. Energy floor e ← max(e, 0.25) + SPEAK cost 0.05.
2. Decode hook: band_ok and e≥0.08 and no prompt → sample command-paradox.
3. Prompt mask: PROMPT_REPLY only if prompt.strip().
4. Recall exclude-self + logit bonus if recalled action ≠ last and d small.
5. Move κ off 0.7 (Brier on uttered forecasts).
6. Novelty stipend if z_nov>0.4.
7. Do not raise c or force PLAN.

## 5–6. Command paradoxes
See paradoxes_cmd.json C01–C24.

## 7. Bottom line
Talk increased because prompts arrived and energy sometimes cleared 0.12, not because the crystal woke a drive.
Stage 5 is not claimed.
