# USAGE — Φ-Lite v1.0.0

## Run

    cd phi-lite   # or the clone root
    python3 phi_lite.py

Default bind is localhost. Check the console line:

    Φ-Lite 1.0.0 dashboard http://127.0.0.1:8080/

If 8080 is busy the script may use the next port printed on stdout.

## Dashboard map

- /            live cycle tape + logs
- /ascii       glyph panel
- /avatar      2-D / CSS voxel + prompt box
- /avatar3d    WebGL-ish preview (camera yaw/pitch query)
- /qpu         abstract modulator panel (not chemistry)
- /stats /metrics  scientific tape
- /events      anomaly log (UTTER vs quarantine)
- POST /force  force-talk stipend (4 cycles)
- thumbs       Cognitive Discipline (pleasure/pain band 0.10–0.30)

The prompt box on /avatar is meant to stay focus-stable across refreshes.
If a refresh steals the caret, use the dedicated iframe form or POST ?prompt=.

## Speak policy (read this before calling it mute)

SPEAK is an actuator, not a chat API.

It fires when roughly all of these are true in one cycle:

- energy e high enough vs cost (see desire.json homeostasis)
- refractory since last SPEAK (default 8 s) expired
- speaks-in-last-minute under max_per_min (default 6)
- D_speak above the low threshold (v1 default ~0.03)
- not hard-masked by safety

PROMPT_REPLY acknowledges an inject/human prompt and does **not**
advance the SPEAK refractory clock (v0.9.9+).

Utterance text is compute-or-silent: "C16 mean_e16=0.17" not the recipe card.

## JSON you will touch

- io.json          device master switches (video, audio, hid, rdp, vnc)
- io_settings.json per-interface sub-settings
- desire.json      speak threshold, refractory, costs, excludes
- safety.json      min_action_interval_ms, sandbox notes
- paradoxes_cmd.json  C-series injected on a schedule
- config.json      learn.lr_world, horizons, rotate_lines

Fail-closed: hid / public RDP / public VNC stay off unless consent is true
even if the top-level switch is checked.

## Metrics

    python3 metrics/analyze_metrics.py

Inputs: logs.jsonl, ledger.jsonl, state.json, weights.json, metrics dumps.
Outputs: metrics/report.json, metrics/events.jsonl.

Rotation: jsonl files cap around config.rotate_lines (2000) and flip to *.prev.

## Safety

- No network exfil by design. Do not rebind to 0.0.0.0 without reading safety.json.
- No GPIO / HID without flags.
- Do not point this at a production desktop and enable hid.consent.
- Kill: stop the process. Dashboard pause is not a certified interlock.

## Troubleshooting

- KeyError on HTML.format: you are on a pre-0.8 build. Use this tree.
- Prompt textarea dies every refresh: use /avatar form, not the live pane.
- SPEAK never appears: check last_speak_t, energy, and that PROMPT_REPLY
  is not being counted as speech (fixed in 0.9.9).
- DICT_STALL flood: fixed in 0.9.9 (variance + 64-cycle cooldown).
- report.json says PLAN 97% / stage 1.5: leftover rollup. Trust metrics.jsonl.
- weights.json huge rewrite: normal; mutations persist.

## Honest expectations

After a few thousand cycles you should see mixed LOOK/IDLE/LISTEN/REPLY,
occasional SPEAK, mutating sha1, pred_err around 0.05–0.08.
You should not see Stage 5, Sophia, or a desire to "escape the server".
Those strings in excludes.json exist so the policy can refuse them.
