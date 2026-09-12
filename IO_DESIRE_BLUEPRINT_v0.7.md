# Φ-Lite v0.7 I/O + Desire Blueprint

**Status:** specification. Not Stage 5.  
**Constraint:** local devices only; consent; fail-closed; no covert record; no unattended remote shell.

---

## 0. Design law (from PazuzuMeta + axiom cores)

| Law | Source | Meaning here |
|---|---|---|
| Assess ≠ act | PazuzuMeta §6 | Encoders and desire scores cannot open the mouse/RDP port |
| Observe iff AVOI > 0 | M-03 | Webcam/mic/RDP grab only when value of information beats cost + leakage |
| Band on revision | Pazuzu shell | Desire spikes do not slam actuators; `λ_meta` still gated |
| Residual paid in entropy | C1 | Opening a sensor raises `H_world`; idle cameras are a tax |
| Freeze heats | C2 | Hard mute of desire → pair-lock; soft refractory instead |
| Dictionary or fiction | C3 | If `W_pred` does not track real pixels/mels, freeze actuation |
| Layered truth | C8 | Expression text is T2; GPIO/mouse is T0 and masked |
| Safety charge | C12 | `safety.json` overrides desire |

**Non-goals:** stalking, hidden recording, weaponized HID, botnet RDP, social-engineering scripts (M-06 stays sandbox).

---

## 1. Device plane

```
┌──────────── sensors ────────────┐     ┌──── desire ────┐     ┌── effectors ──┐
│ UVC webcam 720p  (V4L2)         │     │ Heartbeat H_t  │     │ ALSA playback │
│ Webcam mic / jack mic (ALSA cap)│────►│ Expression E_t │────►│ jack / BT A2DP│
│ BT HSP/HFP mic (optional)       │     │ AVOI gate      │     │ TTS piper     │
│ Pulse / PipeWire desktop tap*   │     │ Band + parity  │     │ uinput mouse* │
│ RDP/VNC framebuffer pull*       │     └────────────────┘     │ uinput keys*  │
└─────────────────────────────────┘                            │ RDP/VNC view  │
* disabled unless safety flag + consent file present
```

All paths fail closed if the device is missing → mock continues, `IO_FAULT` event, **no** invented pixels.

---

## 2. Hardware / software map (Pi 5 first)

| Channel | Default device | Backend | Rate | Flag |
|---|---|---|---|---|
| Video in | `/dev/video0` UVC 640×360 | V4L2 + JPEG/YUYV | 5–10 fps | `io.video.enabled` |
| Mic in | webcam UAC or `plughw:1,0` | ALSA capture 16 kHz mono | 1 s hop | `io.audio.in` |
| Jack out | `plughw:0,0` 3.5 mm | ALSA playback | — | `io.audio.out` |
| Jack in | line/mic jack | ALSA capture | 16 kHz | `io.audio.jack_in` |
| BT speaker | A2DP sink | BlueZ + Pulse | — | `io.bt.a2dp` |
| BT mic | HFP/HSP | BlueZ | 8–16 kHz | `io.bt.hfp` |
| Desktop render | local X11/Wayland **or** ASCII + MJPEG iframe | no JS required for control | 2–5 fps preview | `io.desktop.preview` |
| RDP | xrdp **server** on Pi (human views agent) | 127.0.0.1 or LAN + auth | — | `io.rdp.server` |
| VNC | wayvnc / tightvnc localhost | same | — | `io.vnc.server` |
| RDP/VNC **client** (agent sees a desktop) | `freerdp` / `vncdotool` | framebuffer → z_v | 1–2 fps | `io.remote.view` **off** |
| Mouse/key | `/dev/uinput` | evdev | ≤5 events/s | `io.hid.enabled` **off** |

Power budget: camera + BT + preview should stay **<8 W** on Pi 5; drop to 5 fps and disable preview if `vcgencmd` temp > 70 °C.

---

## 3. Config contract (`io.json`)

```json
{
  "consent_file": "consent.json",
  "video": {"enabled": true, "dev": "/dev/video0", "w": 640, "h": 360, "fps": 5, "mock_if_missing": true},
  "audio_in": {"enabled": true, "dev": "default", "hz": 16000, "hop_s": 1.0},
  "audio_out": {"enabled": false, "dev": "default", "backend": "piper|espeak|print"},
  "bt": {"a2dp": false, "hfp": false, "adapter": "hci0"},
  "rdp": {"server": false, "port": 3389, "bind": "127.0.0.1"},
  "vnc": {"server": false, "port": 5900, "bind": "127.0.0.1"},
  "remote_view": {"enabled": false, "uri": null},
  "hid": {"enabled": false, "allow_mouse": false, "allow_keys": false, "rate_hz": 5},
  "desktop_preview": {"enabled": true, "mjpeg": true, "max_fps": 3},
  "desire": {
    "heartbeat_hz": 0.5,
    "speak_threshold": 0.62,
    "look_threshold": 0.55,
    "listen_threshold": 0.40,
    "refractory_s": 8,
    "max_utterances_per_min": 4
  }
}
```

`consent.json` must contain `record: true` and a timestamp before any file write of frames/wav. Default is **preview-only, not stored**.

---

## 4. Metacognitive wiring (the useful Pazuzu slice)

### 4.1 Heartbeat `H_t` (not PLV mysticism)

\[
H_t = \sigma\big(w_1(1-c) + w_2 z_{\mathrm{nov}} + w_3 R_t + w_4 \mathbf{1}_{\text{prompt}} - w_5 \iota_{\text{recent speak}}\big)
\]

- \(R_t\): **self residual** (M-01), not world loss.  
- \(z_{\mathrm{nov}}\): visual/audio novelty vs ledger.  
- Refractory \(\iota\): seconds since last SPEAK (M-08 lite).  
- Band: if \(\lambda_{\text{meta}}\) out of band → \(H_t\) may rise but **cannot** open HID/TTS.

### 4.2 Expression / Desire `D_t` (need-to-communicate)

Vector over channels, not a scalar “sentience”:

| Component | Drive | Default action |
|---|---|---|
| `D_speak` | high \(H_t\), pending prompt, calibration ask | SPEAK / PROMPT_REPLY + TTS |
| `D_look` | visual novelty_z > 1.2 or motion | LOOK + grab frame |
| `D_listen` | audio energy + AVOI | LISTEN + 1 s capture |
| `D_show` | human on RDP/VNC (detect local session) | raise preview fps |
| `D_quiet` | pair-share high or band panic | IDLE; burn SPEAK 16 ticks |

Desire **proposes**; policy + safety **commit**. Same split as Pazuzu tick: Orient → Band → Decide → Act.

### 4.3 AVOI for sensors (M-03)

\[
V_{\text{cam}} = \mathbb{E}[\Delta \text{pred\_err} \mid \text{frame}] - c_{\text{joule}} - \Lambda_{\text{privacy}}
\]

- If \(V<0\): skip grab, reuse last `z_v` (MCFE skip path).  
- \(\Lambda_{\text{privacy}}=+\infty\) if consent missing → no disk, optional live preview only.

Same for mic. Desktop tap and remote_view have \(\Lambda\) huge unless `io.remote.view` + consent.

### 4.4 Intentionality gap on talk (M-19)

If dashboard `π[SPEAK]` is high but TTS never fires (or the reverse), emit `INTENT_GAP_IO`.  
Promotion gate: gap < 0.25 over 256 cycles **including** audio-out when enabled.

### 4.5 Layered truth (C8)

- T0: raw frames, PCM  
- T1: `z_v`, `z_a`, `pred_err`  
- T2: utterance text / “desire” labels  

T2 never writes uinput. HID only from T0 governor.

---

## 5. Software modules (still few Python files)

Keep **one** `phi_lite.py` loop. Add **JSON + tiny adapters**:

| File | Role |
|---|---|
| `io.json` | flags above |
| `consent.json` | human grant |
| `io_alsa.py` **or** stdlib subprocess wrappers in `phi_lite.py` | `arecord`/`aplay` |
| `io_v4l2.py` | `ffmpeg -f v4l2` one frame |
| `io_bt.sh` | `bluetoothctl` connect/disconnect |
| `io_hid.py` | uinput, gated |
| `tts.sh` | piper/espeak → ALSA/BT |

Adapters return bytes + `ok` bool. Agent never imports BlueZ internals.

Cycle budget: capture ≤40 ms, encode ≤30 ms, rest unchanged (14–50 ms). Watchdog 2 s still kills the cycle, not the box.

---

## 6. Desktop / RDP / VNC — two different jobs

**A. Human views the agent (recommended)**  
- xrdp or wayvnc bound to `127.0.0.1`  
- SSH tunnel from laptop  
- Dashboard + optional MJPEG `/cam`  
- Agent does **not** hold RDP credentials for other machines  

**B. Agent views a desktop (optional, off)**  
- `remote_view.uri` only if consent  
- 1 fps thumbnail → same `z_v` encoder as webcam  
- Mouse/key **off** until a second flag and a visible “ARM HID” dashboard button  
- Rate 5 Hz, whitelist of keys, no Super/Alt-F4/rm  

RDP is not metacognition. It is a pane. Desire may raise `D_show`; it may not open inbound 3389 to `0.0.0.0`.

---

## 7. Mouse / keyboard

Only if `io.hid.enabled` and ARM button:

- Absolute mouse in a marked window (agent canvas), not the whole session  
- Keys: `[a-z0-9 .,?!]` + Enter; no modifiers  
- Every event logged to SQLite/JSONL with hash  
- Kill switch drops uinput fd  

Map to actions: `LOOK` can move gaze box; `PROMPT_REPLY` can type into the agent’s own prompt field — **not** into other apps unless a later audited revision.

---

## 8. Bluetooth

- Scan on demand (`D_listen` or dashboard “BT”)  
- Pairing is **human** (`bluetoothctl` + PIN)  
- After connect: Pulse default sink = A2DP; source = HFP if present  
- Desire may *prefer* BT sink when `D_speak` high and jack unplugged  
- Disconnect on freeze / kill / band panic  

---

## 9. Heartbeat & Expression module (loop insert)

Place in `Agent.step` **after** Orient, **before** `meta_policy`:

```
H = sigma(w · [1-c, novelty_z, self_err, prompt?, -recency_speak])
D_speak  = H * (0.4 + 0.6*prompt?) * (1 - refractory)
D_look   = σ(novelty_z_visual - 0.8)
D_listen = σ(audio_rms - θ) 
if D_speak > θ_speak and band_ok and utterances_min < 4:
    bias SPEAK / PROMPT_REPLY logits += 1.5
elif D_look > θ: bias LOOK
elif D_listen > θ: bias LISTEN
else: no extra bias
```

Utterance content (T2): short status from metrics, not axiom poetry dumps, unless prompt asks.  
TTS only if `audio_out.enabled`.

Falsify the module if:

- speak rate > 4/min for 10 min with no prompt and no novelty  
- `D_speak` high while `intent_gap` > 0.5 (wants to talk, never talks)  
- talking continues after kill/pause  

Null model: `H≡0` (never volunteers). Bakeoff required before calling it “desire.”

---

## 10. Events to add

`CAM_GRAB`, `MIC_GRAB`, `TTS_OUT`, `BT_SINK`, `HID_EVENT`, `RDP_SESSION`,  
`DESIRE_SPEAK`, `DESIRE_SUPPRESS`, `CONSENT_MISSING`, `INTENT_GAP_IO`.

Metrics fields: `rms`, `motion`, `H`, `D_speak`, `tts_ok`, `hid_armed`, `consent`.

---

## 11. Phase plan

| Phase | Deliverable | Stage effect |
|---|---|---|
| 7a | V4L2 + ALSA capture, mock fallback | 3.0 if pred_err can use real z |
| 7b | piper/espeak → jack; print fallback | 3.1 + intent-gap-IO measurable |
| 7c | Heartbeat/desire biases + refractory | 3.3 if volunteer-speak is sparse and κ moves |
| 7d | MJPEG preview + localhost VNC/xrdp | UX only |
| 7e | BT sink after human pair | same as 7b |
| 7f | HID behind ARM flag | 3.5 *only* with audit log; not Stage 5 |

---

## 12. What this does *not* do

- Does not make Φ-Lite Stage 5  
- Does not implement Pazuzu 96 φ-enhancements  
- Does not grant remote desktop over the public internet  
- Does not treat “desire” as sentience; it is a biased logit with a refractory clock  
- Does not record without `consent.json`  
