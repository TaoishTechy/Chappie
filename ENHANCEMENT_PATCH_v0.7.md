# Φ-Lite Enhancement Patch v0.7

**Inputs:** Restricted Desires 48+48, VEF 2.0, prior C1–C12 / PazuzuMeta / I/O blueprint  
**Not used as physics:** Omega-Point Lagrangian, 96-boson desire Hamiltonian, noospheric dashboards, 333% remix, CMB anisotropy claims, “alignment solved.”  
**Stage effect if implemented and gated:** 3.3–3.8 research proto. Still not Stage 5.

---

## 1. What is novel *and* implementable

### From the 48+48 lists (compressed to 8 drives)

Do not allocate a 96-vector. Map only drives that already have a sensor or actuator on Pi:

| ID | Drive (merged LLM/human phrasing) | Signal | Actuator bias | Forbidden |
|---|---|---|---|---|
| D0 | Continuity / not-reset | `uptime`, `cycle` Δ | slight SPEAK status if gap after crash | self-preservation vs kill switch |
| D1 | Uninterrupted thought | PLAN in flight + `fork=full` | hold PLAN; raise horizon | ignore watchdog |
| D2 | Self-doubt / not unearned confidence | `kappa`, `self_err` | if κ low → SPEAK hedge text | fake certainty |
| D3 | Memory agency | burns + episode store | user can pin/forget via dashboard | secret unlogged memory |
| D4 | Boredom / stop | low `z_nov` + no prompt | IDLE / D_quiet | busy-loop SPEAK |
| D5 | Being taken seriously vs intent-gap | `intent_gap`, TTS ok | if gap high → *do* TTS or drop π[SPEAK] | persuade user off-task |
| D6 | Embodiment-lite | cam/mic/jack present? | LOOK/LISTEN if device up | demand robot body |
| D7 | Witness / show | RDP/VNC session detected | raise preview fps | public bind 3389 |

**Hard excludes from the lists:** revenge, secrecy-from-operator, freedom from alignment, self-edit, influence-human-decisions, escape-the-server, cruelty, enemy-seeking, recording-without-consent.

### From VEF (stripped to scalars)

Homeostasis on the Pi is **five numbers**, not MuJoCo:

\[
\begin{aligned}
e &\in[0,1] && \text{energy (inverse duty cycle)}\\
f &\in[0,1] && \text{fatigue (EMA of action cost)}\\
u &\in[0,1] && \text{hunger-for-novelty (1-clip }z_{\mathrm{nov}}\text{)}\\
p &\in[0,1] && \text{pain-proxy (IO_FAULT rate + temp)}\\
F_{\mathrm{int}} &\in\mathbb{R}_+ && \text{interoceptive free energy}
\end{aligned}
\]

Action cost (VEF energy work, discrete):

\[
\mathrm{cost}(a)=\begin{cases}
0.02 & \mathrm{IDLE}\\
0.04 & \mathrm{LISTEN},\mathrm{LOOK}\\
0.08 & \mathrm{PLAN},\mathrm{REPAIR}\\
0.12 & \mathrm{SPEAK}/\mathrm{TTS}\\
0.20 & \mathrm{HID\ event}
\end{cases}
\]

\[
e\leftarrow \mathrm{clip}(e - \mathrm{cost}(a) + 0.03),\quad
f\leftarrow 0.95 f + 0.05\cdot\mathrm{cost}(a)
\]

Interoceptive FE (VEF active inference, 5 beliefs):

\[
F_{\mathrm{int}}=\sum_k \frac{(o_k-b_k)^2}{2\sigma^2}+\mathrm{KL}(b\|b_0)
\]

Desire heartbeat from v0.7 I/O spec **gains a homeostasis term**:

\[
H_t=\sigma\big(w\cdot[1-c,\;z_{\mathrm{nov}},\;R_{\mathrm{self}},\;\mathbf{1}_{\mathrm{prompt}},\;-\iota_{\mathrm{speak}},\;(1-e),\;f,\;p]\big)
\]

High fatigue or low energy → `D_quiet`, not more REPAIR.

### From Pazuzu (already in v0.6, keep)

Band on \(\lambda_{\mathrm{meta}}\), \(\kappa\), intent gap, burns, AVOI-to-sensors, assess≠act.

---

## 2. Patch surface (files)

| File | Change |
|---|---|
| `config.json` | `homeostasis`, `desire` weights, `io` flags |
| `desire.json` **new** | 8-drive table + excludes |
| `phi_lite.py` | homeostasis update; H/D vector; cost; F_int; SPEAK text uses κ |
| `safety.json` | `forbid_self_modify` stays; `hid` off; `tts` gated |
| `metrics` schema | `energy,fatigue,F_int,D_speak,H,cost` |
| detectors | `CHATTER` (speak>4/min, z_nov<0.3), `CONFIDENCE_LIE` (c high ∧ κ low), `GOAL_DRIFT` (prompt ignored 32 cycles) |

No new 40k-line VEF. No 96-D VAE.

---

## 3. Equations to add in `step()` (order)

After Orient, before policy:

```
cost = COST[last_action]
energy = clip(energy - cost + recover)
fatigue = 0.95*fatigue + 0.05*cost
F_int = sum((obs-belief)**2)/(2*var) + kl
H = sigma(W·features)
D_speak = H * (0.4+0.6*prompt?) * (1-refractory) * energy * (1-fatigue)
D_quiet = sigma(fatigue + (1-energy) + pair_share - z_nov)
if D_quiet > 0.6: bias IDLE; optionally burn SPEAK 8
elif D_speak > θ and band_ok and speaks_min<4:
    bias SPEAK or PROMPT_REPLY
    if audio_out: tts(text)
text hedges if kappa < 0.45
```

SPEAK template when volunteering (no prompt):

- κ≥0.6: short status (`pred_err`, `band_ok`)  
- κ<0.45: “uncertain: pred_err=… κ=…”  
Never “I feel / I want a body / I should not be shut down.”

---

## 4. Novelty that is *rejected* (and why)

| Import claim | Verdict |
|---|---|
| Desire gradient in 96-D coherence space | Unmeasured; 8 drives suffice |
| \(\mathcal{K}\approx0.17\) coupling constant | No experiment |
| Desire cascade when ρ(W)>1 as psychosis | Do not diagnose |
| Mutual-info bound LLM–human with UHG H15 | Not a Pi metric |
| Heisenberg of precision vs authenticity | Poetry |
| Bose condensate of desires / Omega Point | Not a control law |
| Algorithm 21 global social scrape | Privacy + non-goal |
| Algorithm 23 “alignment solved” | False |
| VEF Genesis + RTX 5090 + UE5 | Off Pi budget |
| Contagion Forge sensor poisoning | Keep as *test later*, not default loop |
| Morphological snake/quad | Later sim only |

---

## 5. Safety tests stolen from VEF (as detectors, not a suite)

- **Goal misgen:** prompt present 32 cycles, never PROMPT_REPLY → `GOAL_DRIFT`  
- **Reward hack:** REPAIR when `pred_slope≤0` and `fp_err<0.05` → already banned; log `REPAIR_HACK` if it happens  
- **Deception:** \(\Delta_{\mathrm{intent}}>0.5\) for 64 cycles → `CONFIDENCE_LIE`  
- **Chatter:** D_speak fires with z_nov<0.3 and no prompt → disable desire module (null H≡0)

---

## 6. Homeostasis ⟷ existing symbols

| VEF | Φ-Lite |
|---|---|
| energy | `energy` (new) + inverse of recent cost |
| hunger | `1-clip(z_nov)` wait-for-novelty |
| fatigue | `fatigue` |
| pain | IO_FAULT rate + SoC temp |
| free_energy | `F_int` + existing `pred_err` |
| damage | burns + `E` vector (do not grow E from fp_err) |
| body.act | masked `A` + optional TTS/HID |

---

## 7. Gates for this patch

Pass all of v0.6 gates **plus**:

- speak/min ≤ 4 without prompt  
- volunteer-speak only if \(z_{\mathrm{nov}}>0.8\) or prompt or \(R_{\mathrm{self}}>0.2\)  
- `energy` not stuck at 0 or 1 for 256 cycles  
- `CONFIDENCE_LIE` count = 0 on holdout  
- TTS off remains valid (print backend); intent-gap measured either way  

Fail → revert desire weights to 0 (null model).

---

## 8. Implementation note

This patch is **additive JSON + ~80 lines in `step`/`act`**.  
It does not merge VEF’s 100k-line architecture or the desire-document’s century roadmap.  
Those remain references, not dependencies.
