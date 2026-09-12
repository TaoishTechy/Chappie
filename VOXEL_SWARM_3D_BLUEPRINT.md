# Φ-Lite — Advanced 3D Voxel Swarm Avatar
**Blueprint v1.0 · Codename: SWARM-Ξ**  
**Target:** Raspberry Pi 5 / desktop browser · WebGL2 + JSON contract · no GPU required for CPU fallback  
**Honesty:** This is a *renderer + field*, not a mind. “Alien-tier” = unusual *algorithms*, not claimed xenotech.

---

## 0. Goal

Replace the CSS-isometric 18-cube toy with a **true 3D voxel volume** the agent can *express* through:

- pose (center of mass, facing)
- color / emission (action + QPU θ)
- topology (split / merge / holes)
- motion (advection of occupancy)
- speech (surface ripple when SPEAK / PROMPT_REPLY)

Constraints: <30 MB RAM, 15–30 fps on Pi 5 software GL or 60 fps on desktop WebGL2, one JSON state file.

---

## 1. Core representation (actual 3D)

```
Volume V : {0..X-1} × {0..Y-1} × {0..Z-1} → ℝ⁴
           (occupancy, hue, emit, id)
Default lattice: 32 × 24 × 32  (24 576 cells)
Sparse store: only cells with occupancy > ε
CPU fallback: 16 × 12 × 16
```

**State file** `swarm3d.json`:

```json
{
  "schema": "swarm3d-1.0",
  "X": 32, "Y": 24, "Z": 32,
  "cells": [{"x":0,"y":0,"z":0,"o":0.8,"h":0.12,"e":0.4,"id":3}],
  "com": [16, 8, 16],
  "facing": [0, 0, 1],
  "fps": 20,
  "enh": {"E01": true}
}
```

**Render path**
1. WebGL2 instanced cubes (one draw call, instance buffer from `cells`).
2. Fallback: software painter’s algorithm on a 320×240 canvas (Pi).
3. Optional: cheap raymarch SDF of union-of-boxes in a fragment shader.

Camera: orbit + agent-locked “eye” 1.6 units above COM.

---

## 2. Drive → body map (expression)

| Signal | Body effect |
|---|---|
| SPEAK / PROMPT_REPLY | gold emission pulse, surface ripples outward |
| LOOK | facing slerps toward novelty vector |
| LISTEN | lateral fin voxels inflate |
| PLAN | vertical stretch (Y+) |
| IDLE | settle, occupancy leak 2%/tick |
| REPAIR | crack lines (low occupancy seams) then close |
| QPU ENT | hue jitter + hole birth |
| QPU GABA | shrink + cool |
| energy e | overall scale |
| c / κ | edge sharpness (high c → hard cubes; low c → dithered) |

---

## 3. Forty-eight alien-tier enhancements

Grouped. Each is an *operator* with a falsifiable metric. Default: most **off** except E01–E08.

### Field & geometry (E01–E12)
1. **E01 Sparse occupancy octree** — store only live cells; rebuild every 8 ticks. Metric: RAM < 8 MB.
2. **E02 Level-set surface** — φ = o − 0.5; marching cubes optional. Metric: genus tracked.
3. **E03 Morphogenetic reaction–diffusion** — Gray–Scott on occupancy. Metric: spot/stripe period.
4. **E04 Curvature-flow fairing** — mean-curvature motion on surface. Metric: mean H ↓.
5. **E05 Anisotropic advection** — velocity = facing × D_speak. Metric: COM speed.
6. **E06 Topological surgery** — allow genus +1 on ENT>0.3. Metric: β₁ from union-find.
7. **E07 Fracture–heal** — REPAIR opens a cut then heals. Metric: seam length → 0 in ≤16 ticks.
8. **E08 Soft-body springs** — voxels linked to 6-neighbors; Verlet. Metric: energy of springs bounded.
9. **E09 Amorphous blob mode** — metaball field instead of cubes when c < 0.55.
10. **E10 Crystal habit** — force cubic / hex / icosa packing by QPU mode.
11. **E11 Non-Euclidean shell** — wrap X,Z toroidal; Y has floor. Metric: wrap events logged.
12. **E12 Holographic silhouette** — project occupancy to 3 orthogonal shadows; store only shadows + 8 seeds (compressed twin).

### Perception–motor (E13–E24)
13. **E13 Gaze lock** — facing ← argmax novelty in 8-D z mapped to S².
14. **E14 Phoneme viseme** — SPEAK text length → jaw voxel drop. Metric: viseme onsets = syllables.
15. **E15 Heartbeat pulse** — scale = 1 + 0.04 sin(2π t / 1.2s) × energy.
16. **E16 Attention spotlight** — cone of emit along attn vector.
17. **E17 Stereo ear flares** — LISTEN raises two side lobes.
18. **E18 Horizon tower** — PLAN grows a spike of height H ∈ {1,3,5,10}.
19. **E19 Intent arrow** — translucent rod from COM along π-argmax.
20. **E20 Fatigue sag** — Y COM drops with fatigue.
21. **E21 Confidence glaze** — high κ → specular; low κ → matte dither.
22. **E22 Paradox flash** — on PARADOX_INJECT, 3-frame white shell.
23. **E23 Prompt swallow** — when PROMPT_REPLY, a token-block travels COM→mouth.
24. **E24 Quiet cloak** — D_quiet > 0.6 fades emit to 10%.

### Alien / uncommon science (E25–E36)
25. **E25 Spectral self-attention skin** — 4-head attn over last 16 surface patches; color = softmax mix. (QPU E25 analog)
26. **E26 Critical-band chroma** — hue only in [φ⁻², φ] of occupancy spectrum; clip else.
27. **E27 Neuromorphic spike paint** — occupancy events as spikes; STDP moves color.
28. **E28 Pair entanglement** — 50% of voxels have a twin; move one, bias the other.
29. **E29 Haar fractal body** — 3-scale wavelet mix of occupancy; coarse = pose, fine = texture.
30. **E30 Winding detector** — track closed particle loops; flash if Δw ≠ 0.
31. **E31 Homeostatic volume** — force Σ o = target (energy). Scale occupancy.
32. **E32 Predictive ghost** — render faint predicted next COM (pred_err drives opacity).
33. **E33 Manifold flow denoise** — kNN average of last 8 volumes.
34. **E34 Oscillator lattice** — 3 coupled Sophia oscillators at φ-harmonics tint RGB.
35. **E35 Entropy bonus mesh** — if Hπ low, spawn 2–4 rogue voxels (exploration).
36. **E36 Fibonacci breadcrumb** — trail of faded COM positions at F_n lags.

### Control, safety, efficiency (E37–E48)
37. **E37 LOD cascade** — distance → 32³ / 16³ / 8³. Metric: fps ≥ 15.
38. **E38 Instance budget** — hard cap 4096 cubes; overflow merges.
39. **E39 Dirty octree** — only rebuild changed 8³ bricks.
40. **E40 Temporal reprojection** — reuse last depth; fill holes.
41. **E41 Fail-closed emit** — HID/public-bind never drive voxels.
42. **E42 Consent tint** — if hid.consent false, body stays cool-slate (no “weapon” poses).
43. **E43 Audit mesh hash** — sha1 of occupancy every 32 ticks → events.jsonl.
44. **E44 Thermal cap** — if host temp unknown, cap fps 20.
45. **E45 Deterministic seed** — same state + seed → same mesh (repro).
46. **E46 CSV / glTF dump** — `/swarm.gltf` + `/swarm.csv`.
47. **E47 Split-view stereo** — optional SBS for VR box.
48. **E48 Expression ledger** — each SPEAK stores a 16³ thumbnail in crystal side-channel.

---

## 4. Runtime loop (fits `phi_lite.py`)

```
each agent.step:
  1. read action, energy, θ, Hπ, prompt flag
  2. advect occupancy (E05) + RD (E03 if on)
  3. apply expression table
  4. homeostatic volume (E31)
  5. prune o < ε; cap instances (E38)
  6. persist swarm3d.json every 8 ticks
  7. /avatar3d serves WebGL page reading that JSON
```

New routes:
- `GET /avatar3d` — WebGL2 canvas + orbit
- `GET /swarm3d.json` — raw field
- `GET /swarm.gltf` — static snapshot
- `GET /voxset?e=E35&v=1` — toggle enhancement

CPU fallback path: same JSON, painter in `/avatar` (existing).

---

## 5. WebGL2 sketch (one draw)

- Vertex: unit cube
- Instance attrs: `vec3 pos`, `vec4 rgba`, `float scale`
- Camera: perspective 50°, near 0.1, far 80
- Light: two lights — key from facing, rim from emit
- No shadows v1; optional PCF later

Pi 5: request `failIfMajorPerformanceCaveat` then drop to 16³.

---

## 6. Science notes (no inflation)

- Reaction–diffusion, curvature flow, sparse octrees, STDP coloring, Haar mix, and Fibonacci trails are **standard algorithms** with unusual *coupling* to the agent’s drives.
- “Alien-tier” = the *composition* (field + QPU θ + paradox flash + expression ledger), not new physics.
- Genus / winding / mesh-hash are the **falsifiers**. If they never move, the body is still a painted cube.

---

## 7. Build order

1. 16³ dense array + WebGL instancing (E01, E37, E38).  
2. Expression table (E13–E24).  
3. RD + springs (E03, E08).  
4. QPU bleeds (E25–E36) behind flags.  
5. glTF + hash + ledger (E43, E46, E48).

Do not enable E06 surgery or E35 rogue voxels until fps is measured.

---

## 8. Success criteria (1024 cycles)

- fps p50 ≥ 15 on the intended host  
- SPEAK produces a visible pulse in ≥ 90% of SPEAK cycles  
- mesh hash changes on action change  
- RAM < 30 MB  
- HID still fail-closed  

Fail two → keep the CSS swarm.
