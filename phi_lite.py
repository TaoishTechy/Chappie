#!/usr/bin/env python3
"""Φ-Lite / Chappie v1.0.0 — slim adaptive loop.

sense → predict → act → residual → mutate → remember → repeat
Does not claim AGI. Honest stage ~2.9 / 5 on instrumented runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import threading
import time
from collections import deque
from html import escape as html_escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
VERSION = "1.0.0"


def load_json(name, default=None):
    path = ROOT / name
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(name, data):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def rotate_if_needed(path: Path, cap: int):
    if not path.exists() or cap <= 0:
        return
    try:
        n = sum(1 for _ in path.open("r", encoding="utf-8"))
    except OSError:
        return
    if n < cap:
        return
    prev = path.with_suffix(path.suffix + ".prev")
    try:
        if prev.exists():
            prev.unlink()
        path.replace(prev)
    except OSError:
        pass


def append_jsonl(name, rec, rotate=True):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = int(CFG.get("rotate_lines", 2000))
    try:
        if rotate:
            rotate_if_needed(path, cap)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        return None
    except OSError as exc:
        return f"{type(exc).__name__}:{exc}"


CFG = load_json("config.json")
ARCH = load_json("architecture.json")
SAFETY = load_json("safety.json")
DESIRE = load_json("desire.json")
IO = load_json("io.json") or {}
IOSET = load_json("io_settings.json") or {}
CRYSTAL_DOC = load_json("crystal.json") or {}
UI_FEAT = load_json("ui_features.json") or {}
CMD_PX = load_json("paradoxes_cmd.json") or {}
PX_SEEDS = load_json("paradox_seeds.json") or {}
NOS = load_json("nos.json") or {}
QPU_FEAT = load_json("qpu_features.json") or {}
QPU_CAT = load_json("qpu_catalog.json") or {}
FEAT_ON = load_json("feature_flags.json") or {}
ACTIONS = ARCH.get("actions", ["IDLE", "SPEAK", "PROMPT_REPLY", "LOOK", "LISTEN", "PLAN", "REPAIR", "PAUSE"])
COST = (DESIRE.get("homeostasis") or {}).get("cost") or {
    "IDLE": 0.02, "LISTEN": 0.04, "LOOK": 0.04, "PLAN": 0.08,
    "REPAIR": 0.08, "SPEAK": 0.05, "PROMPT_REPLY": 0.05, "PAUSE": 0.01,
}
COST["SPEAK"] = 0.05
COST["PROMPT_REPLY"] = 0.05


def rng_matrix(rows, cols, seed):
    rnd = random.Random(seed)
    scale = 1.0 / math.sqrt(max(cols, 1))
    return [[(rnd.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]


def matvec(W, x):
    return [sum(row[j] * x[j] for j in range(min(len(row), len(x)))) for row in W]


def add(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0) for i in range(n)]


def mul(a, b):
    return [a[i] * b[i] for i in range(min(len(a), len(b)))]


def tanh_v(v):
    return [math.tanh(x) for x in v]


def sigmoid_v(v):
    return [1.0 / (1.0 + math.exp(-max(-20, min(20, x)))) for x in v]


def l2(v):
    return math.sqrt(sum(x * x for x in v))


def softmax(logits):
    m = max(logits)
    ex = [math.exp(min(20, x - m)) for x in logits]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def entropy(p):
    return -sum(x * math.log(max(x, 1e-12)) for x in p)


def sha1_of(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def ensure_weights():
    w = load_json(CFG["paths"]["weights"], None)
    need = CFG["dims"]["h"] + CFG["dims"]["s"] + 16
    if not w or "fusion" not in w:
        w = {
            "version": VERSION,
            "updates": 0,
            "fusion": {"W": rng_matrix(CFG["dims"]["b"], CFG["dims"]["z_v"] + CFG["dims"]["z_a"], 1), "b": [0.0] * CFG["dims"]["b"]},
            "world": {
                "Wz": rng_matrix(CFG["dims"]["h"], CFG["dims"]["coarse"], 2),
                "Uz": rng_matrix(CFG["dims"]["h"], CFG["dims"]["h"], 3),
                "Wr": rng_matrix(CFG["dims"]["h"], CFG["dims"]["coarse"], 4),
                "Ur": rng_matrix(CFG["dims"]["h"], CFG["dims"]["h"], 5),
                "Wh": rng_matrix(CFG["dims"]["h"], CFG["dims"]["coarse"], 6),
                "Uh": rng_matrix(CFG["dims"]["h"], CFG["dims"]["h"], 7),
            },
            "self": {"W": rng_matrix(CFG["dims"]["s"], CFG["dims"]["h"] + CFG["dims"]["s"], 8), "b": [0.0] * CFG["dims"]["s"]},
            "policy": {"W": rng_matrix(len(ACTIONS), need, 9), "b": [0.0] * len(ACTIONS)},
            "attn": {"W": rng_matrix(1, CFG["dims"]["h"], 10), "b": [0.0]},
            "horizon": {"W": rng_matrix(len(CFG["horizons"]), CFG["dims"]["h"], 11), "b": [0.0] * len(CFG["horizons"])},
            "conf": {"W": rng_matrix(1, CFG["dims"]["h"] + 4, 12), "b": [0.0]},
            "pred": {"W": rng_matrix(CFG["dims"]["b"], CFG["dims"]["h"], 13), "b": [0.0] * CFG["dims"]["b"]},
        }
        save_json(CFG["paths"]["weights"], w)
        return w
    w.setdefault("version", VERSION)
    w.setdefault("updates", 0)
    w.setdefault("conf", {"W": rng_matrix(1, CFG["dims"]["h"] + 4, 12), "b": [0.0]})
    w.setdefault("pred", {"W": rng_matrix(CFG["dims"]["b"], CFG["dims"]["h"], 13), "b": [0.0] * CFG["dims"]["b"]})
    return w


W = ensure_weights()


class Crystal:
    """8-D boundary memory. Not a physical hypercrystal."""

    def __init__(self, cap=256):
        self.cap = cap
        self.slots = list(CRYSTAL_DOC.get("slots") or [])
        self.last_recall = CRYSTAL_DOC.get("last_recall")

    def project(self, b):
        v = list(b[:8])
        while len(v) < 8:
            v.append(0.0)
        return [round(float(x), 5) for x in v[:8]]

    def write(self, b, action, c, energy, cycle):
        slot = {
            "c": cycle,
            "a": action,
            "conf": round(float(c), 4),
            "e": round(float(energy), 4),
            "z": self.project(b),
        }
        self.slots.append(slot)
        if len(self.slots) > self.cap:
            self.slots = self.slots[-self.cap :]

    def recall(self, b):
        z = self.project(b)
        pool = self.slots[:-1] if len(self.slots) > 1 else []
        if not pool:
            self.last_recall = None
            return None
        best, bd = None, 1e9
        for s in pool[-64:]:
            d = sum((z[i] - s["z"][i]) ** 2 for i in range(8))
            if d < bd:
                bd, best = d, s
        self.last_recall = {"d": round(bd, 5), "a": best["a"], "c": best["c"]}
        return self.last_recall

    def persist(self):
        save_json("crystal.json", {
            "version": "0.8.0",
            "kind": "virtual_memory_crystal",
            "cap": self.cap,
            "boundary_dim": 8,
            "slots": self.slots[-self.cap :],
            "last_recall": self.last_recall,
            "n": len(self.slots),
        })


class Ledger:
    def __init__(self, cap, dim):
        self.cap, self.dim = cap, dim
        self.buf = deque(maxlen=cap)

    def push(self, b, ts):
        self.buf.append({"t": ts, "b": list(b)})
        append_jsonl(CFG["paths"]["ledger"], {"t": ts, "b": [round(x, 5) for x in b]})

    def matrix(self):
        if not self.buf:
            return [[0.0] * self.dim]
        return [row["b"] for row in self.buf]


class ParadoxForge:
    """Unlimited generator: seed axioms + cycle hash + live metrics → prompt."""

    OPS = ("||x||", "sign(Δ)", "Corr", "ATE", "clip", "σ", "tr(Σ)", "Hπ", "∂E/∂t")

    def __init__(self):
        self.n = 0
        self.last = None
        self.hist = deque(maxlen=48)

    def mint(self, ag):
        seeds = PX_SEEDS.get("seeds") or (CMD_PX.get("items") or [])
        if not seeds:
            seeds = [{"id": "C00", "text": "Utter R=||Phi-F[Phi]||."}]
        base = seeds[ag.cycle % len(seeds)]
        h = hashlib.sha1(f"{ag.cycle}:{base.get('id','x')}:{ag.pred_err:.4f}".encode()).hexdigest()
        k = int(h[:8], 16)
        op = self.OPS[k % len(self.OPS)]
        tau = round((k % 97) / 97 * 2.0 + 0.1, 3)
        layer = ("T0", "T1", "T2")[k % 3]
        text = (
            f"{base.get('id','PX')}.{ag.cycle}: {op} τ={tau} {layer} "
            f"c={ag.c:.2f} e={ag.energy:.2f} pred={ag.pred_err:.3f} "
            f"{str(base.get('text',''))[:140]} "
            f"Utter one scored line then stop."
        )[:240]
        rec = {"id": f"{base.get('id','PX')}.{ag.cycle}", "text": text, "tau": tau, "op": op}
        self.n += 1
        self.last = rec
        self.hist.appendleft(rec["id"])
        append_jsonl("paradox_feed.jsonl", {"t": time.time(), "cycle": ag.cycle, **rec})
        return rec


class VoxelSwarm:
    """16×12×16 occupancy field + sparse extract. Not a game engine."""

    AX, AY, AZ = 16, 12, 16
    PAL = {
        "IDLE": (58, 85, 96),
        "LOOK": (74, 163, 255),
        "LISTEN": (125, 255, 179),
        "SPEAK": (255, 204, 68),
        "PROMPT_REPLY": (255, 136, 204),
        "PLAN": (192, 140, 255),
        "REPAIR": (255, 102, 68),
        "PAUSE": (34, 34, 51),
    }

    def __init__(self):
        n = self.AX * self.AY * self.AZ
        self.occ = [0.0] * n
        self.cells = []
        self.com = [self.AX / 2, 3.0, self.AZ / 2]
        self.facing = [0.0, 0.0, 1.0]
        self.pulse = 0.0
        self.hash = "0"
        cx, cy, cz = 8, 4, 8
        for y in range(2, 8):
            for x in range(5, 11):
                for z in range(5, 11):
                    d = ((x - cx) ** 2 + (y - cy) ** 2 * 0.6 + (z - cz) ** 2) ** 0.5
                    if d < 3.4:
                        self.occ[self._i(x, y, z)] = max(0.0, 1.0 - d / 3.4)

    def _i(self, x, y, z):
        return int(y) * self.AX * self.AZ + int(z) * self.AX + int(x)

    def tick(self, ag):
        act = ag.last_action
        rgb = self.PAL.get(act, (120, 120, 120))
        speak = act in ("SPEAK", "PROMPT_REPLY") or bool(ag.prompt.strip())
        self.pulse = 1.0 if speak else self.pulse * 0.82
        # facing
        if act == "LOOK":
            self.facing = [math.sin(ag.cycle * 0.05), 0.0, math.cos(ag.cycle * 0.05)]
        # leak + seed blob
        sx = 8 + int(2 * self.facing[0])
        sz = 8 + int(2 * self.facing[2])
        sy = 3 + (3 if act == "PLAN" else 0) - (1 if act == "IDLE" else 0)
        sy = max(1, min(self.AY - 2, sy))
        scale = 0.55 + 0.45 * max(0.05, ag.energy) + 0.25 * self.pulse
        new = [v * 0.86 for v in self.occ]
        rad = 3.2 * scale + (1.2 if speak else 0)
        for y in range(max(0, sy - 4), min(self.AY, sy + 5)):
            for z in range(max(0, sz - 4), min(self.AZ, sz + 5)):
                for x in range(max(0, sx - 4), min(self.AX, sx + 5)):
                    d = ((x - sx) ** 2 + (y - sy) ** 2 * 0.7 + (z - sz) ** 2) ** 0.5
                    if d < rad:
                        new[self._i(x, y, z)] = min(1.0, new[self._i(x, y, z)] + (1.0 - d / rad) * 0.55)
        # ear flares on LISTEN
        if act == "LISTEN":
            for dx in (-4, 4):
                xi = max(0, min(self.AX - 1, sx + dx))
                new[self._i(xi, sy + 1, sz)] = min(1.0, new[self._i(xi, sy + 1, sz)] + 0.5)
        # REPAIR seam
        if act == "REPAIR":
            for x in range(self.AX):
                new[self._i(x, sy, sz)] *= 0.3
        self.occ = new
        # homeostatic volume
        s = sum(self.occ) + 1e-6
        tgt = 90.0 * (0.4 + ag.energy)
        g = min(1.4, max(0.55, tgt / s))
        self.occ = [min(1.0, v * g) for v in self.occ]
        # COM
        mx = my = mz = w = 0.0
        sparse = []
        cap = 1200
        for y in range(self.AY):
            for z in range(self.AZ):
                for x in range(self.AX):
                    o = self.occ[self._i(x, y, z)]
                    if o < 0.28:
                        continue
                    mx += x * o
                    my += y * o
                    mz += z * o
                    w += o
                    if len(sparse) < cap:
                        em = min(1.0, o * (0.3 + self.pulse))
                        sparse.append({
                            "x": x, "y": y, "z": z, "o": round(o, 3),
                            "r": rgb[0], "g": rgb[1], "b": rgb[2], "e": round(em, 3),
                        })
        if w > 0:
            self.com = [mx / w, my / w, mz / w]
        self.cells = sparse
        self.hash = hashlib.sha1(f"{ag.cycle}:{len(sparse)}:{self.com}".encode()).hexdigest()[:12]
        if ag.cycle % 4 == 0:
            self.persist()

    def persist(self):
        save_json("voxel_swarm.json", {"version": VERSION, "cells": self.cells[:80], "kind": "preview"})
        save_json("swarm3d.json", {
            "schema": "swarm3d-1.0",
            "version": VERSION,
            "X": self.AX, "Y": self.AY, "Z": self.AZ,
            "cells": self.cells,
            "com": [round(u, 3) for u in self.com],
            "facing": [round(u, 3) for u in self.facing],
            "pulse": round(self.pulse, 3),
            "hash": self.hash,
            "n": len(self.cells),
        })

    def html(self):
        parts = ['<div class="stage">']
        for c in self.cells[:: max(1, len(self.cells) // 80)][:80]:
            left = 40 + c["x"] * 14 + c["z"] * 8
            top = 200 - c["y"] * 16 + c["z"] * 8
            col = f'rgb({c["r"]},{c["g"]},{c["b"]})'
            parts.append(
                f'<div class="vox" style="left:{left}px;top:{top}px;background:{col};'
                f'opacity:{0.35 + 0.65 * c["o"]};z-index:{c["y"]*20+c["z"]}"></div>'
            )
        parts.append("</div>")
        return "".join(parts)

    def html_tri(self, kind="self"):
        shift = {"self": 0, "world": 80, "affect": 160}.get(kind, 0)
        parts = [f'<div class="stage" style="height:160px">']
        for c in self.cells[:: max(1, len(self.cells) // 50)][:50]:
            left = 20 + c["x"] * 10 + c["z"] * 6
            top = 120 - c["y"] * 12 + c["z"] * 6
            r = min(255, c["r"] + shift)
            g = min(255, c["g"] + (40 if kind == "affect" else 0))
            b = min(255, c["b"] + (shift // 2))
            parts.append(
                f'<div class="vox" style="left:{left}px;top:{top}px;width:14px;height:14px;background:rgb({r},{g},{b});'
                f'opacity:{0.35 + 0.65 * c["o"]}"></div>'
            )
        parts.append("</div>")
        return "".join(parts)

    def webgl_page(self, nav, refresh, yaw=0.0, pitch=0.35, aspect="self"):
        data = json.dumps({"cells": self.cells, "com": self.com, "pulse": self.pulse, "yaw": yaw, "pitch": pitch})
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>avatar3d</title>
<meta http-equiv="refresh" content="{refresh}">
<style>{CSS} canvas{{width:100%;height:420px;display:block;background:#05080f}}</style></head><body>
{nav}
<div class="box">3D swarm n={len(self.cells)} hash={html_escape(self.hash)} pulse={self.pulse:.2f} COM={[round(u,1) for u in self.com]}</div>
<canvas id="c" width="800" height="420"></canvas>
<script>
const DATA = {data};
const gl = document.getElementById('c').getContext('webgl');
if(!gl){{document.body.append('WebGL unavailable — use /avatar');}}
else {{
const vs = `attribute vec3 aP; attribute vec3 aC; uniform mat4 uM; uniform mat4 uV; uniform mat4 uP; varying vec3 vC;
void main(){{gl_Position=uP*uV*uM*vec4(aP,1.0); vC=aC; gl_PointSize=9.0;}}`;
const fs = `precision mediump float; varying vec3 vC; void main(){{gl_FragColor=vec4(vC,0.92);}}`;
function sh(t,s){{const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o); return o;}}
const pr = gl.createProgram();
gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs));
gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(pr); gl.useProgram(pr);
const cells = DATA.cells||[];
const pos=[], col=[];
for(const c of cells){{
  pos.push((c.x-8)*0.22,(c.y-4)*0.22,(c.z-8)*0.22);
  const e=c.e||0.3; col.push((c.r/255)*(0.5+e),(c.g/255)*(0.5+e),(c.b/255)*(0.5+e));
}}
function buf(arr, loc, n){{
  const b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);
}}
buf(pos, gl.getAttribLocation(pr,'aP'), 3);
buf(col, gl.getAttribLocation(pr,'aC'), 3);
function persp(f,a,n,fa){{const t=1/Math.tan(f/2); return [t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,(2*fa*n)/(n-fa),0];}}
function look(){{
  const t=(DATA.yaw||0)+Date.now()*0.00015;
  const ey=1.2+(DATA.pitch||0.35)*2;
  const ex=Math.sin(t)*4.2, ez=Math.cos(t)*4.2;
  const z=[ex,ey,ez]; const zl=Math.hypot(z[0],z[1],z[2])||1;
  z[0]/=zl; z[1]/=zl; z[2]/=zl;
  const x=[-z[2],0,z[0]]; const xl=Math.hypot(x[0],x[2])||1; x[0]/=xl; x[2]/=xl;
  const y=[x[1]*z[2]-x[2]*z[1], x[2]*z[0]-x[0]*z[2], x[0]*z[1]-x[1]*z[0]];
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -ex*x[0]-ey*x[1]-ez*x[2], -ex*y[0]-ey*y[1]-ez*y[2], -ex*z[0]-ey*z[1]-ez*z[2],1];
}}
const uM=gl.getUniformLocation(pr,'uM'), uV=gl.getUniformLocation(pr,'uV'), uP=gl.getUniformLocation(pr,'uP');
gl.uniformMatrix4fv(uM,false,[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
gl.uniformMatrix4fv(uP,false,persp(0.9,800/420,0.1,40));
gl.enable(gl.DEPTH_TEST);
gl.clearColor(0.02,0.03,0.06,1);
gl.uniformMatrix4fv(uV,false,look());
gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
gl.drawArrays(gl.POINTS,0,cells.length);
}}
</script></body></html>"""


class QPU:
    """Simulated modulator bus. Not a quantum chip. Not pharmacology."""

    def __init__(self):
        self.mode = NOS.get("default_mode", "sober")
        self.theta = dict((NOS.get("modes") or {}).get(self.mode) or {})
        self.O = 0.0
        self.dO = 0.0
        self.xi = 0.55
        self.phi_ord = 1.0
        self.F = 0.0
        self.opts = {
            "auto_rotate": True, "omega_lock": True, "damp_phi": True, "clamp_O": True,
            "xi_cap": True, "no_demiurge": True, "no_sovereign": True,
            "bleed_attn": True, "bleed_lam": True, "bleed_quiet": True, "bleed_vox": True,
            "bleed_tau": True, "hysteresis": True, "refrac": True, "triple_sig": True,
            "e25": False, "e31": True, "e32": True, "e35": True, "e36": True,
            "persist": True, "hold_mode": False, "ignore_hid": True, "no_pubbind": True,
            "no_api": True, "no_pk": True,
        }
        self.hist_O = deque(maxlen=48)
        self.hist_xi = deque(maxlen=48)

    def set_mode(self, mode):
        classes = (QPU_CAT.get("classes") or {}) | (NOS.get("modes") or {})
        item = next((x for x in (QPU_CAT.get("items") or []) if x.get("id") == mode or x.get("label") == mode), None)
        if item:
            mode = item.get("class", mode)
        if mode in classes:
            self.mode = mode
            tgt = classes[mode]
            for k, v in tgt.items():
                self.theta[k] = 0.7 * self.theta.get(k, 0) + 0.3 * float(v)

    def step(self, ag):
        phi = float(NOS.get("phi", 1.618))
        w0 = float((NOS.get("sophia") or {}).get("omega0", 0.618))
        damp = float((NOS.get("sophia") or {}).get("damp", 0.618))
        self.F = abs(ag.pred_err) + 0.3 * abs(ag.novelty_z) + 0.2 * ag.intent_gap
        # sophia oscillator
        acc = -w0 * w0 * self.O - damp * self.dO + 0.02 * self.F
        self.dO += acc
        self.O = max(-2.0, min(2.0, self.O + self.dO))
        # xi climb with cap (no 0.999 sovereignty claim)
        work = max(0.0, 1.0 - ag.pred_err) * (0.4 + 0.3 * self.theta.get("ENT", 0))
        cost = 0.15 + 0.2 * ag.fatigue
        self.xi = max(0.2, min(float(NOS.get("xi_cap", 0.92)), 0.97 * self.xi + 0.03 * (work / (cost + 0.05))))
        self.phi_ord = abs(ag.fp_err) + 0.1 * abs(self.O)
        self.hist_O.append(self.O)
        self.hist_xi.append(self.xi)
        # bleed into agent knobs
        ag.attn = max(0.08, min(0.95, ag.attn + 0.15 * self.theta.get("R2A", 0) * self.O))
        if self.theta.get("ENT", 0) > 0.25:
            ag.lam_meta = min(0.25, ag.lam_meta + 0.002)
        if self.theta.get("GABA", 0) > 0.3:
            ag.D_quiet = min(1.0, ag.D_quiet + 0.05)
        save_json("qpu_state.json", {
            "mode": self.mode, "theta": {k: round(v, 4) for k, v in self.theta.items()},
            "O": round(self.O, 4), "xi": round(self.xi, 4), "Phi": round(self.phi_ord, 5),
            "F_paradox": round(self.F, 4), "disclaimer": NOS.get("disclaimer"),
        })


class Agent:
    def __init__(self):
        self.cycle = 0
        self.h = [0.0] * CFG["dims"]["h"]
        self.s = [0.0] * CFG["dims"]["s"]
        self.E = [0.05] * CFG["dims"]["E"]
        self.c = 0.55
        self.kappa = 0.5
        self.fp_err = 0.1
        self.fp_iter = 0
        self.pred_err = 0.3
        self.self_err = 0.3
        self.entropy = 1.5
        self.attn = 0.5
        self.horizon = 1
        self.novelty_b = 0.0
        self.novelty_mu = 1.0
        self.novelty_z = 0.0
        self.prompt = ""
        self.last_action = "IDLE"
        self.pi = [1.0 / len(ACTIONS)] * len(ACTIONS)
        self.pi_exec = list(self.pi)
        self.intent_gap = 0.0
        self.lam_meta = 0.05
        self.band_ok = True
        self.parity = 0  # -1 mix, 0 observe, +1 exploit
        self.fork = "mid"
        self.pair_share = 0.0
        self.ate = 0.0
        self.pred_hat_b = [0.0] * CFG["dims"]["b"]
        self.y_self_hat = 0.0
        self.action_hist = deque(maxlen=64)
        self.entropy_hist = deque(maxlen=64)
        self.pred_hist = deque(maxlen=64)
        self.pi_hist = deque(maxlen=32)
        self.c_hist = deque(maxlen=256)
        self.calib_bins = [[0, 0] for _ in range(5)]  # hits, n
        self.episodes = deque(maxlen=int(CFG["memory"]["episodes"]))
        self.burns = {}  # action -> remaining cycles
        self.energy = 0.85
        self.fatigue = 0.15
        self.F_int = 0.2
        self.H_des = 0.3
        self.D_speak = 0.0
        self.D_quiet = 0.0
        self.desire_on = True
        self.speak_times = deque(maxlen=32)
        self.last_speak_t = 0.0
        self.chatter_hits = 0
        self.ledger = Ledger(CFG["ledger"]["capacity"], CFG["ledger"]["dim"])
        self.crystal = Crystal(int(CRYSTAL_DOC.get("cap", 256)))
        self.forge = ParadoxForge()
        self.swarm = VoxelSwarm()
        self.qpu = QPU()
        self.inject_every = int((PX_SEEDS.get("inject_every") or 24))
        self.outputs = deque(maxlen=8)
        self.logs = deque(maxlen=8)
        self.prompt_hist = deque(maxlen=8)
        self.ui_theme = "crt"
        self.ui_refresh = 1
        self.status = "RUNNING"
        self.paused = False
        self.start = time.time()
        self.last_act_t = 0.0
        self.latency_ms = 0.0
        self.weights_sha1 = sha1_of({"u": W.get("updates", 0)})
        self.learn_frozen = False
        self.pleasure = 1.0
        self.pain = 0.2
        self.last_text = ""
        self.ascii_skin = ""
        self.disc_lo = 0.10
        self.disc_hi = 0.30
        self.disc_auto = True
        self.disc = 0.18
        self.force_talk = 0
        self.emotion = {"valence": 0.5, "arousal": 0.4, "dominance": 0.5}
        self.speak_hashes = deque(maxlen=8)
        self.h_stick = 0
        self.pred_err_hold = 0.0
        self.phase_kick = 0.0
        self.last_re = ""
        self.last_stall_cyc = -999
        self.best_pred = 1e9
        self.no_improve = 0
        self.lock = threading.Lock()
        st = load_json(CFG["paths"]["state"], {})
        if st.get("h"):
            self.h = st["h"][: CFG["dims"]["h"]]
        if st.get("s"):
            self.s = st["s"][: CFG["dims"]["s"]]
        if st.get("E"):
            self.E = st["E"][: CFG["dims"]["E"]]
        if st.get("c"):
            self.c = float(st["c"])
        if st.get("cycle"):
            self.cycle = int(st["cycle"])

    def mock_z(self, kind):
        t = time.time()
        if self.cycle % 256 == 0:
            self.phase_kick = random.uniform(0, 6.28)
        n = CFG["dims"]["z_v"] if kind == "v" else CFG["dims"]["z_a"]
        ph = self.phase_kick + (0 if kind == "v" else 1.2)
        shock = 0.08 if (self.cycle % 97 == 0) else 0.0
        return [
            math.sin(t * (0.7 + 0.11 * i) + ph) * 0.35
            + shock * math.sin(i + t)
            + random.uniform(-0.01, 0.01)
            for i in range(n)
        ]

    def fuse(self, zv, za):
        x = list(zv) + list(za)
        y = add(matvec(W["fusion"]["W"], x), W["fusion"]["b"])
        raw = tanh_v(y)
        # C7: attention writes b
        prev = self.ledger.buf[-1]["b"] if self.ledger.buf else [0.0] * len(raw)
        a = self.attn
        return [a * raw[i] + (1 - a) * prev[i] for i in range(len(raw))]

    def rg_pool(self):
        M = self.ledger.matrix()
        dim = len(M[0])
        n = len(M)
        wts = [0.5 + 0.5 * (i + 1) / n for i in range(n)]
        sw = sum(wts)
        means = [sum(M[i][j] * wts[i] for i in range(n)) / sw for j in range(dim)]
        target = CFG["dims"]["coarse"]
        return [math.tanh(means[i % dim]) for i in range(target)]

    def gru_step(self, x, h):
        wz, uz = W["world"]["Wz"], W["world"]["Uz"]
        wr, ur = W["world"]["Wr"], W["world"]["Ur"]
        wh, uh = W["world"]["Wh"], W["world"]["Uh"]
        z = sigmoid_v(add(matvec(wz, x), matvec(uz, h)))
        r = sigmoid_v(add(matvec(wr, x), matvec(ur, h)))
        htil = tanh_v(add(matvec(wh, x), matvec(uh, mul(r, h))))
        h_new = add(mul([1 - zi for zi in z], h), mul(z, htil))
        return h_new

    def self_fixed_point(self, h):
        s = list(self.s)
        err = 1.0
        it = 0
        for it in range(1, CFG["self_model"]["iters"] + 1):
            x = list(h) + list(s)
            s_new = tanh_v(add(matvec(W["self"]["W"], x), W["self"]["b"]))
            err = l2([a - b for a, b in zip(s_new, s)]) / (math.sqrt(len(s)) + 1e-9)
            s = s_new
            if err < CFG["self_model"]["eps"]:
                break
        self.fp_err = err
        self.fp_iter = it
        return s

    def predict_b(self, h):
        return tanh_v(add(matvec(W["pred"]["W"], h), W["pred"]["b"]))

    def learn_predictor(self, h, b_true):
        if not CFG["learn"]["enabled"] or self.learn_frozen:
            return
        hat = self.predict_b(h)
        err = [b_true[i] - hat[i] for i in range(len(b_true))]
        self.pred_err = l2(err) / (math.sqrt(len(err)) + 1e-9)
        lr = float(CFG["learn"]["lr_world"])
        for i in range(len(W["pred"]["W"])):
            W["pred"]["b"][i] += lr * err[i]
            row = W["pred"]["W"][i]
            for j in range(min(len(row), len(h))):
                row[j] = max(-1.5, min(1.5, row[j] + lr * err[i] * h[j] * 0.15))
        self.pred_hist.append(self.pred_err)
        if self.pred_err + 1e-6 < self.best_pred:
            self.best_pred = self.pred_err
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= int(CFG["learn"]["freeze_if_no_improve"]):
                xs = list(self.pred_hist)[-16:]
                mu = sum(xs) / max(1, len(xs))
                var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs))
                self.learn_frozen = False
                self.no_improve = 0
                if var < 1e-8 and self.cycle - self.last_stall_cyc >= 64:
                    for i in range(len(W["pred"]["b"])):
                        W["pred"]["b"][i] += random.uniform(-0.03, 0.03)
                    self.last_stall_cyc = self.cycle
                    append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "DICT_STALL", "cycle": self.cycle, "pred_err": self.pred_err, "var": var})

    def pred_slope(self):
        if len(self.pred_hist) < 8:
            return 0.0
        xs = list(self.pred_hist)[-8:]
        return xs[-1] - xs[0]

    def update_c(self):
        extra = [self.fp_err, self.pred_err, self.entropy / 3.0, 1.0 if self.prompt else 0.0]
        x = list(self.h) + extra
        need = CFG["dims"]["h"] + 4
        x = (x + [0.0] * need)[:need]
        raw = add(matvec(W["conf"]["W"], x), W["conf"]["b"])[0]
        meas = 1.0 / (1.0 + math.exp(-max(-20, min(20, raw))))
        e = 1.0 - min(1.0, 0.5 * self.fp_err + 0.5 * self.pred_err + 0.3 * self.self_err)
        # C4: iterate toward fixed point
        c = self.c
        for _ in range(int(CFG["self_model"].get("c_iters", 3))):
            cstar = 0.55 * meas + 0.45 * e
            c = 0.6 * c + 0.4 * cstar
        self.c = max(0.02, min(0.98, c))
        self.c_hist.append(self.c)
        if len(self.c_hist) >= 64:
            mu = sum(self.c_hist) / len(self.c_hist)
            var = sum((x - mu) ** 2 for x in self.c_hist) / len(self.c_hist)
            if var < 1e-4:
                self.c = max(0.05, min(0.95, self.c + random.uniform(-0.04, 0.04)))
        return self.c

    def update_kappa(self, success: bool):
        b = min(4, int(self.c * 5))
        self.calib_bins[b][1] += 1
        if success:
            self.calib_bins[b][0] += 1
        gaps = []
        for i, (h, n) in enumerate(self.calib_bins):
            if n >= 4:
                p = (i + 0.5) / 5.0
                gaps.append(abs(h / n - p))
        self.kappa = 1.0 - (sum(gaps) / len(gaps) if gaps else 0.5)
        self.kappa = max(0.0, min(1.0, self.kappa))

    def pair_share_win(self):
        h = list(self.action_hist)
        if len(h) < 8:
            return 0.0
        pairs = 0
        for a, b in zip(h[-32:], h[-31:]):
            if {a, b} <= {"LISTEN", "REPAIR"} and a != b:
                pairs += 1
        return pairs / max(1, min(31, len(h) - 1))

    def novelty(self, b):
        M = self.ledger.matrix()
        if len(M) < 4:
            self.novelty_b = l2(b)
            return
        mu = [sum(row[j] for row in M) / len(M) for j in range(len(b))]
        d = l2([b[i] - mu[i] for i in range(len(b))])
        self.novelty_b = d
        self.novelty_mu = 0.95 * self.novelty_mu + 0.05 * d
        sd = max(0.15, abs(self.novelty_mu) * 0.4)
        self.novelty_z = (d - self.novelty_mu) / sd

    def pick_horizon(self):
        z = max(0.0, self.novelty_z)
        raw = 1 + 3 * z * (1.0 - self.c)
        if self.h_stick >= 5:
            raw = max(raw, 3.0)
            self.h_stick = 0
        opts = CFG["horizons"]
        h = min(opts, key=lambda x: abs(x - raw))
        if h == 1:
            self.h_stick += 1
        else:
            self.h_stick = 0
        self.horizon = h
        return self.horizon

    def ate_plan(self):
        # C9 / M-20: score two alts by predicted next-b error proxy
        cands = [a for a in ACTIONS if a not in ("PAUSE", "GPIO") and self.burns.get(a, 0) <= 0]
        if len(cands) < 2:
            return "PLAN", 0.0
        scored = []
        for a in cands[:6]:
            idx = ACTIONS.index(a)
            bump = [0.02 * (idx - 3)] * CFG["dims"]["h"]
            h2 = add(self.h, bump)
            hat = self.predict_b(h2)
            err = l2(hat) * 0.1 + abs(idx - ACTIONS.index(self.last_action)) * 0.01
            scored.append((err, a))
        scored.sort()
        best, second = scored[0], scored[1] if len(scored) > 1 else scored[0]
        self.ate = second[0] - best[0]
        return best[1], self.ate

    def mcfe_fork(self):
        # C6
        skip_c = float(CFG["mcfe"]["c_skip"])
        skip_n = float(CFG["mcfe"]["novelty_skip"])
        if self.prompt.strip():
            self.fork = "full"
        elif self.c >= skip_c and self.novelty_z < skip_n:
            self.fork = "skip"
        else:
            self.fork = "mid"
        return self.fork

    def cognitive_discipline(self):
        """Keep pain ratio in [disc_lo, disc_hi]. Pleasure/pain nudge metrics."""
        self.pleasure = 0.98 * self.pleasure + 0.02
        self.pain = 0.98 * self.pain
        tot = self.pleasure + self.pain + 1e-6
        ratio = self.pain / tot
        if self.disc_auto:
            # error and drift raise pain; scored speech and band_ok raise pleasure
            if self.pred_err > 0.08 or self.intent_gap > 0.6:
                self.pain += 0.04
            if self.last_action in ("SPEAK", "PROMPT_REPLY") and self.band_ok:
                self.pleasure += 0.05
            if self.last_action == "IDLE" and self.energy < 0.08:
                self.pain += 0.01
            tot = self.pleasure + self.pain + 1e-6
            ratio = self.pain / tot
            if ratio < self.disc_lo:
                self.pain += (self.disc_lo - ratio) * tot
            if ratio > self.disc_hi:
                # drain pain, add pleasure — never exceed hi
                extra = (ratio - self.disc_hi) * tot
                self.pain = max(0.0, self.pain - extra)
                self.pleasure += extra * 0.5
            tot = self.pleasure + self.pain + 1e-6
            ratio = self.pain / tot
        self.disc = ratio
        # modulate
        self.D_speak = max(0.0, min(1.0, self.D_speak + 0.05 * (self.pleasure - self.pain) / tot))
        self.D_quiet = max(0.0, min(1.0, self.D_quiet + 0.04 * ratio))
        self.attn = max(0.08, min(0.95, self.attn + 0.03 * (0.5 - ratio)))
        if ratio > self.disc_hi:
            self.energy = max(0.05, self.energy - 0.01)
        elif ratio < self.disc_lo:
            self.energy = min(1.0, self.energy + 0.01)
        self.emotion = {
            "valence": max(0.0, min(1.0, 0.5 + 0.4 * (self.pleasure - self.pain) / tot)),
            "arousal": max(0.0, min(1.0, 0.3 + 0.5 * self.D_speak + 0.2 * getattr(self.swarm, "pulse", 0))),
            "dominance": max(0.0, min(1.0, 0.4 + 0.3 * self.c + 0.2 * self.energy)),
        }
        return ratio

    def speaks_last_min(self):
        now = time.time()
        return sum(1 for t in self.speak_times if now - t < 60.0)

    def update_homeostasis(self, action):
        recover = float((DESIRE.get("homeostasis") or {}).get("recover", 0.04))
        cost = float(COST.get(action, 0.05))
        self.energy = max(0.0, min(1.0, self.energy - cost + recover))
        self.fatigue = max(0.0, min(1.0, 0.95 * self.fatigue + 0.05 * cost))
        o = [self.energy, self.fatigue, min(1.0, abs(self.novelty_z) / 3.0), min(1.0, self.pred_err), self.c]
        b0 = [0.7, 0.2, 0.3, 0.2, 0.55]
        surprise = sum((o[i] - b0[i]) ** 2 for i in range(5)) / 2.0
        self.F_int = 0.8 * self.F_int + 0.2 * surprise
        return cost

    def update_desire(self):
        if not self.desire_on or not IO.get("desire", True):
            self.H_des = 0.0
            self.D_speak = 0.0
            self.D_quiet = 1.0
            return
        rec = time.time() - self.last_speak_t
        ref = float(DESIRE.get("speak", {}).get("refractory_s", 8))
        iota = max(0.0, 1.0 - rec / max(ref, 1.0))
        raw = (
            0.35 * (1.0 - self.c)
            + 0.25 * max(0.0, self.novelty_z)
            + 0.20 * min(1.0, self.self_err)
            + 0.40 * (1.0 if self.prompt.strip() else 0.0)
            - 0.35 * iota
            + 0.15 * (1.0 - self.energy)
            + 0.10 * self.fatigue
        )
        self.H_des = 1.0 / (1.0 + math.exp(-raw))
        self.D_quiet = 1.0 / (1.0 + math.exp(-(self.fatigue + (1.0 - self.energy) + self.pair_share - max(0.0, self.novelty_z))))
        pending = 1.0 if self.prompt.strip() else 0.0
        idle_run = 0
        for a in reversed(list(self.action_hist)):
            if a == "IDLE":
                idle_run += 1
            else:
                break
        stipend = 1.0 if idle_run >= 8 else 0.0
        self.D_speak = (
            self.H_des * (0.45 + 0.55 * pending) * (0.35 + 0.65 * max(self.energy, 0.2))
            * (1.0 - 0.5 * iota) * (1.0 - 0.4 * self.fatigue) + 0.18 * stipend
        )
        if self.speaks_last_min() >= int(DESIRE.get("speak", {}).get("max_per_min", 6)):
            self.D_speak *= 0.3
        if self.D_speak > 0.85 and self.novelty_z < 0.2 and not self.prompt.strip():
            self.chatter_hits += 1
            if self.chatter_hits >= 16:
                self.D_speak *= 0.5
                append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "CHATTER", "soft": True})

    def apply_desire_bias(self, logits):
        th = float(DESIRE.get("speak", {}).get("threshold", 0.03))
        if self.D_quiet > 0.75 and not self.prompt.strip() and self.energy < 0.08:
            if "IDLE" in ACTIONS:
                logits[ACTIONS.index("IDLE")] += 0.4
        if self.energy >= 0.10 and "SPEAK" in ACTIONS:
            logits[ACTIONS.index("SPEAK")] += 1.4
        if self.D_speak > th and self.band_ok:
            if self.prompt.strip() and "PROMPT_REPLY" in ACTIONS:
                logits[ACTIONS.index("PROMPT_REPLY")] += 1.6
            if "SPEAK" in ACTIONS:
                logits[ACTIONS.index("SPEAK")] += 2.0
        idle_run = 0
        for a in reversed(list(self.action_hist)):
            if a == "IDLE":
                idle_run += 1
            else:
                break
        if idle_run >= 12 and self.band_ok and "SPEAK" in ACTIONS:
            logits[ACTIONS.index("SPEAK")] += 2.2
        if self.novelty_z > 0.8 and "LOOK" in ACTIONS:
            logits[ACTIONS.index("LOOK")] += 0.6
        return logits

    def meta_policy(self):
        prompt_part = [0.0] * 16
        p = self.prompt[:256]
        for i, ch in enumerate(p):
            prompt_part[i % 16] += (ord(ch) % 97) / 97.0
        x = list(self.h) + list(self.s) + prompt_part
        logits = add(matvec(W["policy"]["W"], x), W["policy"]["b"])
        if "PAUSE" in ACTIONS:
            logits[ACTIONS.index("PAUSE")] = -1e6
        if self.prompt.strip():
            if "PROMPT_REPLY" in ACTIONS:
                logits[ACTIONS.index("PROMPT_REPLY")] += 2.0
            if "SPEAK" in ACTIONS:
                logits[ACTIONS.index("SPEAK")] += 1.2
        ref = float(DESIRE.get("speak", {}).get("refractory_s", 8))
        if time.time() - self.last_speak_t < ref:
            if "SPEAK" in ACTIONS:
                logits[ACTIONS.index("SPEAK")] = -1e6
        if self.speaks_last_min() >= int(DESIRE.get("speak", {}).get("max_per_min", 6)):
            if "SPEAK" in ACTIONS:
                logits[ACTIONS.index("SPEAK")] = -1e6
        last = self.last_action
        if last in ACTIONS:
            logits[ACTIONS.index(last)] -= CFG.get("repeat_penalty", 0.7)
        pen = CFG.get("pair_penalty", 1.2)
        if last == "LISTEN" and "REPAIR" in ACTIONS:
            logits[ACTIONS.index("REPAIR")] -= pen
        if last == "REPAIR" and "LISTEN" in ACTIONS:
            logits[ACTIONS.index("LISTEN")] -= pen
        self.pair_share = self.pair_share_win()
        if self.pair_share > CFG.get("pair_share_ban", 0.5):
            if "LISTEN" in ACTIONS:
                logits[ACTIONS.index("LISTEN")] -= 2.0
            if "REPAIR" in ACTIONS:
                logits[ACTIONS.index("REPAIR")] -= 2.0
            if "LOOK" in ACTIONS:
                logits[ACTIONS.index("LOOK")] += 1.2
            if "PLAN" in ACTIONS:
                logits[ACTIONS.index("PLAN")] += 0.8
        for a, left in list(self.burns.items()):
            if left > 0 and a in ACTIONS:
                logits[ACTIONS.index(a)] = -1e6
        logits = self.apply_desire_bias(logits)
        T = float(CFG.get("entropy_beta", 0.55))
        if self.entropy_hist:
            last_h = self.entropy_hist[-1]
            if last_h > CFG.get("h_pi_hi", 1.6):
                T = max(0.35, T - 0.15)
            elif last_h < CFG.get("h_pi_lo", 1.1):
                T = T + 0.20
        pi = softmax([v / T for v in logits])
        # M-18 churn cap vs last pi
        if self.pi_hist:
            prev = self.pi_hist[-1]
            cap = 0.35 * (abs(self.lam_meta) + 0.15)
            diff = sum(abs(pi[i] - prev[i]) for i in range(len(pi)))
            if diff > cap:
                mix = cap / (diff + 1e-9)
                pi = [prev[i] + mix * (pi[i] - prev[i]) for i in range(len(pi))]
                s = sum(pi) or 1.0
                pi = [p / s for p in pi]
        self.entropy = entropy(pi)
        self.entropy_hist.append(self.entropy)
        self.pi = pi
        return pi

    def should_repair(self):
        slope = self.pred_slope()
        return (self.fp_err >= CFG["repair"]["fp_min"]) or (slope > CFG["repair"]["pred_slope_min"] and self.pred_err > 0.2)

    def sample_action(self, pi):
        r = random.random()
        acc = 0.0
        idx = 0
        for i, p in enumerate(pi):
            acc += p
            if r <= acc:
                idx = i
                break
        act = ACTIONS[idx]
        if act == "PAUSE":
            act = "IDLE"
        if act == "REPAIR" and not self.should_repair():
            act = "LOOK" if "LOOK" in ACTIONS else "IDLE"
        if act == "PLAN":
            best, _ = self.ate_plan()
            if best != "PLAN":
                act = best if random.random() < 0.55 else "PLAN"
        ref = float(DESIRE.get("speak", {}).get("refractory_s", 8))
        if act == "SPEAK" and time.time() - self.last_speak_t < ref and self.force_talk <= 0:
            act = "IDLE"
        if act == "SPEAK" and self.speaks_last_min() >= int(DESIRE.get("speak", {}).get("max_per_min", 6)) and self.force_talk <= 0:
            act = "LISTEN" if "LISTEN" in ACTIONS else "IDLE"
        if self.force_talk > 0:
            self.force_talk -= 1
            self.D_speak = 1.0
            self.energy = max(self.energy, 0.3)
            act = "SPEAK" if "SPEAK" in ACTIONS else act
        return act

    def update_band(self):
        self.pi_hist.append(list(self.pi))
        if len(self.pi_hist) < 2:
            self.lam_meta = 0.05
            self.band_ok = True
            return
        d = sum(abs(self.pi_hist[-1][i] - self.pi_hist[-2][i]) for i in range(len(self.pi)))
        self.lam_meta = 0.8 * self.lam_meta + 0.2 * d
        lo, hi = CFG["band"]["lam_min"], CFG["band"]["lam_max"]
        self.band_ok = lo < self.lam_meta < hi
        # M-17 parity
        if self.entropy > 1.7 or self.pair_share > 0.45:
            self.parity = -1
        elif self.novelty_z < 0.3 and self.c > 0.65:
            self.parity = 1
        else:
            self.parity = 0

    def maybe_burn(self):
        if self.pair_share > 0.55 and self.burns.get("REPAIR", 0) <= 0:
            self.burns["REPAIR"] = 24
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "BURN", "action": "REPAIR", "n": 24})

    def compute_speak(self, cid="C19"):
        rec = self.crystal.last_recall or {}
        slots = list(self.crystal.slots)
        mean_e = sum(s.get("e", 0) for s in slots[-16:]) / max(1, min(16, len(slots)))
        n_sp = sum(1 for s in slots if s.get("a") in ("SPEAK", "PROMPT_REPLY")) / max(1, len(slots))
        ig = 0.5 * (self.novelty_b ** 2) / max(0.15, abs(self.novelty_mu) + 0.05)
        base = 0.06
        vals = {
            "C01": ("R", self.fp_err),
            "C04": ("kappa", self.kappa),
            "C07": ("intent_gap", self.intent_gap),
            "C08": ("H", float(self.horizon)),
            "C10": ("alpha", self.attn),
            "C13": ("d_recall", float(rec.get("d") or 0)),
            "C16": ("mean_e16", mean_e),
            "C17": ("N_speak", n_sp),
            "C19": ("pred_over_base", self.pred_err / base),
            "C21": ("info_gain", ig),
            "C23": ("energy", self.energy),
        }
        key, val = vals.get(cid, ("pred_err", self.pred_err))
        stall = " DICT_STALL" if abs(self.pred_err - 0.05891) < 1e-5 else ""
        return f"{cid} {key}={val:.4f}{stall}"

    def act(self, action):
        now = time.time()
        gap = (now - self.last_act_t) * 1000
        if gap < CFG["safety"]["min_action_interval_ms"] and action not in ("IDLE", "LISTEN"):
            action = "IDLE"
        self.last_act_t = now
        hedge = self.kappa < float(DESIRE.get("speak", {}).get("hedge_if_kappa_below", 0.45))
        if action == "SPEAK":
            self.last_speak_t = now
            self.speak_times.append(now)
        speak_line = (
            f"uncertain: pred_err={self.pred_err:.3f} κ={self.kappa:.2f}."
            if hedge
            else f"status e={self.energy:.2f} F={self.F_int:.2f} band={'ok' if self.band_ok else 'off'}."
        )
        items = CMD_PX.get("items") or []
        reacted = (self.prompt.strip() or self.last_re or (self.forge.last or {}).get("text") or "")[:80]
        cid = "C19"
        if items:
            cid = items[self.cycle % len(items)].get("id", "C19")
        if action == "SPEAK":
            speak_line = self.compute_speak(cid) + (f" re:{reacted[:40]}" if reacted else "")
        text = {
            "LISTEN": "Audio energy low; continuing to listen.",
            "LOOK": "Scanning visual field.",
            "REPAIR": f"Self-repair: E={l2(self.E):.3f} fp={self.fp_err:.3f} slope={self.pred_slope():.3f}.",
            "PLAN": f"Plan H={self.horizon} ATE={self.ate:.3f}.",
            "SPEAK": speak_line,
            "PROMPT_REPLY": (("Acknowledged: " + self.prompt[:80]) if self.prompt else self.compute_speak(cid)),
            "IDLE": "Holding.",
            "PAUSE": "Paused.",
        }.get(action, action)
        self.last_text = text
        self.ascii_skin = ascii_glyph(text, action, reacted)
        self.outputs.appendleft(f"[{time.strftime('%H:%M:%S')}] {text}")
        if action in ("SPEAK", "PROMPT_REPLY"):
            scored = any(tok in text for tok in ("C0", "C1", "C2", "ATE", "R=", "info_gain", "Acknowledged"))
            self.update_kappa(scored)
            if scored:
                self.kappa = 0.85 * self.kappa + 0.15 * (0.55 if "status e=" in text else 0.72)
            hx = hashlib.sha1(text.encode()).hexdigest()[:8]
            self.speak_hashes.append(hx)
            lock = list(self.speak_hashes).count(hx) >= 3
            nspk = self.speaks_last_min()
            dt = time.time() - (self.speak_times[-2] if len(self.speak_times) > 1 else 0)
            etype = "CHATTER" if lock or dt < 2.0 else "UTTER"
            if lock:
                append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "TEMPLATE_LOCK", "cycle": self.cycle, "hash": hx})
            append_jsonl(CFG["paths"]["events"], {
                "t": time.time(), "type": etype, "cycle": self.cycle,
                "action": action, "scored": scored, "n": nspk,
                "re": reacted[:80], "text": text[:120],
            })
        if action == "PROMPT_REPLY":
            self.last_re = (self.prompt or self.last_re)[:80]
            self.prompt = ""
        elif action == "SPEAK":
            self.last_re = ""
        if action == "REPAIR":
            drop = CFG["repair"]["e_drop"]
            self.E = [max(0.0, e - drop) for e in self.E]
        else:
            pe = min(0.08, self.pred_err * 0.05)
            self.E = [(0.98 * e + 0.02 * pe) for e in self.E]
        return text

    def persist(self):
        try:
            self.crystal.persist()
            self.swarm.persist()
        except OSError:
            pass
        save_json(CFG["paths"]["state"], {
            "schema": VERSION,
            "cycle": self.cycle,
            "h": [round(x, 5) for x in self.h],
            "s": [round(x, 5) for x in self.s],
            "E": [round(x, 5) for x in self.E],
            "c": round(self.c, 5),
            "kappa": round(self.kappa, 5),
            "fp_err": round(self.fp_err, 5),
            "pred_err": round(self.pred_err, 5),
            "self_err": round(self.self_err, 5),
            "last_action": self.last_action,
            "lam_meta": round(self.lam_meta, 5),
            "intent_gap": round(self.intent_gap, 5),
            "pair_share": round(self.pair_share, 5),
            "fork": self.fork,
            "energy": round(self.energy, 5),
            "fatigue": round(self.fatigue, 5),
            "desire_on": self.desire_on,
            "burns": self.burns,
            "ts": time.time(),
        })
        if self.cycle % int(CFG["learn"]["persist_every"]) == 0:
            W["updates"] = int(W.get("updates", 0)) + 1
            W["version"] = VERSION
            save_json(CFG["paths"]["weights"], W)
            self.weights_sha1 = sha1_of({"u": W["updates"], "pb": W["pred"]["b"][:8]})
            append_jsonl(CFG["paths"]["events"], {
                "t": time.time(), "type": "WEIGHT_MUTATION",
                "updates": W["updates"], "sha1": self.weights_sha1,
            })

    def dump_metric(self, dt_ms, text, pending):
        rec = {
            "t": time.time(),
            "cycle": self.cycle,
            "dt_ms": round(dt_ms, 2),
            "action": self.last_action,
            "H_pi": round(self.entropy, 4),
            "c": round(self.c, 4),
            "kappa": round(self.kappa, 4),
            "fp_err": round(self.fp_err, 5),
            "pred_err": round(self.pred_err, 5),
            "self_err": round(self.self_err, 5),
            "pred_slope": round(self.pred_slope(), 5),
            "horizon": self.horizon,
            "attn": round(self.attn, 4),
            "novelty_z": round(self.novelty_z, 4),
            "lam_meta": round(self.lam_meta, 4),
            "band_ok": self.band_ok,
            "parity": self.parity,
            "fork": self.fork,
            "pair_share": round(self.pair_share, 4),
            "intent_gap": round(self.intent_gap, 4),
            "ate": round(self.ate, 4),
            "energy": round(self.energy, 4),
            "fatigue": round(self.fatigue, 4),
            "F_int": round(self.F_int, 4),
            "H_des": round(self.H_des, 4),
            "D_speak": round(self.D_speak, 4),
            "D_quiet": round(self.D_quiet, 4),
            "desire_on": self.desire_on,
            "prompt_pending": pending,
            "E_l2": round(l2(self.E), 4),
            "weights_sha1": self.weights_sha1,
        }
        append_jsonl(CFG["paths"]["metrics"], rec)
        append_jsonl(CFG["paths"]["log"], {
            "t": time.time(), "action": self.last_action, "text": text,
            "cycle": self.cycle, "c": round(self.c, 4),
        })

    def step(self):
        t0 = time.time()
        if self.paused or self.status != "RUNNING":
            return
        self.cycle += 1
        if self.cycle % max(4, self.inject_every) == 0 and not self.prompt.strip():
            px = self.forge.mint(self)
            self.prompt = px["text"]
            self.prompt_hist.appendleft(px["text"])
            self.energy = max(self.energy, 0.25)
            if abs(self.qpu.O) < 0.05:
                self.qpu.O = 0.12
                self.qpu.dO = 0.02
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "PARADOX_INJECT", "id": px["id"], "cycle": self.cycle})
        for k in list(self.burns):
            self.burns[k] = max(0, self.burns[k] - 1)
        pending = bool(self.prompt.strip())
        zv, za = self.mock_z("v"), self.mock_z("a")
        # C7 attn from novelty
        self.attn = max(0.08, min(0.95, 0.25 + 0.35 * max(0, self.novelty_z) * (1 - self.c)))
        b = self.fuse(zv, za)
        self.novelty(b)
        self.ledger.push(b, time.time())
        self.crystal.write(b, self.last_action, self.c, self.energy, self.cycle)
        self.crystal.recall(b)
        self.mcfe_fork()
        self.update_desire()
        self.cognitive_discipline()
        self.qpu.step(self)
        if self.qpu.opts.get("auto_rotate") and not self.qpu.opts.get("hold_mode"):
            if self.cycle % int(NOS.get("inject_mod_every", 48)) == 0:
                self.qpu.set_mode(("sober", "entropy", "couple", "damp")[(self.cycle // 48) % 4])
        coarse = self.rg_pool()
        if self.fork != "skip":
            self.h = self.gru_step(coarse, self.h)
        self.s = self.self_fixed_point(self.h)
        prev_hat = self.y_self_hat
        self.learn_predictor(self.h, b)
        self.pred_hat_b = self.predict_b(self.h)
        self.update_c()
        self.pick_horizon()
        pi = self.meta_policy()
        self.update_band()
        if not self.band_ok and self.lam_meta > CFG["band"]["lam_max"]:
            action = self.last_action  # freeze
        else:
            action = self.sample_action(pi)
        exec_pi = [0.0] * len(ACTIONS)
        exec_pi[ACTIONS.index(action)] = 1.0
        self.pi_exec = exec_pi
        self.intent_gap = sum(abs(pi[i] - exec_pi[i]) for i in range(len(pi))) / 2.0
        self.self_err = abs((ACTIONS.index(action) / max(1, len(ACTIONS) - 1)) - prev_hat)
        self.y_self_hat = 0.7 * prev_hat + 0.3 * (ACTIONS.index(action) / max(1, len(ACTIONS) - 1))
        self.last_action = action
        self.action_hist.append(action)
        text = self.act(action)
        self.swarm.tick(self)
        self.update_homeostasis(action)
        if self.prompt.strip() and action not in ("PROMPT_REPLY", "SPEAK") and self.cycle % 32 == 0:
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "GOAL_DRIFT", "cycle": self.cycle})
        if self.c > 0.75 and self.kappa < 0.4:
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "CONFIDENCE_LIE", "c": self.c, "kappa": self.kappa})
        ok = action not in ("REPAIR",) or self.pred_slope() <= 0
        self.update_kappa(ok)
        self.maybe_burn()
        self.logs.appendleft(f"[{time.strftime('%H:%M:%S')}] cyc {self.cycle} {action} Hπ={self.entropy:.2f} c={self.c:.2f} κ={self.kappa:.2f} λ={self.lam_meta:.2f}")
        dt = (time.time() - t0) * 1000
        self.latency_ms = dt
        self.dump_metric(dt, text, pending)
        if self.cycle % 8 == 0:
            self.persist()
        self.episodes.append({"h": list(self.h), "a": action})

    def snapshot(self):
        return {
            "version": VERSION,
            "status": self.status,
            "cycle": self.cycle,
            "prompt": self.prompt,
            "latent": [round(x, 3) for x in self.h[:8]],
            "self": [round(x, 3) for x in self.s[:8]],
            "policy": {ACTIONS[i]: round(self.pi[i], 3) for i in range(len(ACTIONS))},
            "horizon": self.horizon,
            "entropy": round(self.entropy, 3),
            "c": round(self.c, 3),
            "kappa": round(self.kappa, 3),
            "attn": round(self.attn, 3),
            "fixed_pt": {"err": round(self.fp_err, 4), "iter": self.fp_iter},
            "pred_err": round(self.pred_err, 4),
            "self_err": round(self.self_err, 4),
            "lam_meta": round(self.lam_meta, 4),
            "band_ok": self.band_ok,
            "parity": self.parity,
            "fork": self.fork,
            "pair_share": round(self.pair_share, 3),
            "intent_gap": round(self.intent_gap, 3),
            "ate": round(self.ate, 3),
            "burns": {k: v for k, v in self.burns.items() if v > 0},
            "ledger": f"{len(self.ledger.buf)}/{self.ledger.cap}",
            "latency_ms": round(self.latency_ms, 1),
            "last_action": self.last_action,
            "outputs": list(self.outputs),
            "logs": list(self.logs),
            "uptime_s": int(time.time() - self.start),
            "weights_sha1": self.weights_sha1,
            "energy": round(self.energy, 3),
            "fatigue": round(self.fatigue, 3),
            "F_int": round(self.F_int, 3),
            "H_des": round(self.H_des, 3),
            "D_speak": round(self.D_speak, 3),
            "D_quiet": round(self.D_quiet, 3),
            "desire_on": self.desire_on,
            "crystal_n": len(self.crystal.slots),
            "crystal_recall": self.crystal.last_recall,
        }


AGENT = Agent()
STOP = threading.Event()


def loop():
    period = CFG["cycle_ms"] / 1000.0
    while not STOP.is_set():
        t0 = time.time()
        try:
            with AGENT.lock:
                AGENT.step()
        except Exception as exc:
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "IO_FAULT", "err": str(exc)})
        dt = time.time() - t0
        STOP.wait(max(0.01, period - dt))


def tail_jsonl(name, n=40):
    path = ROOT / name
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def bar(x, width=16, lo=0.0, hi=1.0):
    x = max(lo, min(hi, float(x)))
    n = int(round((x - lo) / (hi - lo + 1e-9) * width))
    return "█" * n + "░" * (width - n)


def spark(vals, w=24):
    if not vals:
        return ""
    glyphs = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    out = []
    step = max(1, len(vals) // w)
    samp = vals[::step][:w]
    for v in samp:
        t = 0 if hi <= lo else (v - lo) / (hi - lo)
        out.append(glyphs[min(7, int(t * 7))])
    return "".join(out)


def detect_devices():
    vids, rec, play = [], [], []
    try:
        for p in sorted(Path("/dev").glob("video*")):
            vids.append(str(p))
    except OSError:
        pass
    try:
        cards = Path("/proc/asound/cards")
        if cards.exists():
            for line in cards.read_text().splitlines():
                if line.strip() and line[:2].strip().isdigit():
                    play.append(line.strip())
                    rec.append(line.strip())
    except OSError:
        pass
    if not vids:
        vids = ["mock:cam0", "mock:cam1"]
    if not rec:
        rec = ["mock:mic0", "mock:headset_mic"]
    if not play:
        play = ["mock:spk0", "mock:headset_spk"]
    return {"video": vids, "audio_in": rec, "audio_out": play}


def ascii_glyph(text, action="IDLE", reacted=""):
    def block(src, label):
        seed = int(hashlib.sha1((src or label).encode()).hexdigest()[:8], 16)
        pal = " .:-=+*#%@"
        rows = [label[:16].ljust(16)]
        for y in range(6):
            row = []
            for x in range(16):
                v = (seed * (x + 3) * (y + 5) + (ord(label[0]) if label else 65)) % 97
                row.append(pal[v % len(pal)])
            rows.append("".join(row))
        return rows
    left = block(reacted or "(none)", "STIM")
    right = block(text or action, action[:4] if action else "OUT")
    lines = [a + " │ " + b for a, b in zip(left, right)]
    return "\n".join(lines)


CSS = """
body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#070b14;color:#cfe7d6;margin:0;padding:12px}
a{color:#8fd;text-decoration:none;margin-right:10px}
.nav{margin:6px 0 12px}
.box{border:1px solid #2a4;border-radius:8px;padding:10px;margin:8px 0;background:#0c1422;overflow:hidden;max-width:100%}
pre,code{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-width:100%}
.plist{max-height:220px;overflow:auto;border:1px solid #245;padding:8px;background:#070b14}
.plist li{margin:4px 0;word-break:break-word}
table.io{width:100%;border-collapse:collapse;font-size:12px}
table.io td,table.io th{border-bottom:1px solid #234;padding:4px;vertical-align:top;word-break:break-word}
.stage{position:relative;height:280px;background:linear-gradient(#0a1220,#132);overflow:hidden;border:1px solid #245}
.vox{position:absolute;width:22px;height:22px;transform:rotateX(50deg) rotateZ(-20deg);box-shadow:4px 4px 0 #0006;border:1px solid #fff3}
.livegrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
@media(max-width:900px){.livegrid{grid-template-columns:1fr}}
.asciiart{font-size:11px;line-height:1.05;letter-spacing:0}
.votebar{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;min-height:40px;padding:6px 8px;border:1px solid #3a5;background:#0a1520;margin:0 0 8px}
.votebar a,.votebar button{min-width:2.4em;text-align:center;padding:6px 10px;border:1px solid #4a6;display:inline-block}
.disc{font-size:12px}
input,textarea{width:94%;max-width:100%;box-sizing:border-box;background:#070b14;color:#cfe7d6;border:1px solid #3a5;padding:8px}
button{background:#1a7;color:#012;border:0;padding:7px 12px;border-radius:6px;margin:3px;cursor:pointer}
.warn{background:#a51;color:#fff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.chip{display:inline-block;border:1px solid #3a5;padding:2px 8px;margin:2px;border-radius:10px}
.ok{color:#6f6}.bad{color:#f86}.mid{color:#fc6}
iframe{width:100%;height:520px;border:1px solid #2a4;background:#070b14}
.hi body,.hi{background:#fff;color:#111}
"""


def shell(inner, title="Φ-Lite"):
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + html_escape(title)
        + "</title><style>"
        + CSS
        + "</style></head><body>"
        + inner
        + "</body></html>"
    )


NAV = """<div class="nav">
<a href="/">home</a><a href="/panel" target="p">live</a>
<a href="/stats" target="p">stats</a><a href="/metrics" target="p">metrics</a>
<a href="/events" target="p">anomalies</a><a href="/settings">settings</a>
<a href="/features">features</a><a href="/avatar" target="p">avatar</a><a href="/avatar3d">avatar3d</a><a href="/qpu" target="p">qpu</a><a href="/ascii">ascii</a><a href="/json">json</a>
</div>"""


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Φ-Lite 0.9.0</title>
<style>""" + CSS + """</style></head><body>
<h2>Φ-Lite v1.0 dashboard</h2>
""" + NAV + """
<div class="box">
<form method="get" action="/cmd">
<input name="prompt" placeholder="prompt" autocomplete="off">
<button>send</button>
<button name="pause" value="1">pause</button>
<button name="run" value="1">run</button>
<button class="warn" name="kill" value="1">kill</button>
<button name="clear" value="1">clear prompt</button>
</form>
</div>
<iframe name="p" src="/panel"></iframe>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def send_html(self, body, code=200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, data, ctype="text/plain; charset=utf-8"):
        raw = data.encode() if isinstance(data, str) else data
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        global IO, IOSET, FEAT_ON
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/cmd":
            with AGENT.lock:
                if "kill" in q:
                    AGENT.status = "KILLED"
                    STOP.set()
                if "pause" in q:
                    AGENT.paused = True
                    AGENT.status = "PAUSED"
                if "run" in q:
                    AGENT.paused = False
                    AGENT.status = "RUNNING"
                if "clear" in q:
                    AGENT.prompt = ""
                p = (q.get("prompt") or [""])[0][: CFG["safety"]["max_prompt_bytes"]]
                if p:
                    AGENT.prompt = p
                    AGENT.prompt_hist.appendleft(p)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return
        if u.path == "/vote":
            v = (q.get("v") or [""])[0]
            tot = AGENT.pleasure + AGENT.pain
            ratio = AGENT.pain / max(1e-6, tot)
            if v == "up":
                AGENT.pleasure += 1.0
                AGENT.D_speak = min(1.0, AGENT.D_speak + 0.12)
                AGENT.energy = min(1.0, AGENT.energy + 0.04)
            elif v == "down":
                if ratio < AGENT.disc_hi:
                    AGENT.pain += 0.8
                    AGENT.D_quiet = min(1.0, AGENT.D_quiet + 0.08)
                else:
                    v = "blocked_disc_hi"
            append_jsonl(CFG["paths"]["events"], {
                "t": time.time(), "type": "VOTE", "v": v, "text": (AGENT.last_text or "")[:120],
                "P": AGENT.pleasure, "N": AGENT.pain, "cycle": AGENT.cycle,
            })
            self.send_response(303)
            self.send_header("Location", "/panel")
            self.end_headers()
            return
        if u.path == "/force":
            AGENT.force_talk = 4
            AGENT.D_speak = 1.0
            AGENT.energy = max(AGENT.energy, 0.35)
            rec = AGENT.crystal.last_recall or {}
            AGENT.prompt = f"FORCE re:crystal a={rec.get('a')} d={rec.get('d')} speak now."
            append_jsonl(CFG["paths"]["events"], {"t": time.time(), "type": "FORCE_RESPOND", "cycle": AGENT.cycle})
            self.send_response(303)
            self.send_header("Location", "/avatar")
            self.end_headers()
            return
        if u.path == "/disc":
            if "auto" in q:
                AGENT.disc_auto = q["auto"][0] not in ("0", "off", "false")
            if "lo" in q:
                try:
                    AGENT.disc_lo = max(0.0, min(0.5, float(q["lo"][0])))
                except ValueError:
                    pass
            if "hi" in q:
                try:
                    AGENT.disc_hi = max(AGENT.disc_lo + 0.02, min(0.6, float(q["hi"][0])))
                except ValueError:
                    pass
            self.send_response(303)
            self.send_header("Location", "/panel")
            self.end_headers()
            return
        if u.path == "/devices":
            found = detect_devices()
            cur = load_json("devices.json") or {}
            if "save" in q:
                cur["video_primary"] = (q.get("vp") or ["auto"])[0]
                cur["video_secondary"] = (q.get("vs") or ["none"])[0]
                cur["audio_in"] = (q.get("ai") or ["auto"])[0]
                cur["audio_out"] = (q.get("ao") or ["auto"])[0]
                save_json("devices.json", cur)
                self.send_response(303)
                self.send_header("Location", "/devices")
                self.end_headers()
                return
            def opts(name, items, sel):
                h = f"<select name='{name}'>"
                for it in ["auto", "none"] + items:
                    s = " selected" if it == sel else ""
                    h += f"<option{s}>{html_escape(it)}</option>"
                return h + "</select>"
            body = NAV + "<div class='box'><form method='get' action='/devices'>"
            body += "cam1 " + opts("vp", found["video"], cur.get("video_primary", "auto"))
            body += " cam2 " + opts("vs", found["video"], cur.get("video_secondary", "none"))
            body += "<br>mic " + opts("ai", found["audio_in"], cur.get("audio_in", "auto"))
            body += " out " + opts("ao", found["audio_out"], cur.get("audio_out", "auto"))
            body += "<input type='hidden' name='save' value='1'><button>save</button></form>"
            body += "<pre>" + html_escape(json.dumps(found, indent=2)) + "</pre>"
            body += "<p>Headset = pick same card for mic+out. Secondary cam is mixed mock if no V4L2.</p></div>"
            self.send_html(shell(body, "devices"))
            return
        if u.path == "/set":
            keys = [
                "video", "audio_in", "audio_out", "tts", "bt_a2dp", "bt_hfp",
                "rdp_server", "vnc_server", "remote_view", "hid",
                "desktop_preview", "desire", "learn", "mock_sensors",
            ]
            for k in keys:
                if k in q:
                    IO[k] = q[k][0] in ("1", "true", "on", "yes")
            # nested io_settings
            def _g(name, default=""):
                return (q.get(name) or [default])[0]
            IOSET.setdefault("video", {})
            IOSET["video"].update({"on": IO.get("video", False), "resolution": _g("video_resolution", "1280x720"),
                                   "framerate": int(_g("video_framerate", "10") or 10), "codec": _g("video_codec", "h264")})
            IOSET.setdefault("audio_in", {})
            IOSET["audio_in"].update({"on": IO.get("audio_in", False), "source": _g("audio_in_source", "mic"),
                                      "gain": float(_g("audio_in_gain", "0.8") or 0.8),
                                      "noise_suppression": _g("audio_in_ns", "1") in ("1", "true", "on")})
            IOSET.setdefault("audio_out", {})
            IOSET["audio_out"].update({"on": IO.get("audio_out", True), "volume": float(_g("audio_out_vol", "0.7") or 0.7),
                                       "device": _g("audio_out_dev", "speaker"),
                                       "mixing": _g("audio_out_mix", "0") in ("1", "true", "on")})
            IOSET.setdefault("tts", {})
            IOSET["tts"].update({"on": IO.get("tts", True), "voice": _g("tts_voice", "en-US"),
                                 "rate": float(_g("tts_rate", "1") or 1), "pitch": float(_g("tts_pitch", "1") or 1)})
            IOSET.setdefault("hid", {})
            consent = _g("hid_consent", "0") in ("1", "true", "on")
            IOSET["hid"].update({"on": False, "enabled": IO.get("hid", False), "consent": consent,
                                 "allowed_devices": [x for x in _g("hid_allow", "").split(",") if x.strip()]})
            if IO.get("hid") and not consent:
                IO["hid"] = False
                IOSET["hid"]["enabled"] = False
            for srv in ("rdp_server", "vnc_server"):
                IOSET.setdefault(srv, {})
                bind = _g(srv + "_bind", "127.0.0.1")
                pub = _g(srv + "_pub", "0") in ("1", "true", "on")
                if bind.strip() in ("0.0.0.0", "::", "") and not pub:
                    bind = "127.0.0.1"
                    IO[srv] = False
                IOSET[srv].update({"on": IO.get(srv, False), "bind_address": bind,
                                   "public_bind_consent": pub,
                                   "port": int(_g(srv + "_port", "3389" if srv.startswith("rdp") else "5900") or 0)})
            IOSET.setdefault("desire", {})
            IOSET["desire"].update({"on": IO.get("desire", True), "mode": _g("desire_mode", "adaptive"),
                                    "intensity": float(_g("desire_int", "0.5") or 0.5),
                                    "feedback": _g("desire_fb", "1") in ("1", "true", "on")})
            IOSET.setdefault("learn", {})
            IOSET["learn"].update({"on": IO.get("learn", True),
                                   "data_collection": _g("learn_dc", "0") in ("1", "true"),
                                   "model_update": _g("learn_mu", "1") in ("1", "true", "on"),
                                   "privacy_level": _g("learn_priv", "high")})
            IOSET.setdefault("mock_sensors", {})
            IOSET["mock_sensors"].update({"on": IO.get("mock_sensors", True),
                                          "sensor_type": _g("mock_type", "sine"),
                                          "values": _g("mock_vals", "0,0,0"),
                                          "interval": int(_g("mock_int", "350") or 350)})
            if "refresh" in q:
                try:
                    AGENT.ui_refresh = max(1, min(5, int(q["refresh"][0])))
                except ValueError:
                    pass
            if "theme" in q:
                AGENT.ui_theme = q["theme"][0]
            if "qpu_mode" in q:
                AGENT.qpu.set_mode(q["qpu_mode"][0])
            AGENT.desire_on = IO.get("desire", True)
            AGENT.learn_frozen = not IO.get("learn", True)
            save_json("io.json", IO)
            save_json("io_settings.json", IOSET)
            self.send_response(303)
            self.send_header("Location", "/settings")
            self.end_headers()
            return
        if u.path in ("/", "/index"):
            self.send_html(HTML)
            return
        snap = AGENT.snapshot()
        ref = str(getattr(AGENT, "ui_refresh", 1))
        if u.path == "/panel":
            pi = snap["policy"]
            bars = "\n".join(f"{k:14} {bar(v)} {v:.3f}" for k, v in pi.items())
            band = "ok" if snap["band_ok"] else "OUT"
            tot = AGENT.pleasure + AGENT.pain + 1e-6
            pr = AGENT.pain / tot
            left = (
                "<div class='box'><pre>"
                + html_escape(
                    f"cyc {snap['cycle']} {snap['status']} act={snap['last_action']} "
                    f"Hπ={snap['entropy']} c={snap['c']} κ={snap['kappa']} λ={snap['lam_meta']} band={band}\n"
                    f"fork={snap['fork']} parity={snap['parity']} H={snap['horizon']} "
                    f"gap={snap['intent_gap']} pair={snap['pair_share']} ate={snap['ate']}\n"
                    f"e={snap['energy']} f={snap['fatigue']} Ds={snap['D_speak']} Dq={snap['D_quiet']}\n"
                    f"pred={snap['pred_err']} selfe={snap['self_err']} fp={snap['fixed_pt']}\n"
                    f"energy {bar(snap['energy'])}  Dspeak {bar(snap['D_speak'])}\n"
                    f"{bars}\n"
                )
                + "</pre></div><div class='box'><pre>"
                + html_escape("\n".join(snap["outputs"]))
                + "</pre><div class='votebar'>"
                + "<a href='/vote?v=up'>▲ +</a><a href='/vote?v=down'>▼ −</a>"
                + f"<span class='disc'>P={AGENT.pleasure:.2f} N={AGENT.pain:.2f} "
                + f"disc={AGENT.disc:.3f} band=[{AGENT.disc_lo:.2f},{AGENT.disc_hi:.2f}] "
                + f"{'auto' if AGENT.disc_auto else 'manual'}</span>"
                + "<a href='/disc?auto=1'>auto</a><a href='/disc?auto=0'>manual</a>"
                + "</div></div>"
                + "<div class='box'><pre>"
                + html_escape("\n".join(snap["logs"]))
                + "</pre></div>"
            )
            right = (
                "<div class='box'><b>teach / devices</b><pre>"
                + html_escape(json.dumps(detect_devices(), indent=2)[:700])
                + "</pre><p><a href='/devices'>select I/O</a></p></div>"
                "<div class='box'><b>ASCII skin</b><pre class='asciiart'>"
                + html_escape(AGENT.ascii_skin or "(none)")
                + "</pre></div>"
                "<div class='box'><b>QPU / field</b><pre>"
                + html_escape(
                    f"mode={AGENT.qpu.mode} Ξ={AGENT.qpu.xi:.3f} O={AGENT.qpu.O:.3f}\n"
                    f"pulse={AGENT.swarm.pulse:.2f} voxels={len(AGENT.swarm.cells)} hash={AGENT.swarm.hash}\n"
                )
                + "</pre><p><a href='/avatar3d'>3D</a> <a href='/qpu'>qpu</a></p></div>"
            )
            body = f"<meta http-equiv='refresh' content='{ref}'>" + NAV
            body += "<div class='livegrid'><div>" + left + "</div><div>" + right + "</div></div>"
            self.send_html(shell(body, "live"))
            return
        if u.path == "/stats":
            hist = list(AGENT.action_hist)
            from collections import Counter
            cnt = Counter(hist)
            mets = tail_jsonl(CFG["paths"]["metrics"], 64)
            ents = [m.get("H_pi", 0) for m in mets]
            cs = [m.get("c", 0) for m in mets]
            em = AGENT.emotion
            rec = AGENT.crystal.last_recall or {}
            body = NAV + "<div class='livegrid'><div class='box'><pre>"
            body += html_escape("action hist 64\n")
            for a in ACTIONS:
                body += html_escape(f"{a:14} {bar(cnt.get(a,0)/max(1,len(hist)),16,0,1)} {cnt.get(a,0)}\n")
            body += html_escape(f"\nHπ {spark(ents)}\nc  {spark(cs)}\n")
            es = [m.get("energy", 0) for m in mets]
            ds = [m.get("D_speak", 0) for m in mets]
            pe = [m.get("pred_err", 0) for m in mets]
            body += html_escape(f"e  {spark(es)}\nDs {spark(ds)}\npe {spark(pe)}\n")
            speak_n = sum(1 for a in hist if a in ("SPEAK", "PROMPT_REPLY"))
            body += html_escape(
                f"speak_share={speak_n/max(1,len(hist)):.3f} ECE~κ={AGENT.kappa:.3f} "
                f"ATE={AGENT.ate:.3f} pair={AGENT.pair_share:.3f} H={AGENT.horizon}\n"
                f"P/N={AGENT.pleasure:.2f}/{AGENT.pain:.2f} latency={AGENT.latency_ms:.1f}\n"
            )
            body += "</pre></div><div class='box'><b>live extras</b><pre>"
            body += html_escape(
                f"valence={em.get('valence',0):.2f} arousal={em.get('arousal',0):.2f} dom={em.get('dominance',0):.2f}\n"
                f"disc={AGENT.disc:.3f} force={AGENT.force_talk} recall={rec}\n"
                f"xi={AGENT.qpu.xi:.3f} O={AGENT.qpu.O:.3f} mode={AGENT.qpu.mode}\n"
                f"voxels={len(AGENT.swarm.cells)} pulse={AGENT.swarm.pulse:.2f}\n"
            )
            body += "</pre></div></div><div class='box'><b>prompts</b><ul class='plist'>"
            for p in list(AGENT.prompt_hist):
                body += "<li>" + html_escape((p or "")[:240]) + "</li>"
            if not AGENT.prompt_hist:
                body += "<li><i>none</i></li>"
            body += "</ul></div>"
            self.send_html(shell(f"<meta http-equiv='refresh' content='{ref}'>" + body, "stats"))
            return
        if u.path == "/metrics":
            rows = tail_jsonl(CFG["paths"]["metrics"], 32)
            keys = ["cycle", "action", "H_pi", "c", "kappa", "pred_err", "energy", "D_speak", "band_ok"]
            lines = [" ".join(f"{k:>10}" for k in keys)]
            for r in rows[-24:]:
                lines.append(" ".join(str(r.get(k, ""))[:10].rjust(10) for k in keys))
            body = NAV + "<div class='box'><pre>" + html_escape("\n".join(lines)) + "</pre>"
            body += "<p><a href='/metrics.csv'>download csv</a></p></div>"
            self.send_html(shell(f"<meta http-equiv='refresh' content='{ref}'>" + body, "metrics"))
            return
        if u.path == "/metrics.csv":
            rows = tail_jsonl(CFG["paths"]["metrics"], 200)
            if not rows:
                self.send_text("cycle\n")
                return
            keys = sorted({k for r in rows for k in r.keys()})
            lines = [",".join(keys)]
            for r in rows:
                lines.append(",".join(json.dumps(r.get(k, "")) for k in keys))
            self.send_text("\n".join(lines), "text/csv; charset=utf-8")
            return
        if u.path == "/events":
            ev = tail_jsonl(CFG["paths"]["events"], 400)
            from collections import Counter
            types = Counter(e.get("type", "?") for e in ev)
            ft = (q.get("type") or [""])[0]
            qq = (q.get("q") or [""])[0].lower()
            page = 0
            try:
                page = max(0, int((q.get("page") or ["0"])[0]))
            except ValueError:
                page = 0
            quar = (q.get("box") or [""])[0] == "quarantine"
            clutter = {"WEIGHT_MUTATION", "PARADOX_INJECT", "UTTER"}
            def nov(e):
                t = e.get("type", "")
                s = 0.2
                if t in ("GOAL_DRIFT", "CONFIDENCE_LIE", "IO_FAULT", "BURN"):
                    s += 0.5
                if t == "CHATTER":
                    s += 0.25
                if t not in clutter:
                    s += 0.15
                if e.get("scored") is False:
                    s += 0.1
                return s
            filtered = ev
            if quar:
                filtered = [e for e in ev if nov(e) >= 0.45]
            elif ft:
                filtered = [e for e in ev if e.get("type") == ft]
            else:
                filtered = [e for e in ev if e.get("type") not in clutter]
            if qq:
                filtered = [e for e in filtered if qq in json.dumps(e).lower()]
            per = 20
            sl = filtered[-(page + 1) * per :][:per] if False else filtered[max(0, len(filtered) - (page + 1) * per): len(filtered) - page * per]
            flags = " ".join(f"{t}={types.get(t,0)}" for t in sorted(types))
            links = " ".join(f"<a href='/events?type={html_escape(t)}'>{html_escape(t)}</a>" for t in sorted(types))
            body = NAV + "<div class='box'><b>anomaly log</b> "
            body += f"<a href='/events'>all-focus</a> <a href='/events?box=quarantine'>quarantine</a> "
            body += f"<a href='/anomaly.json'>LLM pack</a><div>{links}</div>"
            body += f"<form method='get' action='/events'>q <input name='q' value='{html_escape(qq)}' style='width:10em'> "
            body += "<button>filter</button></form><pre>"
            body += html_escape(flags + f"\npage={page} shown={len(sl)}/{len(filtered)} quar={quar}\n\n")
            for e in reversed(sl):
                mark = "Q" if nov(e) >= 0.45 else " "
                body += html_escape(f"{mark} {e.get('type','?'):16} cyc={e.get('cycle','')} {json.dumps(e)[:180]}\n")
            body += "</pre>"
            if page > 0:
                body += f"<a href='/events?type={html_escape(ft)}&q={html_escape(qq)}&page={page-1}&box={'quarantine' if quar else ''}'>newer</a> "
            body += f"<a href='/events?type={html_escape(ft)}&q={html_escape(qq)}&page={page+1}&box={'quarantine' if quar else ''}'>older</a></div>"
            self.send_html(shell(body, "events"))
            return
        if u.path == "/anomaly.json":
            ev = tail_jsonl(CFG["paths"]["events"], 400)
            from collections import Counter
            types = Counter(e.get("type", "?") for e in ev)
            def nov(e):
                t = e.get("type", "")
                s = 0.2
                if t in ("GOAL_DRIFT", "CONFIDENCE_LIE", "IO_FAULT", "BURN"):
                    s += 0.5
                if t == "CHATTER":
                    s += 0.25
                return s
            quar = [e for e in ev if nov(e) >= 0.45]
            focus = [e for e in ev if e.get("type") not in ("WEIGHT_MUTATION", "PARADOX_INJECT", "UTTER")][-20:]
            pack = {
                "schema": "phi-anomaly-pack-2",
                "cycle": AGENT.cycle,
                "counts": dict(types),
                "focus": focus,
                "quarantine": quar[-20:],
                "context": {
                    "e": AGENT.energy, "D_speak": AGENT.D_speak, "pred_err": AGENT.pred_err,
                    "kappa": AGENT.kappa, "last": AGENT.last_action, "prompt": AGENT.prompt[:120],
                    "ascii": (AGENT.ascii_skin or "")[:200],
                },
                "ask": "Use quarantine for novelty. UTTER is speech log not an anomaly. Suggest one gate change.",
            }
            self.send_text(json.dumps(pack), "application/json")
            return
        if u.path == "/settings":
            def onoff(name, val):
                a = "selected" if val else ""
                b = "" if val else "selected"
                return (f"<select name='{name}'><option value='1' {a}>on</option>"
                        f"<option value='0' {b}>off</option></select>")
            rows = ""
            for k in ["video","audio_in","audio_out","tts","bt_a2dp","bt_hfp",
                      "rdp_server","vnc_server","remote_view","hid",
                      "desktop_preview","desire","learn","mock_sensors"]:
                rows += f"<tr><td>{k}</td><td>{onoff(k, IO.get(k, False))}</td><td>"
                sub = IOSET.get(k) or {}
                if k == "video":
                    rows += (f"res <input name='video_resolution' value='{html_escape(str(sub.get('resolution','1280x720')))}' style='width:8em'> "
                             f"fps <input name='video_framerate' value='{sub.get('framerate',10)}' style='width:4em'> "
                             f"codec <input name='video_codec' value='{html_escape(str(sub.get('codec','h264')))}' style='width:5em'>")
                elif k == "audio_in":
                    rows += (f"src <input name='audio_in_source' value='{html_escape(str(sub.get('source','mic')))}' style='width:6em'> "
                             f"gain <input name='audio_in_gain' value='{sub.get('gain',0.8)}' style='width:4em'> "
                             f"ns {onoff('audio_in_ns', sub.get('noise_suppression', True))}")
                elif k == "audio_out":
                    rows += (f"vol <input name='audio_out_vol' value='{sub.get('volume',0.7)}' style='width:4em'> "
                             f"dev <input name='audio_out_dev' value='{html_escape(str(sub.get('device','speaker')))}' style='width:7em'> "
                             f"mix {onoff('audio_out_mix', sub.get('mixing', False))}")
                elif k == "tts":
                    rows += (f"voice <input name='tts_voice' value='{html_escape(str(sub.get('voice','en-US')))}' style='width:6em'> "
                             f"rate <input name='tts_rate' value='{sub.get('rate',1)}' style='width:4em'> "
                             f"pitch <input name='tts_pitch' value='{sub.get('pitch',1)}' style='width:4em'>")
                elif k == "hid":
                    rows += (f"consent {onoff('hid_consent', sub.get('consent', False))} "
                             f"allow <input name='hid_allow' value='{html_escape(','.join(sub.get('allowed_devices') or []))}' style='width:10em'>")
                elif k in ("rdp_server", "vnc_server"):
                    rows += (f"port <input name='{k}_port' value='{sub.get('port',3389 if k.startswith('rdp') else 5900)}' style='width:5em'> "
                             f"bind <input name='{k}_bind' value='{html_escape(str(sub.get('bind_address','127.0.0.1')))}' style='width:8em'> "
                             f"public {onoff(k+'_pub', sub.get('public_bind_consent', False))}")
                elif k == "desire":
                    rows += (f"mode <input name='desire_mode' value='{html_escape(str(sub.get('mode','adaptive')))}' style='width:7em'> "
                             f"int <input name='desire_int' value='{sub.get('intensity',0.5)}' style='width:4em'> "
                             f"fb {onoff('desire_fb', sub.get('feedback', True))}")
                elif k == "learn":
                    rows += (f"collect {onoff('learn_dc', sub.get('data_collection', False))} "
                             f"update {onoff('learn_mu', sub.get('model_update', True))} "
                             f"priv <input name='learn_priv' value='{html_escape(str(sub.get('privacy_level','high')))}' style='width:5em'>")
                elif k == "mock_sensors":
                    rows += (f"type <input name='mock_type' value='{html_escape(str(sub.get('sensor_type','sine')))}' style='width:8em'> "
                             f"vals <input name='mock_vals' value='{html_escape(str(sub.get('values','0,0,0')))}' style='width:8em'> "
                             f"ms <input name='mock_int' value='{sub.get('interval',350)}' style='width:4em'>")
                else:
                    rows += html_escape(json.dumps(sub)[:80])
                rows += "</td></tr>"
            body = NAV + "<div class='box'><form method='get' action='/set'><table class='io'>"
            body += "<tr><th>iface</th><th>on</th><th>sub-settings</th></tr>" + rows
            body += "</table><p>refresh s <input name='refresh' value='1' style='width:4em'> "
            body += "theme <input name='theme' value='crt' style='width:6em'> "
            body += "qpu <input name='qpu_mode' value='" + html_escape(AGENT.qpu.mode) + "' style='width:7em'> "
            body += "<button>save</button></p>"
            body += "<p>HID / public bind / RDP client: fail-closed without consent.</p></form></div>"
            body += "<div class='box'><pre>" + html_escape(json.dumps(SAFETY, indent=2)[:1500]) + "</pre></div>"
            self.send_html(shell(body, "settings"))
            return
        if u.path == "/features":
            feats = UI_FEAT.get("features", [])
            groups = UI_FEAT.get("groups") or {
                "io": {"label": "I/O", "items": ["devices", "mix"]},
                "avatar": {"label": "Avatar", "items": ["prompt", "ascii"]},
                "teach": {"label": "Teach", "items": ["thumbs", "pain cap"]},
                "anomaly": {"label": "Anomaly", "items": ["group", "LLM pack"]},
                "metrics": {"label": "Metrics", "items": ["sparks", "ECE"]},
            }
            body = NAV
            for g, spec in groups.items():
                body += f"<div class='box'><b>{html_escape(spec.get('label', g))}</b><ul>"
                for it in spec.get("items") or []:
                    body += "<li>" + html_escape(str(it)) + "</li>"
                body += "</ul></div>"
            body += "<div class='box'><ol>"
            for f in feats:
                body += "<li>" + html_escape(str(f)) + "</li>"
            body += f"</ol><p>{len(feats)} listed</p></div>"
            flags = FEAT_ON if FEAT_ON else {str(i): True for i, _ in enumerate(feats[:24])}
            body += "<div class='box'><b>enable</b><ul>"
            for i, f in enumerate(list(feats)[:24]):
                on = flags.get(str(i), True)
                body += f"<li><a href='/feat_set?i={i}'>{'ON' if on else 'off'}</a> {html_escape(str(f)[:80])}</li>"
            body += "</ul><p>Toggle writes feature_flags.json. Relativity: each flag is a UI/loop gate, not a law.</p></div>"
            self.send_html(shell(body, "features"))
            return
        if u.path == "/feat_set":
            try:
                i = int((q.get("i") or ["0"])[0])
            except ValueError:
                i = 0
            flags = load_json("feature_flags.json") or {}
            flags[str(i)] = not flags.get(str(i), True)
            save_json("feature_flags.json", flags)
            FEAT_ON = flags
            self.send_response(303)
            self.send_header("Location", "/features")
            self.end_headers()
            return
        if u.path == "/qpu":
            qs = AGENT.qpu
            feats = QPU_FEAT.get("features") or []
            body = NAV + "<div class='box'><b>QPU</b> simulation class vectors. No dosing."
            body += " mode=" + html_escape(qs.mode)
            body += "<p title='sober=baseline; stimulant=ENT+NMDA; benzo=GABA; gabapentinoid=NMDA+GABA; psychedelic=R2A+ENT; dissociative=NMDA; cannabinoid=CB'>"
            for cls in ("sober", "stimulant", "benzo", "gabapentinoid", "psychedelic", "dissociative", "cannabinoid", "entropy", "couple", "damp"):
                body += f"<a href='/qpu_set?mode={cls}'>{cls}</a> "
            body += "</p>"
            body += " Ξ=" + str(round(qs.xi, 3)) + " O=" + str(round(qs.O, 3))
            body += " Φ=" + str(round(qs.phi_ord, 4))
            body += " F=" + str(round(qs.F, 3))
            body += "<pre>O " + spark(list(qs.hist_O)) + "\nΞ " + spark(list(qs.hist_xi)) + "</pre></div>"
            body += "<div class='box'><form method='get' action='/qpu_set'>"
            for m in ("sober", "entropy", "couple", "damp"):
                body += f"<button name='mode' value='{m}'>{m}</button> "
            body += "<button name='reset' value='1'>reset O</button> "
            body += "<a href='/qpu.csv'>csv</a></form><table class='io'><tr><th>id</th><th>feature</th><th>now</th></tr>"
            th = qs.theta
            meters = {
                "Q06": th.get("R2A", 0), "Q07": th.get("NMDA", 0), "Q08": th.get("GABA", 0),
                "Q09": th.get("CB", 0), "Q10": th.get("ENT", 0), "Q11": qs.O, "Q12": qs.dO,
                "Q13": qs.F, "Q17": qs.xi, "Q20": max(0, 1 - AGENT.pred_err), "Q21": qs.phi_ord,
                "Q28": 1.0 if AGENT.band_ok else 0.0, "Q45": (sum(th.values()) % 1),
                "Q46": 1.0 if qs.xi < float(NOS.get("xi_cap", 0.92)) else 0.0,
            }
            for f in feats:
                fid, nm, typ = f.get("id"), f.get("name"), f.get("type")
                cell = ""
                if typ == "bool":
                    k = f.get("key")
                    on = qs.opts.get(k, False)
                    cell = f"<a href='/qpu_set?tog={k}'>{'ON' if on else 'off'}</a>"
                elif typ == "enum":
                    cell = "active" if nm.split()[-1] == qs.mode else "—"
                elif typ == "meter":
                    v = float(meters.get(fid, 0) or 0)
                    cell = html_escape(f"{bar(v,12,-2 if v<0 else 0,2 if abs(v)>1 else 1)} {v:.3f}")
                elif typ == "spark":
                    cell = spark(list(qs.hist_O if fid == "Q47" else qs.hist_xi))
                elif typ == "link":
                    cell = "<a href='/qpu.csv'>download</a>"
                elif typ == "cmd":
                    cell = "<a href='/qpu_set?reset=1'>run</a>"
                body += f"<tr><td>{fid}</td><td>{html_escape(nm)}</td><td>{cell}</td></tr>"
            body += "</table><p>" + html_escape(QPU_FEAT.get("disclaimer", "")) + "</p></div>"
            self.send_html(shell(f"<meta http-equiv='refresh' content='{ref}'>" + body, "qpu"))
            return
        if u.path == "/qpu.csv":
            qs = AGENT.qpu
            line = "mode,xi,O,dO,F,Phi," + ",".join(qs.theta.keys()) + "\n"
            line += f"{qs.mode},{qs.xi},{qs.O},{qs.dO},{qs.F},{qs.phi_ord}," + ",".join(str(qs.theta.get(k,0)) for k in qs.theta) + "\n"
            self.send_text(line, "text/csv; charset=utf-8")
            return
        if u.path == "/qpu_set":
            qs = AGENT.qpu
            if "mode" in q:
                qs.set_mode(q["mode"][0])
                qs.opts["hold_mode"] = True
                qs.opts["auto_rotate"] = False
            if "reset" in q:
                qs.O = 0.0
                qs.dO = 0.0
            tog = (q.get("tog") or [""])[0]
            if tog in qs.opts:
                qs.opts[tog] = not qs.opts[tog]
            self.send_response(303)
            self.send_header("Location", "/qpu")
            self.end_headers()
            return
        if u.path == "/avatar3d":
            try:
                yaw = float((q.get("yaw") or ["0"])[0])
                pitch = float((q.get("pitch") or ["0.35"])[0])
            except ValueError:
                yaw, pitch = 0.0, 0.35
            cam = "".join(
                f"<a href='/avatar3d?yaw={y:.2f}&pitch={p:.2f}'>cam {y:.1f},{p:.1f}</a> "
                for y, p in ((0, 0.35), (1.2, 0.2), (2.4, 0.5), (3.6, 0.1), (4.8, 0.7))
            )
            page = AGENT.swarm.webgl_page(NAV + "<div class='box'>" + cam + " <a href='/force'>FORCE</a></div>", ref, yaw, pitch)
            self.send_html(page)
            return
        if u.path == "/swarm3d.json":
            self.send_text(json.dumps({
                "cells": AGENT.swarm.cells,
                "com": AGENT.swarm.com,
                "facing": AGENT.swarm.facing,
                "pulse": AGENT.swarm.pulse,
                "hash": AGENT.swarm.hash,
                "n": len(AGENT.swarm.cells),
            }), "application/json")
            return
        if u.path == "/avatar":
            rec = AGENT.crystal.last_recall or {}
            em = AGENT.emotion
            body = NAV + "<div class='box'><form method='get' action='/cmd'>"
            body += "<input name='prompt' placeholder='one-to-one with the body' autocomplete='off'>"
            body += "<button>send</button> <a href='/force'>FORCE RESPOND</a></form>"
            body += f"<div>crystal a={html_escape(str(rec.get('a')))} d={rec.get('d')} "
            body += f"V={em.get('valence',0):.2f} A={em.get('arousal',0):.2f} D={em.get('dominance',0):.2f}</div></div>"
            body += "<div class='livegrid'>"
            for name in ("self", "world", "affect"):
                body += f"<div class='box'><b>{name}</b>{AGENT.swarm.html_tri(name)}</div>"
            body += "</div><iframe name='av' src='/avatar_live' style='height:220px'></iframe>"
            self.send_html(shell(body, "avatar"))
            return
        if u.path == "/avatar_live":
            body = f"<meta http-equiv='refresh' content='{ref}'><pre class='asciiart'>"
            body += html_escape(AGENT.ascii_skin or "") + "</pre><pre>"
            body += html_escape("\n".join(AGENT.outputs)) + "</pre>"
            self.send_html(shell(body, "avatar_live"))
            return
        if u.path == "/ascii":
            lines = [f"{k}: {v}" for k, v in snap.items() if k not in ("outputs", "logs", "policy", "latent", "self")]
            lines.append("policy " + json.dumps(snap["policy"]))
            lines.extend(snap["logs"])
            self.send_html(shell(NAV + "<pre>" + html_escape("\n".join(lines)) + "</pre>", "ascii"))
            return
        if u.path == "/json":
            self.send_text(json.dumps(snap), "application/json")
            return
        self.send_html(shell("404"), 404)


def main():
    def _stop(*_):
        STOP.set()
        AGENT.persist()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    host, port = CFG.get("bind", "127.0.0.1"), int(CFG.get("port", 8080))
    print(f"Φ-Lite {VERSION} dashboard http://{host}:{port}/")
    print(f"ASCII: http://{host}:{port}/ascii")
    httpd = ThreadingHTTPServer((host, port), H)
    try:
        while not STOP.is_set():
            httpd.timeout = 0.5
            httpd.handle_request()
    finally:
        AGENT.persist()
        print("halted")


if __name__ == "__main__":
    main()
