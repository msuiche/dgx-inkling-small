# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small certified target LM head (port of DSv4.1 DSV41_R4_CERT_HEAD + r6 head: REFINE=row, SCREEN DQ2).

Install path: vllm/models/inkling2/ops/cert_head.py
Callers: ops/sync_free.py (target head, INKLING_CERT_HEAD) and inkling2/dspark.py (INKLING_DRAFT_HEAD=mxint8).

TARGET HEAD DTYPE: the live Inkling2 head is fp32-output (Inkling2MMConfig head_dtype='float32'): the stock head is
torch.mm(x_bf16, W^T, out_dtype=float32) and the soft cap is the fp32 chain torch.tanh(l / cap) * cap. The whole
certificate therefore works on fp32 logits (no bf16 rounding anywhere); sync_free only arms it when head_dtype is fp32.

Per call, on this rank's vocab shard W [V, K] (bf16, K = 4608):
  xnorm    ||x_g|| per 128-column group of every hidden row, ||x||, a NaN/inf flag
  screen   s = x . D^T with D an MXINT8 copy of W (int8 + one power-of-two exponent per 32 weights, 119 MB
           instead of 231 MB), and a proven per-logit bound
               B = sum_g ||x_g|| ||W_vg - D_vg||  +  2 c_acc(K) ||x|| max(||W_v||, ||D_v||)
           lo = fl32(s - B - slop), hi = fl32(s + B + slop) (slop = 2^-21 relative covers the fp32 rounding of the
           two sums, so lo <= true <= hi holds with round-to-nearest), T[m] = max_v lo (atomic)
  cand     per vocab row v: selected iff, for some row m, chain(hi[m, v]) >= chain(T[m]) (chain = the stock fp32
           soft cap tanh(x / cap) * cap, bit-equal to the torch ops over every finite fp32 and monotone, both
           verified exhaustively at first use); selected rows are listed, every other (m, v) is written NaN (the
           local argmax reads NaN as -inf, as the stock path does).
  refine   persistent grid over the list: exact bf16 MMA, fp32 accumulate, ascending K, stored as fp32 (no rounding).

Why the token is the stock greedy token (ties included): the stock logit l_v (fp32 cuBLAS sum) obeys
lo_v <= l_v <= hi_v. T <= max_u l_u, so chain(T) <= chain(l_u) <= max chain. A cut row has chain(hi_v) < chain(T)
hence chain(l_v) < max chain: it is neither the max nor tied with it. Rows tied at the max (after the soft cap;
saturation creates ties for large logits) all have chain(hi) >= max chain >= chain(T): they are all computed, so the
leftmost rule of the local argmax sees exactly the stock set. The bound is applied in the CHAIN domain on purpose: a
cut test hi_v < T on the raw logits would drop rows that the soft cap rounds to the same value as the max.
The refine reproduces the stock torch.mm(out_dtype=fp32) head bit for bit (proved per row count by oracle_cert.py
and re-proved on the real shard for M = 2..MAXM at start-up, see sync_free); unqualified M stays on the stock head.

Drafter: `dq_logits` is the screen without the bound (s rounded to bf16): the drafter's base logits read the
same int8 table instead of a private fp8 copy (INKLING_DRAFT_HEAD=mxint8), 0.87% weight error instead of 2.6%.

Env (read once; every TP rank must see identical values):
  INKLING_CERT_HEAD=off|1|check    1 = armed steps use the certified head (needs INKLING_SYNC_FREE=argmax|audit AND
                                   an fp32 head, otherwise nothing happens); check = run stock + certified, count
                                   mismatches, return the stock result
  INKLING_CERT_HEAD_MAXM=64        largest row count served (the kernels support <= 128)
  INKLING_CERT_HEAD_TILE=32,256,4,3  screen BN,BK,num_warps,num_stages
  INKLING_CERT_HEAD_REFINE=128,4,3,0  refine BK,num_warps,num_stages,grid (0 = 4 x SM count)
  INKLING_CERT_HEAD_LOG_EVERY=4000   log (one host read) cumulative candidate stats every N cert calls; 0 = never
  INKLING_CERT_HEAD_LISTCAP=6000     if the mean listed rows per call over a window of LOG_EVERY (or 4000) calls
                                   exceeds this, this rank falls back to the stock head for good; 0 = never
"""
from __future__ import annotations

import os

import numpy as np
import torch

try:  # CPU unit tests import this file without vllm
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("inkling_cert_head")

_OFF = ("", "0", "off", "false", "no")


def parse_mode(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in _OFF:
        return "off"
    if v in ("1", "on", "true", "argmax"):
        return "1"
    if v in ("check", "audit"):
        return "check"
    logger.warning("INKLING_CERT_HEAD=%r not recognised (expected off|1|check); feature OFF", raw)
    return "off"


def _ints(name: str, default: str, n: int) -> tuple:
    v = os.environ.get(name, default).strip() or default
    parts = [p.strip() for p in v.split(",")]
    if len(parts) != n or not all(p.lstrip("-").isdigit() for p in parts):
        raise ValueError(f"{name}={v!r}: expected {n} comma-separated integers")
    return tuple(int(p) for p in parts)


MODE = parse_mode(os.environ.get("INKLING_CERT_HEAD", "off"))
MAXM = max(1, min(128, int(os.environ.get("INKLING_CERT_HEAD_MAXM", "64") or 64)))
SBN, SBK, SWARPS, SSTAGES = _ints("INKLING_CERT_HEAD_TILE", "32,256,4,3", 4)
XBK, XWARPS, XSTAGES, XGRID = _ints("INKLING_CERT_HEAD_REFINE", "128,4,3,0", 4)
LOG_EVERY = int(os.environ.get("INKLING_CERT_HEAD_LOG_EVERY", "4000") or 0)
LISTCAP = int(os.environ.get("INKLING_CERT_HEAD_LISTCAP", "6000") or 0)
PERIOD = LOG_EVERY if LOG_EVERY > 0 else 4000  # window of the list-length cap (and of the log)
DRAFT_MXINT8 = os.environ.get("INKLING_DRAFT_HEAD", "").strip().lower() in ("mxint8", "int8", "1", "on", "true")

GROUP = 128
BV = 16  # refine rows per tile (tl.dot M dimension)

# Certificate constants (identical to DSv4.1 r4; inflations cost nothing measurable in candidates).
INFL_X = 1.0 + 2.0 ** -14   # group / total norm of x (fp32 sum of <= 128 squares + sqrt)
INFL_Q = 1.0 + 2.0 ** -14   # fp32 sum_g ||x_g|| R_vg
INFL_B = 1.0 + 2.0 ** -20   # final product / add
SLOP_REL = 2.0 ** -21       # fp32 rounding of s -/+ B
SLOP_ABS = 2.0 ** -126
X_ABS = 2.0 ** -59          # underflow of x_i^2 in a group


def c_acc(K: int) -> float:
    """Assumed fp32 tensor-core accumulation error per unit ||x|| ||w|| (K/16 MMA steps, each modelled as 17
    additions with one rounding of 2^-23; Fasi et al. 2021 style). oracle_cert.py measures the real error
    against float64 and reports the margin."""
    return (K / 16.0) * 17.0 * 2.0 ** -23


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _f32(x: float) -> float:
    return float(np.float32(x))


def inv_cap_of(cap: float) -> float:
    return float(np.float32(1.0) / np.float32(cap))


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# =====================================================================================================
# Tables (pure torch)
# =====================================================================================================
def round_up_f32(x64: torch.Tensor) -> torch.Tensor:
    """float64 (>= 0 or inf) -> smallest fp32 >= x."""
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, torch.nextafter(f, torch.full_like(f, float("inf"))), f)


def make_mxint8(w: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [V, K] -> (int8 [V, K], exponent byte uint8 [V, K/32]); D = q * 2^(e-127), |q| <= 127."""
    V, K = w.shape
    assert K % 32 == 0
    q = torch.empty((V, K), dtype=torch.int8, device=w.device)
    e = torch.empty((V, K // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        wf = w[r0:r1].float().view(r1 - r0, K // 32, 32)
        amax = wf.abs().amax(-1)
        ex = torch.where(amax > 0, torch.ceil(torch.log2(amax.clamp_min(1e-38) / 127.0)), torch.zeros_like(amax))
        eb = (ex + 127.0).clamp(1.0, 254.0)
        qq = torch.round(wf / torch.exp2(eb - 127.0)[..., None]).clamp(-127, 127)
        q[r0:r1] = qq.view(r1 - r0, K).to(torch.int8)
        e[r0:r1] = eb.to(torch.uint8)
    return q, e


def deq_rows(q: torch.Tensor, e: torch.Tensor, r0: int, r1: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows r0:r1 of D as the screen kernel builds them: (bf16-cast values as fp32, raw fp32 values)."""
    qf = q[r0:r1].float()
    sc = (e[r0:r1].to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, dim=1)
    raw = qf * sc
    return raw.to(torch.bfloat16).float(), raw


class CertTable:
    """Per-rank MXINT8 table of one lm_head shard; shared by the target head and the drafter base logits."""


@torch.inference_mode()
def build_table(W: torch.Tensor, vreal: int | None = None, group: int = GROUP) -> CertTable:
    """Tables for one rank's head shard W [V, K] bf16. Rows >= vreal are vocabulary padding: their weights may be
    anything (non-finite values are zeroed in the table; the screen never lets them raise T or be selected)."""
    V, K = W.shape
    tolerant = vreal is None  # drafter-side call (vocab size unknown): non-finite rows are sanitised, never fatal
    vreal = V if vreal is None else max(0, min(V, int(vreal)))
    if W.dtype != torch.bfloat16 or W.stride(1) != 1 or K % group or group % 32 or K % SBK or K % XBK:
        raise ValueError(f"head shard {tuple(W.shape)} {W.dtype} does not fit group={group} BK={SBK},{XBK}")
    if not tolerant and not bool(torch.isfinite(W[:vreal]).all()):
        raise ValueError("head shard has non-finite weights in the real vocabulary")
    NG = K // group
    q = torch.empty((V, K), dtype=torch.int8, device=W.device)
    e = torch.empty((V, K // 32), dtype=torch.uint8, device=W.device)
    rt = torch.empty((NG, V), dtype=torch.float32, device=W.device)
    nv = torch.empty((V,), dtype=torch.float32, device=W.device)
    n_unsafe = 0
    for r0 in range(0, V, 1024):
        r1 = min(V, r0 + 1024)
        wc = W[r0:r1]
        nonfin = ~torch.isfinite(wc).all(dim=1) if tolerant else None
        if r1 > vreal or tolerant:  # padding rows (or tolerant mode): sanitise
            wc = torch.nan_to_num(wc.float(), nan=0.0, posinf=0.0, neginf=0.0).to(torch.bfloat16)
        qc, ec = make_mxint8(wc)
        q[r0:r1], e[r0:r1] = qc, ec
        d_bf, d_raw = deq_rows(qc, ec, 0, r1 - r0)
        unsafe = (d_bf != d_raw).any(dim=1)  # the kernel's bf16 cast of D must be exact
        w64 = wc.double()
        res = (w64 - d_bf.double()).view(r1 - r0, NG, group)
        rg = round_up_f32(res.pow(2).sum(-1).sqrt() * (1.0 + 2.0 ** -40))
        if nonfin is not None:
            unsafe = unsafe | nonfin  # a sanitised row has no valid bound: it is always a candidate (exact refine)
        rg[unsafe] = float("inf")
        rt[:, r0:r1] = rg.t()
        nrm = torch.maximum(w64.norm(dim=1), d_bf.double().norm(dim=1)) * (1.0 + 2.0 ** -40)
        nv[r0:r1] = round_up_f32(nrm)
        n_unsafe += int(unsafe[: max(0, vreal - r0)].sum())
    t = CertTable()
    t.q, t.e, t.rt, t.nv = q, e, rt, nv
    t.V, t.K, t.group, t.n_unsafe, t.vreal = V, K, group, n_unsafe, vreal
    d0, _ = deq_rows(q, e, 0, min(V, 4096))
    w0 = W[: d0.shape[0]].float()
    t.eps = float(((w0 - d0).norm(dim=1) / w0.norm(dim=1).clamp_min(1e-30)).mean())
    t.bytes = q.numel() + e.numel() + rt.numel() * 4 + nv.numel() * 4
    t.exp_ok = int(e.min()) >= 1 and int(e.max()) <= 254
    return t



# INKLING_CERT_HEAD_CACHE=<dir>: persist the MXINT8 table (~123 MB/rank) and reload it
# instead of rebuilding. The table is a pure function of (W, vreal); the build spends
# ~12 s per rank inside fp64 residual math, which this skips. Bit-identical by
# construction (the same bytes are restored), and guarded by shape + a content hash of
# a strided sample of W so a different checkpoint can never load a stale table.
_CACHE_DIR = os.environ.get("INKLING_CERT_HEAD_CACHE", "").strip()


def _wkey(W: torch.Tensor) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(str(tuple(W.shape)).encode())
    h.update(str(W.dtype).encode())
    # sample rows/cols strided: cheap and sensitive enough to distinguish checkpoints
    sample = W[:: max(1, W.shape[0] // 64), :: max(1, W.shape[1] // 64)]
    h.update(memoryview(sample.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)))
    return h.hexdigest()[:16]


def _cache_load(W: torch.Tensor, vreal: int):
    if not _CACHE_DIR:
        return None
    try:
        path = os.path.join(_CACHE_DIR, f"cert-{_wkey(W)}-{W.shape[0]}x{W.shape[1]}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        t = CertTable()
        t.V, t.K, t.group = int(z["V"]), int(z["K"]), int(z["group"])
        t.vreal, t.n_unsafe = int(z["vreal"]), int(z["n_unsafe"])
        t.eps, t.bytes = float(z["eps"]), int(z["bytes"])
        t.exp_ok = int(z["exp_ok"])
        # keep the tensors on the same device as W (the kernels read them there)
        t.q = torch.from_numpy(z["q"]).to(W.device)
        t.e = torch.from_numpy(z["e"]).to(W.device)
        t.rt = torch.from_numpy(z["rt"]).to(W.device)
        t.nv = torch.from_numpy(z["nv"]).to(W.device)
        if t.q.shape != (t.V, t.K) or t.e.shape != (t.V, t.K // 32):
            return None
        return t
    except Exception:  # noqa: BLE001
        return None


def _cache_store(t) -> None:
    if not _CACHE_DIR:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        import hashlib
        w = getattr(t, "_wkey", None)
        if w is None:
            return
        path = os.path.join(_CACHE_DIR, f"cert-{w}-{t.V}x{t.K}.npz")
        tmp = path + ".tmp.npz"
        np.savez(tmp, q=t.q.cpu().numpy(), e=t.e.cpu().numpy(), rt=t.rt.cpu().numpy(),
                 nv=t.nv.cpu().numpy(), eps=np.float64(t.eps), bytes=np.int64(t.bytes),
                 V=np.int64(t.V), K=np.int64(t.K), group=np.int64(t.group),
                 vreal=np.int64(t.vreal), n_unsafe=np.int64(t.n_unsafe),
                 exp_ok=np.int64(t.exp_ok))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


_TABLES: dict = {}


def get_table(W: torch.Tensor, vreal: int | None = None) -> CertTable | None:
    """Build once per head weight tensor (target head and drafter share the aliased weight); None on failure."""
    key = (int(W.data_ptr()), tuple(W.shape))
    t = _TABLES.get(key)
    if t is not None:
        return t or None
    if _capturing():
        return None
    if _CACHE_DIR:
        ct = _cache_load(W, vreal)
        if ct is not None:
            _TABLES[key] = ct
            try:
                from . import l2pf
                l2pf.prewarm(l2pf.get_head_weight(num_tokens=8), kind="head")
            except Exception:
                pass
            logger.info("INKLING cert head: MXINT8 table %dx%d loaded from cache (%.1f MB)",
                        ct.V, ct.K, ct.bytes / 1e6)
            return ct
    try:
        t = build_table(W, vreal)
        if not t.exp_ok:
            raise ValueError("exponent byte outside 1..254")
        torch.cuda.empty_cache()
        if _CACHE_DIR:
            t._wkey = _wkey(W)
            _cache_store(t)
        logger.info("INKLING cert head: MXINT8 table %dx%d built: %.1f MB (int8 %.1f + exp %.1f + rt/nv %.1f), "
                    "rel eps %.4f%%, unsafe rows %d, vreal %d", t.V, t.K, t.bytes / 1e6, t.q.numel() / 1e6,
                    t.e.numel() / 1e6, (t.rt.numel() + t.nv.numel()) * 4 / 1e6, t.eps * 100, t.n_unsafe, t.vreal)
    except Exception as ex:  # noqa: BLE001
        logger.warning("INKLING cert head: table build failed (%r)", ex)
        t = False
    _TABLES[key] = t
    if t:
        try:
            from . import l2pf
            l2pf.prewarm(l2pf.get_head_weight(num_tokens=8), kind="head")
        except Exception:
            pass
    return t or None


def reference_bounds(t: CertTable, x: torch.Tensor, vreal: int, screen_acc: torch.Tensor | None = None):
    """Torch mirror of xnorm + screen: (lo, hi fp32 [M, V], T [M], bad [M])."""
    M, K = x.shape
    G, NG = t.group, K // t.group
    xf = x.float()
    xg = xf.view(M, NG, G).pow(2).sum(-1).sqrt() * INFL_X + X_ABS
    xn = xg.pow(2).sum(-1).sqrt() * INFL_X
    bad = ~torch.isfinite(xn)
    if screen_acc is None:
        d_bf, _ = deq_rows(t.q, t.e, 0, t.V)
        screen_acc = xf @ d_bf.t()
    qb = xg @ t.rt
    B = (qb * INFL_Q + 2.0 * c_acc(K) * (xn[:, None] * t.nv[None, :])) * INFL_B
    B = torch.where(torch.isnan(B), torch.full_like(B, float("inf")), B)
    slop = (screen_acc.abs() + B) * SLOP_REL + SLOP_ABS
    lo = screen_acc - B - slop
    hi = screen_acc + B + slop
    nonfin = ~torch.isfinite(screen_acc)
    lo = torch.where(nonfin | torch.isnan(lo), torch.full_like(lo, float("-inf")), lo)
    hi = torch.where(nonfin | torch.isnan(hi), torch.full_like(hi, float("inf")), hi)
    if vreal < t.V:
        lo[:, vreal:] = float("-inf")
        hi[:, vreal:] = float("-inf")
    T = lo.max(dim=1).values
    return lo, hi, T, bad


# =====================================================================================================
# Kernels
# =====================================================================================================
_K: dict = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tldevice

    @triton.jit
    def _chain(x, inv_cap, cap, DIVMODE: tl.constexpr):
        """LogitsProcessor soft cap on fp32 logits: tanh(x / cap) * cap as the three torch fp32 kernels.
        DIVMODE 0: x * (1/cap) (what torch does for a CPU-scalar divisor), 1: IEEE x / cap."""
        if DIVMODE == 0:
            y = x * inv_cap
        else:
            y = tldevice.div_rn(x, cap)
        t = tldevice.tanh(y)
        # libdevice tanh may flush fp32-denormal inputs to 0 (torch keeps them): tanh(y) == y there
        t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, t)
        return t * cap

    @triton.jit(do_not_specialize=["base", "n"])
    def _chain_sweep_kernel(X, Y, base, n, inv_cap, cap, DIVMODE: tl.constexpr, BLOCK: tl.constexpr):
        """Ordered sweep of every finite fp32: key = base + i enumerates the finite floats in ascending value
        order (key = ~bits for negatives, bits | 2^31 for non-negatives). Writes x and chain(x)."""
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        key = (base + i).to(tl.uint32)
        sign = tl.full((BLOCK,), 0x80000000, tl.uint32)
        bits = tl.where((key >> 31) != 0, key ^ sign, ~key)
        x = bits.to(tl.float32, bitcast=True)
        tl.store(X + i, x, mask=m)
        tl.store(Y + i, _chain(x, inv_cap, cap, DIVMODE), mask=m)

    @triton.jit
    def _xnorm_kernel(X, sx, XG, XN, BAD, THR, CNT, NLIVE, M, K: tl.constexpr, G: tl.constexpr, NGP: tl.constexpr,
                      INFL: tl.constexpr, XABS: tl.constexpr):
        m = tl.program_id(0)
        if m == 0:
            tl.store(CNT, 0)
        nl = tl.load(NLIVE)
        nl = tl.where((nl >= 1) & (nl <= M), nl, M)
        if m >= nl:
            tl.store(BAD + m, 0)
            tl.store(THR + m, float("-inf"))
            return
        NG: tl.constexpr = K // G
        gi = tl.arange(0, NGP)
        gok = gi < NG
        x = tl.load(X + m * sx + gi[:, None] * G + tl.arange(0, G)[None, :], mask=gok[:, None],
                    other=0.0).to(tl.float32)
        s = tl.sum(x * x, axis=1)
        xg = tl.sqrt_rn(s) * INFL + XABS
        xg = tl.where(gok, xg, 0.0)
        tl.store(XG + m * NG + gi, xg, mask=gok)
        xn = tl.sqrt_rn(tl.sum(xg * xg, axis=0)) * INFL
        tl.store(XN + m, xn)
        tl.store(BAD + m, ((xn != xn) | (xn == float("inf"))).to(tl.int32))
        tl.store(THR + m, float("-inf"))

    @triton.jit
    def _dq_tile(Q, E, n64, nok, k0, K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        # D tile [BK, BN] as bf16: top 16 bits of the exact fp32 product q * 2^(e-127) (DSv4.1 r6 DQ=2; exact
        # because |q| <= 127 has <= 7 significant bits and every exponent byte the builder emits is 1..254)
        G32: tl.constexpr = BK // 32
        k = k0 + tl.arange(0, BK)
        q = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0, eviction_policy="evict_first")
        eb = tl.load(E + n64[None, :] * (K // 32) + (k0 // 32 + tl.arange(0, G32))[:, None],
                     mask=nok[None, :], other=127, eviction_policy="evict_first")
        sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        qf = (q.to(tl.int32) + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0
        wf = tl.reshape(tl.reshape(qf, (G32, 32, BN)) * sc[:, None, :], (BK, BN))
        return (wf.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _screen_kernel(X, sx, Q, E, RT, NV, XG, XN, HI, THR, NLIVE, M, V, VREAL,
                       K: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, CACC2: tl.constexpr, INFL_Q: tl.constexpr, INFL_B: tl.constexpr,
                       SLOP_REL: tl.constexpr, SLOP_ABS: tl.constexpr):
        pid = tl.program_id(0)
        nl = tl.load(NLIVE)
        nl = tl.where((nl >= 1) & (nl <= M), nl, M)
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < nl
        nok = n < V
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            wd = _dq_tile(Q, E, n64, nok, k0, K, BK, BN)
            acc = tl.dot(x, wd, acc)
        NG: tl.constexpr = K // G
        qb = tl.zeros((BM, BN), dtype=tl.float32)
        for g in range(0, NG):
            r = tl.load(RT + g * V + n, mask=nok, other=0.0, eviction_policy="evict_first")
            xg = tl.load(XG + m * NG + g, mask=mok, other=0.0)
            qb += xg[:, None] * r[None, :]
        nv = tl.load(NV + n, mask=nok, other=0.0)
        xn = tl.load(XN + m, mask=mok, other=0.0)
        B = (qb * INFL_Q + CACC2 * (xn[:, None] * nv[None, :])) * INFL_B
        B = tl.where(B != B, float("inf"), B)
        slop = (tl.abs(acc) + B) * SLOP_REL + SLOP_ABS
        lo = acc - B - slop  # fp32; slop (2^-21 relative) covers the rounding of both sums
        hi = acc + B + slop
        nonfin = (acc != acc) | (tl.abs(acc) == float("inf"))
        lo = tl.where(nonfin | (lo != lo), float("-inf"), lo)
        hi = tl.where(nonfin | (hi != hi), float("inf"), hi)
        real = n < VREAL  # padding rows never raise T and are never needed
        lo = tl.where(real[None, :], lo, float("-inf"))
        hi = tl.where(real[None, :], hi, float("-inf"))
        ok = mok[:, None] & nok[None, :]
        tl.store(HI + m[:, None] * V + n[None, :], hi, mask=ok)
        lof = tl.where(ok, lo, float("-inf"))
        tl.atomic_max(THR + m, tl.max(lof, axis=1), mask=mok)

    @triton.jit
    def _dq_kernel(X, sx, Q, E, OUT, so, M, V, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):
        """Drafter base logits: OUT[m, v] = bf16(x . D^T) (the screen's dot, no bound)."""
        pid = tl.program_id(0)
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < M
        nok = n < V
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            wd = _dq_tile(Q, E, n64, nok, k0, K, BK, BN)
            acc = tl.dot(x, wd, acc)
        tl.store(OUT + m[:, None] * so + n[None, :], acc.to(tl.bfloat16), mask=mok[:, None] & nok[None, :])

    @triton.jit
    def _cand_kernel(HI, THR, BAD, OUT, so, IDX, CNT, STATS, NLIVE, M, V, VREAL, inv_cap, cap, slack,
                     BMX: tl.constexpr, CB: tl.constexpr, DIVMODE: tl.constexpr):
        """Row j is listed iff j < VREAL and (any bad row, or any live row m with chain(hi[m, j]) + slack >=
        chain(T[m]) (slack = 2 x the chain's non-monotonic dip, see chain_check), or T[m] == -inf). Every non-listed (m, j) of OUT is written NaN."""
        pid = tl.program_id(0)
        nl = tl.load(NLIVE)
        nl = tl.where((nl >= 1) & (nl <= M), nl, M)
        j = pid * CB + tl.arange(0, CB)
        ok = j < V
        mm = tl.arange(0, BMX)
        mfull = mm < M
        mok = mm < nl
        bad = tl.max(tl.load(BAD + mm, mask=mok, other=0), axis=0)
        h = tl.load(HI + mm[:, None] * V + j[None, :], mask=mok[:, None] & ok[None, :],
                    other=float("-inf"))
        T = tl.load(THR + mm, mask=mok, other=float("inf"))
        ch = _chain(h, inv_cap, cap, DIVMODE)
        ct = _chain(T, inv_cap, cap, DIVMODE)
        hit = (ch + slack >= ct[:, None]) | (T == float("-inf"))[:, None]
        hrow = tl.max((hit & mok[:, None]).to(tl.int32), axis=0)
        sel = ((hrow > 0) | (bad > 0)) & (j < VREAL) & ok
        ni = sel.to(tl.int32)
        cnt = tl.sum(ni, axis=0)
        base = tl.atomic_add(CNT, cnt)
        pos = base + tl.cumsum(ni, axis=0) - ni
        tl.store(IDX + pos, j, mask=sel)
        nan = (tl.full((BMX, CB), 0x7FC00000, tl.uint32)).to(tl.float32, bitcast=True)
        # Write NaN to all non-listed entries of live rows AND all entries of dead rows (m >= nl).
        write_nan = mfull[:, None] & ok[None, :] & ((ni == 0)[None, :] | (~mok[:, None]))
        tl.store(OUT + mm[:, None] * so + j[None, :], nan, mask=write_nan)
        tl.atomic_add(STATS, cnt.to(tl.int64))

    @triton.jit
    def _refine_kernel(X, sx, W, sw, OUT, so, IDX, CNT, HIST, NLIVE, M, NPROG,
                       K: tl.constexpr, BV: tl.constexpr, BMX: tl.constexpr, BK: tl.constexpr):
        """Listed row j: same tl.dot tile [BV, BMX], ascending K, fp32 accumulate, stored as fp32 like the
        stock torch.mm(out_dtype=fp32) row (bit-equal for the qualified M); stored in place. Rows per chunk spread over the grid."""
        pid = tl.program_id(0)
        n = tl.load(CNT)
        if pid == 0:
            b = ((n > 0).to(tl.int32) + (n > 1).to(tl.int32) + (n > 2).to(tl.int32) + (n > 4).to(tl.int32)
                 + (n > 8).to(tl.int32) + (n > 16).to(tl.int32) + (n > 64).to(tl.int32) + (n > 256).to(tl.int32))
            tl.atomic_add(HIST + b, 1)
        nl = tl.load(NLIVE)
        nl = tl.where((nl >= 1) & (nl <= M), nl, M)
        rpc = tl.minimum(tl.maximum((n + NPROG - 1) // NPROG, 1), BV)
        nch = (n + rpc - 1) // rpc
        mm = tl.arange(0, BMX)
        mok = mm < nl
        rr = tl.arange(0, BV)
        for c in range(pid, nch, NPROG):
            r = c * rpc + rr
            rok = (rr < rpc) & (r < n)
            j = tl.load(IDX + r, mask=rok, other=0)
            j64 = j.to(tl.int64)
            acc = tl.zeros((BV, BMX), dtype=tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                w = tl.load(W + j64[:, None] * sw + k[None, :], mask=rok[:, None], other=0.0)
                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)
                acc = tl.dot(w, xt, acc)
            tl.store(OUT + mm[None, :] * so + j[:, None], acc.to(OUT.dtype.element_ty), mask=rok[:, None] & mok[None, :])

    _K.update(triton=triton, chain_sweep=_chain_sweep_kernel, xnorm=_xnorm_kernel, screen=_screen_kernel, dq=_dq_kernel,
              cand=_cand_kernel, refine=_refine_kernel)
    return _K


# =====================================================================================================
# Launch
# =====================================================================================================
_NSM: list = []


def _grid_n(dev) -> int:
    if XGRID:
        return XGRID
    if not _NSM:
        _NSM.append(int(torch.cuda.get_device_properties(dev).multi_processor_count))
    return 4 * _NSM[0]


def chain_check(cap: float, device, mode: int = 0, chunk: int = 1 << 24) -> tuple[bool, int, float]:
    """Exhaustive over EVERY finite fp32 (4.26e9 values, ascending order): (this module's chain == torch
    `torch.tanh(x / cap) * cap` bit for bit, number of differing values, dip) where dip = max over all a < b of
    chain(a) - chain(b) (0 for a monotone chain; NaN outputs make it inf). The libdevice/torch tanhf is not exactly
    monotone: a one-ulp dip exists at |x| = 18 for cap 30 (chain(17.999998) > chain(18.0)); the cut test in
    _cand_kernel adds 2 * dip of slack so that it stays sound."""
    k = _kernels()
    triton = k["triton"]
    lo_key, hi_key = 0x00800000, 0xFF7FFFFF  # key(-FLT_MAX) .. key(+FLT_MAX)
    x = torch.empty((chunk,), dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    ndiff_t = torch.zeros((), dtype=torch.int64, device=device)
    nan_t = torch.zeros((), dtype=torch.int64, device=device)
    dip_t = torch.zeros((), dtype=torch.float32, device=device)
    carry = None
    for base in range(lo_key, hi_key + 1, chunk):
        n = min(chunk, hi_key + 1 - base)
        k["chain_sweep"][(triton.cdiv(n, 1024),)](x, y, base, n, inv_cap_of(cap), _f32(cap), DIVMODE=mode, BLOCK=1024)
        xs, ys = x[:n], y[:n]
        ref = torch.tanh(xs / cap) * cap
        ndiff_t += (ref.view(torch.int32) != ys.view(torch.int32)).sum()
        nan_t += torch.isnan(ys).sum()
        cm = torch.cummax(ys, 0).values
        if carry is not None:
            cm = torch.maximum(cm, carry)
        dip_t = torch.maximum(dip_t, (cm - ys).max())
        carry = cm[n - 1].clone()
    ndiff, dip = int(ndiff_t.item()), float(dip_t.item())
    if int(nan_t.item()):
        dip = float("inf")
    return ndiff == 0, ndiff, dip


def slack_of(dip: float) -> float:
    """Slack added to chain(hi) in the cut test: 2 * dip (see chain_check) + 4 fp32 ulps of the cap range for the add."""
    return 0.0 if dip <= 0.0 else float(np.float32(2.0 * dip * (1.0 + 2.0 ** -10) + 2.0 ** -17))


_CHAIN_CACHE = os.environ.get("INKLING_CHAIN_CACHE", "").strip() or os.path.join(
    os.environ.get("INKLING_CERT_HEAD_CACHE", "") or os.path.join(os.path.expanduser("~"), ".cache", "certhead"),
    "chain")


def _chain_check_cached(cap: float, device, mode: int) -> tuple[bool, int, float]:
    """``chain_check`` memoised on disk.

    The check is EXHAUSTIVE over every finite fp32 (4.26e9 values) -- ~11 s per mode,
    ~11 s at start-up because mode 0 passes first. Its result is a pure function of
    (cap, mode, chain kernel) : the chain and torch's tanhf are fixed code, so the
    verdict cannot differ on the same build/GPU arch. Cache (same, ndiff, dip) keyed
    by cap + mode + the compute capability, and fall back to a full check on any miss
    or doubt (corrupt file, arch change, INKLING_CHAIN_CACHE=0).
    """
    if not _CHAIN_CACHE or os.environ.get("INKLING_CHAIN_CACHE", "") == "0":
        return chain_check(cap, device, mode)
    try:
        cc = torch.cuda.get_device_capability(device)
        key = f"chain-{cap!r}-m{mode}-sm{cc[0]}{cc[1]}.json"
        path = os.path.join(_CHAIN_CACHE, key)
        import json
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            if d.get("cap") == cap and int(d.get("mode", -1)) == mode and list(d.get("cc", ())) == list(cc):
                logger.info("INKLING chain_check %s: reused cached exhaustive verdict "
                            "(same=%s ndiff=%d dip=%.3g)", key, d["same"], d["ndiff"], d["dip"])
                return bool(d["same"]), int(d["ndiff"]), float(d["dip"])
    except Exception:  # noqa: BLE001
        pass
    same, nd, dip = chain_check(cap, device, mode)
    try:
        import json
        os.makedirs(_CHAIN_CACHE, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cap": cap, "mode": mode, "cc": list(torch.cuda.get_device_capability(device)),
                       "same": bool(same), "ndiff": int(nd), "dip": float(dip)}, f)
        os.replace(tmp, path)
        logger.info("INKLING chain_check: wrote exhaustive verdict to %s", path)
    except Exception:  # noqa: BLE001
        pass
    return same, nd, dip


def pick_chain_mode(cap: float, device) -> tuple[int, int, float]:
    """First DIVMODE whose chain is bit-equal to torch and (nearly) monotone: (mode, ndiff of the last tried, dip);
    mode -1 when none passes (the certificate is then not usable)."""
    last = (0, float("inf"))
    for mode in (0, 1):
        same, nd, dip = _chain_check_cached(cap, device, mode)
        if same and dip <= 1e-3 * cap:
            return mode, 0, dip
        last = (nd, dip)
    return -1, last[0], last[1]


def _bufs(t: CertTable, M: int, dev):
    NG = t.K // t.group
    return dict(
        xg=torch.empty((M, NG), dtype=torch.float32, device=dev),
        xn=torch.empty((M,), dtype=torch.float32, device=dev),
        bad=torch.empty((M,), dtype=torch.int32, device=dev),
        thr=torch.empty((M,), dtype=torch.float32, device=dev),
        hi=torch.empty((M, t.V), dtype=torch.float32, device=dev),
        idx=torch.empty((t.V,), dtype=torch.int32, device=dev),
        cnt=torch.empty((1,), dtype=torch.int32, device=dev),
    )


class CertState:
    """Per-rank, per-process state of the certified head: counters live on device."""

    def __init__(self, t: CertTable, W: torch.Tensor, vreal: int, cap: float, divmode: int = 0, dip: float = 0.0):
        dev = W.device
        self.t, self.W, self.vreal, self.cap = t, W, int(vreal), float(cap)
        self.inv_cap, self.cap32 = inv_cap_of(cap), _f32(cap)
        self.divmode = int(divmode)                                       # chain variant picked by pick_chain_mode
        self.dip, self.slack = float(dip), slack_of(dip)                  # chain non-monotonicity and cut slack
        self.stats = torch.zeros((1,), dtype=torch.int64, device=dev)   # [listed rows (sum over calls)]
        self.hist = torch.zeros((9,), dtype=torch.int32, device=dev)    # list length bins per call
        self.m_ok: dict[int, bool] = {}
        self.calls = 0
        self.disabled = False        # set by tick() when the list-length cap trips: stock head from then on
        self.last_listed = 0
        self.last_calls = 0
        self.debug: dict | None = None
        self.check: torch.Tensor | None = None


_NLIVE_FULL: dict = {}


def _nlive_for(M: int, dev, num_tokens: int = 0) -> torch.Tensor:
    """Device int32 [1] live-row count: ROWAR buf when M matches the armed target forward, else M."""
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as ra
        # Safe under BOTH FULL CUDA graph replay and eager: whenever the pure-decode forward that
        # just ran had num_tokens == M == _ROW_FULL (and rowar_begin has armed at least once),
        # _n_live_kernel ran on this exact step and updated ROWAR["buf"] even if an earlier prefill
        # cleared ROWAR["last_live"].
        if (
            ra._ROW_AR
            and ra.ROWAR.get("begins", 0) > 0
            and num_tokens == M == ra._ROW_FULL
            and ra.ROWAR.get("buf") is not None
        ):
            return ra.ROWAR["buf"]
        if num_tokens == 0:
            ll = ra.ROWAR.get("last_live")
            if ll is not None and ll[0] == M and ll[1] is not None:
                return ll[1]
    except Exception:
        pass
    key = (int(dev.index or 0), M)
    t = _NLIVE_FULL.get(key)
    if t is None:
        t = torch.tensor([M], dtype=torch.int32, device=dev)
        _NLIVE_FULL[key] = t
    return t


def certified_local(
    cs: CertState, x: torch.Tensor, debug: dict | None = None, num_tokens: int = 0
) -> torch.Tensor:
    """x [M, K] bf16 -> fp32 [M, V]: the exact stock fp32 head logits on every listed row, NaN elsewhere (the local
    argmax reads NaN as -inf). No host sync; M <= 128."""
    k = _kernels()
    triton = k["triton"]
    t = cs.t
    x = x.contiguous()
    M, K = x.shape
    V, dev = t.V, x.device
    NG = K // t.group
    b = _bufs(t, M, dev)
    nlive = _nlive_for(M, dev, num_tokens=num_tokens)
    out = torch.empty((M, V), dtype=torch.float32, device=dev)
    BM = max(16, next_pow2(M))
    k["xnorm"][(M,)](x, x.stride(0), b["xg"], b["xn"], b["bad"], b["thr"], b["cnt"], nlive, M, K=K, G=t.group,
                     NGP=next_pow2(NG), INFL=INFL_X, XABS=X_ABS)
    k["screen"][(triton.cdiv(V, SBN),)](
        x, x.stride(0), t.q, t.e, t.rt, t.nv, b["xg"], b["xn"], b["hi"], b["thr"], nlive, M, V, cs.vreal,
        K=K, G=t.group, BM=BM, BN=SBN, BK=SBK, CACC2=2.0 * c_acc(K), INFL_Q=INFL_Q, INFL_B=INFL_B,
        SLOP_REL=SLOP_REL, SLOP_ABS=SLOP_ABS, num_warps=SWARPS if BM <= 32 else max(SWARPS, 8),
        num_stages=SSTAGES)
    cb = max(16, 4096 // BM)
    k["cand"][(triton.cdiv(V, cb),)](
        b["hi"], b["thr"], b["bad"], out, out.stride(0), b["idx"], b["cnt"], cs.stats, nlive, M, V, cs.vreal,
        cs.inv_cap, cs.cap32, cs.slack, BMX=BM, CB=cb, DIVMODE=cs.divmode, num_warps=4)
    nprog = _grid_n(dev)
    k["refine"][(nprog,)](x, x.stride(0), cs.W, cs.W.stride(0), out, out.stride(0), b["idx"], b["cnt"], cs.hist,
                          nlive, M, nprog, K=K, BV=BV, BMX=BM, BK=XBK, num_warps=XWARPS, num_stages=XSTAGES)
    if debug is not None:
        debug.update(b)
    return out


def exact_all(cs: CertState, x: torch.Tensor) -> torch.Tensor:
    """The refine kernel on EVERY row [0, V): the exact-head reference the qualification compares to the stock head."""
    k = _kernels()
    triton = k["triton"]
    x = x.contiguous()
    M, K = x.shape
    V, dev = cs.t.V, x.device
    BM = max(16, next_pow2(M))
    out = torch.empty((M, V), dtype=torch.float32, device=dev)
    idx = torch.arange(V, dtype=torch.int32, device=dev)
    cnt = torch.full((1,), V, dtype=torch.int32, device=dev)
    hist = torch.zeros((9,), dtype=torch.int32, device=dev)
    nlive = torch.tensor([M], dtype=torch.int32, device=dev)
    nprog = _grid_n(dev)
    k["refine"][(nprog,)](x, x.stride(0), cs.W, cs.W.stride(0), out, out.stride(0), idx, cnt, hist, nlive, M, nprog,
                          K=K, BV=BV, BMX=BM, BK=XBK, num_warps=XWARPS, num_stages=XSTAGES)
    return out


def dq_logits(t: CertTable, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Drafter base logits: bf16(x . D^T) [M, V] from the shared int8 table (graph-capturable, no sync)."""
    k = _kernels()
    triton = k["triton"]
    M, K = x.shape
    BM = max(16, next_pow2(M))
    if out is None:
        out = torch.empty((M, t.V), dtype=torch.bfloat16, device=x.device)
    k["dq"][(triton.cdiv(t.V, SBN),)](x, x.stride(0), t.q, t.e, out, out.stride(0), M, t.V, K=K, BM=BM, BN=SBN,
                                       BK=SBK, num_warps=SWARPS if BM <= 32 else max(SWARPS, 8), num_stages=SSTAGES)
    return out


def dq_supported(x: torch.Tensor, t: CertTable) -> bool:
    return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1 and 0 < x.shape[0] <= 64
            and x.shape[1] == t.K and x.is_cuda)


def tick(cs: CertState, rank: int = 0) -> None:
    """Call once per certified call. Every PERIOD calls: ONE host read of the listed-row counters; log them (when
    LOG_EVERY) and trip the list-length cap (mean listed rows per call over the window > LISTCAP -> this rank
    serves the stock head from now on; the exchanged records are identical either way)."""
    cs.calls += 1
    if cs.calls % PERIOD:
        return
    listed = int(cs.stats[0].item())
    win_calls = cs.calls - cs.last_calls
    mean = (listed - cs.last_listed) / max(1, win_calls)
    cs.last_listed, cs.last_calls = listed, cs.calls
    if LOG_EVERY > 0:
        log_stats(cs, rank)
    if LISTCAP and mean > LISTCAP and not cs.disabled:
        cs.disabled = True
        logger.warning("INKLING_CERT_HEAD rank %d: mean listed rows %.0f/call over the last %d calls exceeds "
                       "LISTCAP %d: stock head from now on", rank, mean, win_calls, LISTCAP)


def log_stats(cs: CertState, rank: int = 0) -> None:
    c = [int(v) for v in cs.stats.tolist()]
    h = [int(v) for v in cs.hist.tolist()]
    calls = max(1, sum(h))
    logger.info("INKLING_CERT_HEAD rank %d: calls %d, listed rows %d (%.2f/call); list length histogram "
                "[0,1,2,3-4,5-8,9-16,17-64,65-256,>256] = %s", rank, sum(h), c[0], c[0] / calls, h)
