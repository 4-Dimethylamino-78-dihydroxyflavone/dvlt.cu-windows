#!/usr/bin/env python3
"""
dvlt_long.py — chunked DVLT reconstruction with confidence-weighted IRLS Sim(3)
seam alignment, loop closure (backbone descriptors + EXIF GPS), global Sim(3)
pose graph, and streaming merge.  Clean-room implementation of the VGGT-Long
recipe (arXiv 2507.16443) adapted to dvlt.exe outputs.

v1.5: skeleton-hub topology (--topology hub), unordered
ingestion (--order unordered), descriptor dedup pre-pass, live/final profiles,
content-addressed chunk cache, manifest.json.
numpy + scipy + PIL only.
"""
import argparse, hashlib, json, math, os, re, shutil, struct, subprocess, sys, time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

IMG_EXTS = {'.jpg', '.jpeg', '.png'}
ANSI = 'mbcs' if os.name == 'nt' else 'utf-8'    # dvlt.exe opens paths through the narrow CRT
# O(N^2) global attention is the VRAM wall: ~80 frames @ 504 px on 16 GB [MEASURED, README].
# The skeleton is clamped to this; --skeleton-size 48 leaves headroom inside one run (§4.2).
VRAM_FRAME_CAP = 80

# ─── Sim(3) algebra ──────────────────────────────────────────────────────────
def _hat(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])

def _sim3_W(om, sig):
    """Left-Jacobian-like W s.t. t = W @ upsilon (Strasdat scale-drift SLAM / Sophus)."""
    th = float(np.linalg.norm(om))
    Om = _hat(om); Om2 = Om @ Om
    s = math.exp(sig)
    eps = 1e-9
    if abs(sig) < eps:
        C = 1.0
        if th < eps:
            A, B = 0.5, 1.0 / 6.0
        else:
            A = (1.0 - math.cos(th)) / th**2
            B = (th - math.sin(th)) / th**3
    else:
        C = (s - 1.0) / sig
        if th < eps:
            A = ((sig - 1.0) * s + 1.0) / sig**2
            B = ((0.5 * sig**2 - sig + 1.0) * s - 1.0) / sig**3
        else:
            a_ = s * math.sin(th); b_ = s * math.cos(th); c_ = th**2 + sig**2
            A = (a_ * sig + (1.0 - b_) * th) / (th * c_)
            B = (C - ((b_ - 1.0) * sig + a_ * th) / c_) / th**2
    return C * np.eye(3) + A * Om + B * Om2

def sim3_exp(v):
    ups, om, sig = np.asarray(v[:3], float), np.asarray(v[3:6], float), float(v[6])
    R = Rotation.from_rotvec(om).as_matrix()
    return math.exp(sig), R, _sim3_W(om, sig) @ ups

def sim3_log(a):
    s, R, t = a
    om = Rotation.from_matrix(R).as_rotvec()
    sig = math.log(s)
    ups = np.linalg.solve(_sim3_W(om, sig), t)
    return np.concatenate([ups, om, [sig]])

def sim3_compose(a, b):
    sa, Ra, ta = a; sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta

def sim3_inv(a):
    s, R, t = a
    return 1.0 / s, R.T, -(R.T @ t) / s

def sim3_apply(a, pts):
    s, R, t = a
    return s * (pts @ R.T) + t

# ─── Weighted Umeyama + IRLS (spec §3.2 stage 3) ─────────────────────────────
def weighted_umeyama(src, tgt, w):
    wn = w / (w.sum() + 1e-12)
    mu_s = wn @ src; mu_t = wn @ tgt
    sc = src - mu_s; tc = tgt - mu_t
    spread_s = math.sqrt(float(wn @ (sc**2).sum(axis=1)))
    spread_t = math.sqrt(float(wn @ (tc**2).sum(axis=1)))
    s = spread_t / max(spread_s, 1e-12)
    sw = np.sqrt(wn)[:, None]
    H = (s * sc * sw).T @ (tc * sw)
    U, _, Vt = np.linalg.svd(H)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Vt[2, :] *= -1
    R = Vt.T @ U.T
    t = mu_t - s * (R @ mu_s)
    return s, R, t

def _weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])

def irls_sim3(src, tgt, w0, delta=0.1, max_iters=5, tol=1e-9):
    w0 = w0 / (w0.sum() + 1e-12)
    if len(src) < 3:                       # empty/degenerate correspondence set: honor the
        return (1.0, np.eye(3), np.zeros(3)), {'converged': False, 'iters': 0,
                                               'n': len(src), 'med_resid': float('inf')}
    est = weighted_umeyama(src, tgt, w0)
    prev_err = None
    info = {'converged': False, 'iters': 0, 'n': len(src)}
    for it in range(max_iters):
        r = np.linalg.norm(tgt - sim3_apply(est, src), axis=1)
        hub = np.ones_like(r)
        big = r > delta
        hub[big] = delta / r[big]
        w = w0 * hub
        new = weighted_umeyama(src, tgt, w)
        # convergence: parameter change + rotation step + relative error change
        dparam = abs(new[0] - est[0]) + float(np.linalg.norm(new[2] - est[2]))
        dang = math.degrees(math.acos(np.clip((np.trace(new[1] @ est[1].T) - 1) / 2, -1, 1)))
        err = float(np.sum(np.where(big, delta * (r - 0.5 * delta), 0.5 * r**2) * w0))
        est = new
        info['iters'] = it + 1
        if (dparam < tol and dang < 0.1) or (prev_err is not None and abs(prev_err - err) < tol * max(prev_err, 1e-30)):
            info['converged'] = True
            break
        prev_err = err
    r = np.linalg.norm(tgt - sim3_apply(est, src), axis=1)
    info['med_resid'] = _weighted_median(r, w0)
    # Exhausting the iteration budget is normal operation (VGGT-Long runs 5 iters as the
    # standard budget). 'converged' means "estimate usable": True whenever the residuals are
    # finite; only the degenerate-input guard (n < 3, above) reports False.
    if np.isfinite(info['med_resid']):
        info['converged'] = True
    return est, info

def _ident(p):
    """Frame identity = resolved original source path. Chunk.files holds the originals
    and every dvlt run is fed exactly those paths (addendum §4.3)."""
    return str(Path(p).resolve())

def shared_frame_indices(chunk_a, chunk_b):
    """Local-index pairs of frames both chunks contain, matched by original-path
    identity. Serial overlap, injected bridges, and loop-chunk halves are all the
    same mechanism (addendum §4.3 — breaking change vs v1's g_start arithmetic)."""
    ia = {}
    for li, f in enumerate(chunk_a.files):
        ia.setdefault(_ident(f), li)
    out, seen = [], set()
    for lj, f in enumerate(chunk_b.files):
        k = _ident(f)
        li = ia.get(k)
        if li is not None and k not in seen:
            seen.add(k)
            out.append((li, lj))
    return out

def _seam_correspondences(chunk_a, chunk_b, max_corr):
    """Per-pixel correspondences on the frames both chunks contain (pooled over frames)."""
    blocks = _frame_blocks(chunk_a, chunk_b, max_corr)
    if not blocks:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    A = np.concatenate([b[0] for b in blocks])
    B = np.concatenate([b[1] for b in blocks])
    w = np.concatenate([b[2] for b in blocks])
    if len(A) > max_corr:
        idx = np.linspace(0, len(A) - 1, max_corr).astype(int)
        A, B, w = A[idx], B[idx], w[idx]
    return A, B, w

def _frame_blocks(chunk_a, chunk_b, max_corr=100_000, per_frame_cap=20_000):
    """Correspondences kept SEPARATED per shared frame.

    Keeping the frame boundaries is what makes frame-level robustness possible; pooling
    them first is precisely the bug fixed in align_shared (see there)."""
    pairs = shared_frame_indices(chunk_a, chunk_b)
    if not pairs:
        return []
    pa, _, ca = chunk_a.pts_grid()
    pb, _, cb = chunk_b.pts_grid()
    cap = max(1000, min(per_frame_cap, max_corr // max(1, len(pairs))))
    out = []
    for li, lj in pairs:
        p = pa[li].reshape(-1, 3); q = pb[lj].reshape(-1, 3)
        wa = ca[li].reshape(-1);   wb = cb[lj].reshape(-1)
        ok = np.isfinite(p).all(1) & np.isfinite(q).all(1)
        if ok.sum() < 50:
            continue
        th = 0.1 * min(np.median(wa[ok]), np.median(wb[ok]))
        ok &= (wa > th) & (wb > th)
        if ok.sum() < 50:
            continue
        p, q, w = p[ok], q[ok], np.sqrt(wa[ok] * wb[ok])
        if len(p) > cap:
            s = np.linspace(0, len(p) - 1, cap).astype(int)
            p, q, w = p[s], q[s], w[s]
        out.append((p, q, w, li, lj))
    return out

def _sim3_discrepancy(a, b):
    """Unit-free disagreement between two Sim(3)s: |log scale ratio| + rotation (rad).
    Translation is deliberately excluded - it is absorbed by scale/rotation about the
    centroid and would otherwise scale with scene size, making one tolerance impossible."""
    d = sim3_compose(sim3_inv(a), b)
    return abs(math.log(max(d[0], 1e-12))) + math.radians(_rot_deg(d[1]))

def frame_consensus_sim3(blocks, tol=0.05, delta=0.1, max_iters=5, tol_conv=1e-9):
    """Two-level robust alignment: consensus ACROSS frames, then IRLS across pixels.

    [MEASURED on a 723-frame indoor set] Pixel-pooled IRLS is robust across pixels but blind
    across frames: a shared frame whose geometry is systematically displaced contributes
    many mutually-consistent pixels, so a pixel-level Huber loss reads it as inlier. Worse,
    Umeyama's scale is a ratio of RMS spreads, so pooling frames that disagree about
    relative PLACEMENT inflates the target spread - the joint fit can land outside the range
    of every individual frame's solution (measured: per-frame s in {0.604..0.775}, pooled fit
    0.853). Fixing this cut the median per-frame residual on that set's worst edges by ~10x and
    left already-good edges untouched.

    This is TEASER's consensus idea (arXiv 2001.07715) applied at the granularity where our
    outliers actually live. We do NOT need its SDP/max-clique machinery: with 2-30 frames an
    exact O(n^2) vote is trivial, and our correspondences are pixel-indexed, so classical
    matching outliers - the thing TEASER exists to survive - cannot occur here at all.

    Returns (est, info) with info['consensus_frac'] / ['n_frames'] / ['n_consensus'].
    """
    n = len(blocks)
    if n == 0:
        return (1.0, np.eye(3), np.zeros(3)), {'converged': False, 'iters': 0, 'n': 0,
                                               'med_resid': float('inf'), 'n_frames': 0,
                                               'n_consensus': 0, 'consensus_frac': 0.0}
    per_frame = [irls_sim3(q, p, w, delta, max_iters, tol_conv)[0] for p, q, w, _, _ in blocks]
    if n == 1:
        keep, rival = [0], 0
    else:                                   # exact max-consensus vote over frames
        votes = [[j for j in range(n) if _sim3_discrepancy(per_frame[i], per_frame[j]) < tol]
                 for i in range(n)]
        keep = max(votes, key=len)
        # Size of the largest cluster DISJOINT from the winner. A substantial rival is the
        # signature of genuinely split geometry: two self-consistent populations, neither of
        # which is noise. Averaging them yields a physically meaningless compromise, so the
        # caller must be able to see it (edge state AMBIGUOUS/NONRIGID) rather than get a
        # confident-looking number. [MEASURED: hub 0<-13, s in {0.60,0.62,0.61} vs
        # {0.78,0.76}; the pooled fit returned 0.853 - outside BOTH populations.]
        ks = set(keep)
        rival = max((len([j for j in v if j not in ks]) for v in votes), default=0)
    P = np.concatenate([blocks[i][0] for i in keep])
    Q = np.concatenate([blocks[i][1] for i in keep])
    W = np.concatenate([blocks[i][2] for i in keep])
    est, info = irls_sim3(Q, P, W, delta, max_iters, tol_conv)
    scales = np.array([t[0] for t in per_frame])
    info['n_frames'] = n
    info['n_consensus'] = len(keep)
    info['consensus_frac'] = len(keep) / n
    info['consensus_frames'] = [blocks[i][3] for i in keep]
    info['rival_frames'] = int(rival)
    info['scale_spread'] = float(scales.max() / max(scales.min(), 1e-12))
    info['per_frame'] = [{'frame': blocks[i][3], 'peer': blocks[i][4],
                          's': round(float(per_frame[i][0]), 5),
                          'n': int(len(blocks[i][0])),
                          'in_consensus': i in set(keep)} for i in range(n)]
    # cache a correspondence subsample for POST-graph ghost validation (edge_ghosts):
    # the only stage that can see whether the optimizer actually realized this edge.
    gs = np.linspace(0, len(P) - 1, min(2048, len(P))).astype(int)
    info['ghost_sample'] = (P[gs].astype(np.float32), Q[gs].astype(np.float32))
    info['state'] = _edge_state(info)
    return est, info

# Edge states (2026-07-19). A binary pass/fail cannot express "two coherent
# populations", which is precisely the state that silently corrupts a merge: it looks like a
# converged fit and is not one. Only GOOD (and WEAK, deliberately) may enter the graph.
EDGE_GOOD, EDGE_WEAK = 'GOOD', 'WEAK'
EDGE_AMBIGUOUS, EDGE_NONRIGID, EDGE_ORPHAN = 'AMBIGUOUS', 'NONRIGID', 'ORPHAN'

def _edge_state(info, weak_frac=0.6, rival_frac=0.2):
    n, nc = info.get('n_frames', 0), info.get('n_consensus', 0)
    if n == 0 or not np.isfinite(info.get('med_resid', float('inf'))):
        return EDGE_ORPHAN
    if n == 1:
        return EDGE_WEAK                 # single shared frame: nothing can corroborate it
    if nc < 2:
        return EDGE_NONRIGID             # no two frames agree at all
    # AMBIGUOUS requires the rival population to have SUBSTANTIAL support (>=2 frames AND
    # >=20% of the shared set). A 2-frame blip against a 27-frame consensus is outvoted,
    # not a second hypothesis - and it cannot pollute the fit (consensus-only refit) nor
    # hide downstream (post-graph ghost validation measures its region anyway).
    if info.get('rival_frames', 0) >= max(2, math.ceil(rival_frac * n)):
        return EDGE_AMBIGUOUS            # a competing self-consistent population exists
    return EDGE_GOOD if info['consensus_frac'] >= weak_frac else EDGE_WEAK

def align_shared(chunk_a, chunk_b, max_corr=100_000, consensus_tol=0.05):
    """Sim(3) mapping chunk_b coords -> chunk_a coords over identity-shared frames.
    One alignment path for rim (adjacent), hub (bridge), and loop halves.

    v1.5.1: the estimator is now two-level (frame consensus, then pixel IRLS) — see
    frame_consensus_sim3. The correspondence construction, conf gating, sqrt(c_A*c_B)
    weights, Huber delta and iteration budget are all v1 verbatim; only the level at
    which outliers are rejected changed."""
    blocks = _frame_blocks(chunk_a, chunk_b, max_corr)
    if blocks and sum(len(b[0]) for b in blocks) >= 100:
        est, info = frame_consensus_sim3(blocks, tol=consensus_tol)
        if info['converged']:
            info['fallback'] = False
            return est, info
    # fallback: camera centers of shared frames (old dvlt_chunked method)
    pairs = shared_frame_indices(chunk_a, chunk_b)
    if len(pairs) < 3:
        return (1.0, np.eye(3), np.zeros(3)), {'converged': False, 'fallback': True,
                                               'iters': 0, 'med_resid': float('inf'), 'n': 0}
    pa, pb = chunk_a.poses(), chunk_b.poses()
    src = np.array([pb[lj]['c2w'][:3, 3] for _, lj in pairs])
    tgt = np.array([pa[li]['c2w'][:3, 3] for li, _ in pairs])
    est = weighted_umeyama(src, tgt, np.ones(len(src)))
    r = np.linalg.norm(tgt - sim3_apply(est, src), axis=1)
    return est, {'converged': True, 'fallback': True, 'iters': 0,
                 'med_resid': float(np.median(r)), 'n': len(src),
                 # camera centers double as the ghost sample so even fallback edges are
                 # subject to post-graph validation rather than exempt from it
                 'ghost_sample': (tgt.astype(np.float32), src.astype(np.float32))}

align_adjacent = align_shared   # rim seams are the serial special case

# ─── I/O ─────────────────────────────────────────────────────────────────────
def read_ply_grid(path):
    with open(path, 'rb') as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:                      # EOF: a truncated header would loop forever
                raise ValueError(f"{path}: truncated PLY (no end_header)")
            header += line
        n = int(re.search(rb"element vertex (\d+)", header).group(1))
        rec = np.frombuffer(f.read(n * 15), dtype=np.uint8).reshape(n, 15)
    xyz = rec[:, :12].copy().view(np.float32).reshape(n, 3)
    rgb = rec[:, 12:].copy()
    return xyz, rgb

def _read_bin(path, magic, ndims):
    with open(path, 'rb') as f:
        if f.read(4) != magic:
            raise ValueError(f"{path}: bad magic (rebuild dvlt.exe with -x support?)")
        dims = struct.unpack(f"<{ndims}i", f.read(4 * ndims))
        data = np.fromfile(f, dtype=np.float32, count=int(np.prod(dims)))
    return data.reshape(dims)

def read_conf(path): return _read_bin(path, b"DVLC", 3)
def read_desc(path): return _read_bin(path, b"DVLD", 2)

def read_poses(path):
    with open(path) as f:
        d = json.load(f)
    return [{'c2w': np.array(fr['extrinsics_c2w'], float).reshape(4, 4),
             'K':   np.array(fr['intrinsics'], float).reshape(3, 3)} for fr in d['frames']]

_GRIDS = {}     # 2-slot LRU: seams/loops touch chunks pairwise in order, so two
                # slots turn ~60 full-grid loads per run (~11 GB I/O) into ~30

class Chunk:
    def __init__(self, idx, g_start, files, out_dir, role='local'):
        self.idx, self.g_start, self.files, self.out_dir = idx, g_start, files, Path(out_dir)
        self.role = role                     # 'local' | 'skeleton' | 'loop'
        self.bridge_mask = [False] * len(files)  # True = injected skeleton-frame slot
        self.g_rows = None                   # global row (index into the run's image list) per file
        self.k = None                        # loop steps this chunk runs at
        self.key = None                      # content-address suffix (addendum §4.7)
    def pts_grid(self):
        key = str(self.out_dir)
        if key in _GRIDS:
            _GRIDS[key] = _GRIDS.pop(key)                  # refresh LRU order
            return _GRIDS[key]
        xyz, rgb = read_ply_grid(self.out_dir / 'scene.ply')
        conf = read_conf(self.out_dir / 'conf.bin')
        S, H, W = conf.shape
        assert len(xyz) == S * H * W, f"{self.out_dir}: PLY not full-grid (must be produced with -c 0)"
        while len(_GRIDS) >= 2:
            _GRIDS.pop(next(iter(_GRIDS)))
        _GRIDS[key] = (xyz.reshape(S, H, W, 3), rgb.reshape(S, H, W, 3), conf)
        return _GRIDS[key]
    def poses(self): return read_poses(self.out_dir / 'poses.json')
    def desc(self): return read_desc(self.out_dir / 'desc.bin')

def make_chunks(images, L, O):
    if L <= 0 or O < 0 or O >= L:
        sys.exit(f"invalid chunking: need 0 <= overlap < chunk_size, got L={L} O={O}")
    step = L - O
    chunks, start, idx = [], 0, 0
    while start < len(images):
        end = min(start + L, len(images))
        chunks.append(Chunk(idx, start, images[start:end], ""))
        if end >= len(images):
            break
        start += step; idx += 1
    return chunks

def _list_is_abs(s):
    """Mirror dvlt.cu expand_list's absolute test (dvlt.cu:71): leading '/' or
    '\\' (rooted) OR a drive colon at position 1. Path.is_absolute() is False for
    rooted-but-driveless Windows paths, so joining them would graft the wrong drive."""
    return s[0] in '/\\' or (len(s) > 1 and s[1] == ':')

def list_images(src):
    """Image paths in dvlt.exe frame order: a directory is sorted bytewise by name
    (mirrors dvlt.cu expand_dir std::sort — case-SENSITIVE, unlike Python's default
    WindowsPath ordering), a .txt list keeps the order written (mirrors dvlt.cu
    expand_list: UTF-8 BOM tolerated, '#' comments and blank lines skipped after
    trimming, relative paths resolve against the list's directory). Paths may span
    any number of directories: frame identity is the resolved original path, so
    dedup/bridges/ownership are indifferent to where a frame lives."""
    src = Path(src)
    if src.is_file() and src.suffix.lower() == '.txt':
        try:
            raw_text = src.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            sys.exit(f"{src} is not UTF-8 (UTF-16?). Re-save as UTF-8 — "
                     f"PowerShell: `Out-File -Encoding utf8`.")
        imgs = []
        for ln, line in enumerate(raw_text.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            p = Path(s) if _list_is_abs(s) else src.parent / s
            if not p.exists():
                sys.exit(f"{src} line {ln}: file not found: {p}")
            if p.suffix.lower() not in IMG_EXTS:
                sys.exit(f"{src} line {ln}: unsupported extension "
                         f"(need {'/'.join(sorted(IMG_EXTS))}): {p}")
            imgs.append(p)
    elif src.is_dir():
        imgs = sorted((p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS),
                      key=lambda p: p.name)
    else:
        sys.exit(f"{src}: not an image directory or .txt list file")
    if not imgs:
        sys.exit(f"no images in {src}")
    imgs = [p.resolve() for p in imgs]      # dvlt.exe runs in its own folder: absolute only
    # ...and opens them in the ANSI code page: refuse what it cannot open now, naming the
    # file, before any GPU run. Round-trip per path - the code-page codec's
    # UnicodeEncodeError carries no usable offset.
    bad = [p for p in imgs if str(p) != str(p).encode(ANSI, 'replace').decode(ANSI)]
    if bad:
        sys.exit(f"{bad[0]}: outside the ANSI code page, which dvlt.exe opens paths through"
                 f" - rename it" + (f" ({len(bad) - 1} more like it)" if len(bad) > 1 else ''))
    return imgs

def _write_list(files, path):
    """Feed a run through a dvlt.exe .txt list (dvlt.cu expand_list): order kept as written
    (line 1 = reference camera), absolute paths, nothing copied, no command-line cap. ANSI
    code page; list_images already refused any path it cannot hold."""
    path.write_text(''.join(f"{_ident(f)}\n" for f in files), encoding=ANSI)
    return path

def run_dvlt(dvlt_exe, image_paths, out_dir, img_size, k):
    out_dir = Path(out_dir)
    need = ['scene.ply', 'poses.json', 'conf.bin', 'desc.bin']
    if all((out_dir / n).exists() for n in need):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(dvlt_exe), '-b', '-c', '0', '-k', str(k), '-s', str(img_size), '-x',
           '-w', str(Path(dvlt_exe).parent / 'model' / 'weights.dvlt'), '-o', str(out_dir),
           str(_write_list(image_paths, out_dir / 'frames.txt'))]
    r = subprocess.run(cmd, cwd=str(Path(dvlt_exe).parent))
    if r.returncode != 0:
        sys.exit(f"dvlt.exe failed (exit {r.returncode}) for {out_dir}")
    missing = [n for n in need if not (out_dir / n).exists()]
    if missing:                # exe exit 0 does not guarantee outputs (e.g. fopen failure)
        sys.exit(f"dvlt.exe reported success but {missing} missing in {out_dir}")

# ─── v1.5: content-addressed cache + encode-only (addendum §4.0, §4.7) ───────
def file_sha1(path, _cache={}):
    p = str(path)
    if p not in _cache:
        h = hashlib.sha1()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        _cache[p] = h.hexdigest()
    return _cache[p]

def chunk_key(files, img_size, k, exe_sha):
    """Content-address for a dvlt run: ordered original paths + img_size + K + exe SHA
    [addendum §4.7, D8]. A K=8 live chunk can never satisfy a K=16 final pass, and a
    bridge-modified frame list never collides with a plain window."""
    h = hashlib.sha1()
    for f in files:
        h.update(_ident(f).encode('utf-8', 'surrogatepass') + b'\n')
    h.update(f"{img_size}|{k}|{exe_sha}".encode())
    return h.hexdigest()[:12]

def run_dvlt_encode(dvlt_exe, image_paths, out_dir, img_size):
    """dvlt.exe -e: whole-corpus descriptors BEFORE any chunking decision exists
    (addendum §4.0). desc.bin rows follow the list file's order. Returns raw f32 [N,768]
    descriptor means."""
    out_dir = Path(out_dir)
    if not (out_dir / 'desc.bin').exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(dvlt_exe), '-b', '-e', '-s', str(img_size),
               '-w', str(Path(dvlt_exe).parent / 'model' / 'weights.dvlt'), '-o', str(out_dir),
               str(_write_list(image_paths, out_dir / 'frames.txt'))]
        r = subprocess.run(cmd, cwd=str(Path(dvlt_exe).parent))
        if r.returncode != 0 or not (out_dir / 'desc.bin').exists():
            sys.exit("dvlt.exe -e/--encode-only failed or unsupported: rebuild dvlt.exe via "
                     "build.bat with the 2026-07-18 encode-only patch (addendum §4.0)")
    desc = read_desc(out_dir / 'desc.bin')
    if len(desc) != len(image_paths):
        sys.exit(f"encode-only returned {len(desc)} rows for {len(image_paths)} frames "
                 f"({out_dir}) — stale cache? delete the dir and re-run")
    return desc

# ─── v1.5: dedup, FPS skeleton, bridges, unordered assignment (§4.1–4.5) ─────
def power_normalize(raw):
    """Signed β=0.5 power → L2. The dedup space [INVARIANT D5]: corpus-independent, so an
    absolute threshold like 0.95 is meaningful. Whitened cosines are corpus-relative
    (adjacent frames measure 0.18–0.39 there) and raw cosines saturate near 1 — neither
    space can serve the other's job. Do not unify."""
    g = np.sign(raw) * np.sqrt(np.abs(raw))
    return (g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def dedup_frames(raw, sim_thresh=0.95, min_keep=0.5, scan_order=None):
    """Greedy near-duplicate drop in power-normalized space (addendum §4.1).
    raw MUST be [N,768] un-reduced descriptor means; whitened/reduced vectors are
    rejected (two-space invariant D5). Returns (kept sorted, dropped {i: (dup_of, sim)},
    clamped)."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 768:
        raise ValueError(f"dedup_frames expects raw [N,768] descriptors, got {tuple(raw.shape)}; "
                         "whitened/reduced vectors must never reach dedup (spec D5)")
    P = power_normalize(raw)
    N = len(P)
    order = list(scan_order) if scan_order is not None else list(range(N))
    kept, dropped = [], {}
    for i in order:
        if kept:
            sims = P[kept] @ P[i]
            j = int(np.argmax(sims))
            if float(sims[j]) > sim_thresh:
                dropped[i] = (kept[j], float(sims[j]))
                continue
        kept.append(i)
    floor = int(math.ceil(min_keep * N))
    clamped = False
    if len(kept) < floor and dropped:
        clamped = True
        # min-keep guard: re-admit the least-duplicate drops until the floor holds
        for i, _ in sorted(dropped.items(), key=lambda kv: kv[1][1]):
            if len(kept) >= floor:
                break
            kept.append(i)
            dropped.pop(i)
    return sorted(kept), dropped, clamped

def skeleton_rows_clustered(Z, n_total, group=4):
    """Skeleton selection that separates COVERAGE from REGISTRABILITY.

    Pure FPS picks the frames that are maximally dissimilar to each other - which is what
    you want for scene coverage and the worst possible input to a multi-view geometry model.
    DVLT triangulates a frame against its co-visible neighbours; an FPS set gives every frame
    a context of maximally-unrelated views, so the skeleton's own internal geometry is
    unreliable exactly where the hub then anchors everything to it.

    [MEASURED on a 723-frame indoor set] FPS skeleton hub edges split into two self-consistent
    scale populations (0.60-0.62 vs 0.76-0.78) while rim seams between overlapping windows
    agreed to 0.5%. The skeleton, not the estimator, carried the inconsistency.

    So: pick n_total/group anchors by FPS (coverage), then fill each anchor out with its
    `group` nearest neighbours in descriptor space (local stereo baselines). Same frame
    budget, same global spread, but every skeleton frame now has co-visible partners.
    """
    n_anchor = max(1, int(np.ceil(n_total / max(1, group))))
    anchors = fps_order(Z)[:n_anchor]
    chosen, taken = [], set()
    for a in anchors:
        sims = Z @ Z[a]
        for i in np.argsort(-sims):
            i = int(i)
            if i in taken:
                continue
            taken.add(i); chosen.append(i)
            if len(chosen) % group == 0 or len(chosen) >= n_total:
                break
        if len(chosen) >= n_total:
            break
    for i in fps_order(Z):                      # top up if groups collided
        if len(chosen) >= n_total:
            break
        if i not in taken:
            taken.add(i); chosen.append(int(i))
    return sorted(chosen[:n_total])

def fps_order(X, start=None):
    """Farthest-point-sampling visit order over rows of X (euclidean; both call sites pass
    L2-normalized rows, where euclidean is monotone in cosine). Deterministic: start =
    farthest row from the mean. Skeleton selection takes the first k entries (§4.2,
    licensed pose-free by ZPressor Table B); unordered dedup scans in this order (§4.1)."""
    X = np.asarray(X, np.float32)
    N = len(X)
    if N == 0:
        return []
    if start is None:
        start = int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))
    order = [start]
    d = np.linalg.norm(X - X[start], axis=1)
    for _ in range(N - 1):
        nxt = int(np.argmax(d))
        order.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return order

def select_bridges(Z, member_rows, skeleton_rows, n_bridges, must_include=None):
    """Rank skeleton frames by max whitened cosine to any chunk member (addendum §4.3).
    Returns global rows, best first (the frame-0 reference-slot candidate). The unordered
    assignment anchor replaces the weakest pick when absent."""
    S = Z[skeleton_rows] @ Z[member_rows].T
    score = S.max(axis=1)
    n = max(2, min(4, int(n_bridges)))
    order = np.argsort(-score)
    picked = [int(skeleton_rows[int(a)]) for a in order[:n]]
    if must_include is not None and int(must_include) not in picked:
        picked[-1] = int(must_include)
    return picked

def build_chunk_files(images, member_rows, bridge_rows):
    """Staged list per addendum §4.3: best bridge at frame 0 (DVLT's reference camera —
    every chunk natively expressed in a shared anchor camera's frame), then members, then
    remaining bridges not already members. bridge_mask marks injected copies only; merge
    exclusion of naturally-present skeleton frames is the ownership rule's job."""
    best = bridge_rows[0]
    mem = [r for r in member_rows if r != best]
    mem_set = set(mem)
    tail = [b for b in bridge_rows[1:] if b not in mem_set]
    rows = [best] + mem + tail
    mask = [True] + [False] * len(mem) + [True] * len(tail)
    return [images[r] for r in rows], mask, rows

def assign_unordered_chunks(Z, skeleton_rows, cap):
    """ZPressor-style support→anchor grouping lifted to frame space (addendum §4.5):
    every non-skeleton row joins its nearest anchor by whitened cosine; over-full
    clusters split by FPS sub-anchors; any group still over cap is hard-sliced.
    Returns list of (anchor_row, [member_rows])."""
    skel = [int(r) for r in skeleton_rows]
    skel_set = set(skel)
    rest = [i for i in range(len(Z)) if i not in skel_set]
    A = Z[skel]
    members = {a: [] for a in range(len(skel))}
    for i in rest:
        members[int(np.argmax(A @ Z[i]))].append(i)
    out = []
    for a, mem in members.items():
        if not mem:
            continue
        if len(mem) <= cap:
            out.append((skel[a], mem))
            continue
        n_sub = -(-len(mem) // cap)
        subs = [mem[k] for k in fps_order(Z[mem])[:n_sub]]
        groups = [[] for _ in subs]
        for i in mem:
            groups[int(np.argmax(Z[subs] @ Z[i]))].append(i)
        for g in groups:
            for s0 in range(0, len(g), cap):            # hard-slice the remainder
                if g[s0:s0 + cap]:
                    out.append((skel[a], g[s0:s0 + cap]))
    return out

def detect_order(images):
    """--order auto (addendum §4.5): serial iff sortable numeric filenames or EXIF
    timestamps are present and monotone; else unordered."""
    if len(images) < 3:
        return 'serial'
    nums = []
    for p in images:
        m = re.findall(r'\d+', p.stem)
        if not m:
            nums = None
            break
        nums.append(int(m[-1]))
    if nums is not None and all(b > a for a, b in zip(nums, nums[1:])):
        return 'serial'
    try:
        from PIL import Image
        ts = []
        for p in images:
            ex = Image.open(p)._getexif() or {}
            t = ex.get(36867) or ex.get(306)     # DateTimeOriginal | DateTime
            if t is None:
                return 'unordered'
            ts.append(str(t))
        if all(b >= a for a, b in zip(ts, ts[1:])):
            return 'serial'
    except Exception:
        pass
    return 'unordered'

# ─── v1.5: skeleton health gate (addendum §4.2) ──────────────────────────────
SKEL_FINITE_MIN = 0.30      # provisional until §8 calibration; recorded in the manifest
SKEL_CONF_MED_MIN = 1.05    # conf = exp(x)+1 > 1 always; a degenerate run sits near 1

def skeleton_health(chunk):
    """Gate (a): plausible internal stats after the skeleton run. Gate (b) — ≥2 of the
    first 3 hub edges failing the residual gate — lives in the orchestrator."""
    pts, _, conf = chunk.pts_grid()
    finite = float((np.isfinite(pts).all(-1) & (np.abs(pts) < 1e4).all(-1)).mean())
    med = float(np.median(conf))
    ok = finite >= SKEL_FINITE_MIN and med >= SKEL_CONF_MED_MIN
    return ok, {'finite_frac': round(finite, 4), 'conf_median': round(med, 4),
                'thresholds': {'finite_frac': SKEL_FINITE_MIN, 'conf_median': SKEL_CONF_MED_MIN}}

def _rot_deg(R):
    return float(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

# ─── EXIF GPS ────────────────────────────────────────────────────────────────
def read_exif_gps(paths):
    from PIL import Image
    def rat(x): return float(x[0]) / float(x[1]) if isinstance(x, tuple) else float(x)
    out = np.full((len(paths), 3), np.nan)
    found = False
    for i, p in enumerate(paths):
        try:
            ex = Image.open(p)._getexif() or {}
            g = ex.get(34853)               # GPSInfo IFD
            if not g:
                continue
            lat = sum(rat(v) / 60**k for k, v in enumerate(g[2]))
            lon = sum(rat(v) / 60**k for k, v in enumerate(g[4]))
            if g.get(1) == 'S': lat = -lat
            if g.get(3) == 'W': lon = -lon
            alt = rat(g.get(6, 0.0))
            out[i] = (lat, lon, alt); found = True
        except Exception:
            continue
    return out if found else None

def gps_to_enu(lla):
    lat0, lon0, alt0 = np.nanmean(lla, axis=0)
    east = (lla[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    north = (lla[:, 0] - lat0) * 110540.0
    up = lla[:, 2] - alt0
    return np.stack([east, north, up], axis=1)

def prep_descriptors(raw):
    g = np.sign(raw) * np.sqrt(np.abs(raw))                       # signed power, beta=0.5
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    mu = g.mean(axis=0)
    X = g - mu
    U, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    d = int(min(512, len(Vt) - 1, max(1, len(raw) - 2)))
    if d < 1:
        return np.zeros((len(raw), 1), np.float32)      # degenerate corpus (N < 3)
    comps, evs = Vt[1:1 + d], Sv[1:1 + d]
    evs_r = np.maximum(np.maximum(evs, 0.3 * evs[0]), 1e-12)  # sigma-floor (rel + abs):
    Z = (X @ comps.T) / evs_r               # unregularized whitening amplifies noise; abs
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)   # floor guards zero spectra
    return Z.astype(np.float32)

def loop_candidates(Z, frame_chunk, sim_thresh=None, t_excl=10, nms=25, top_k=5,
                    unordered=False):
    """Descriptor retrieval for loop candidates.

    Serial: v1 exactly (index-band exclusion, AUTO-tau from the |i-j|>t_excl null, NMS
    +/-nms frames). Unordered: every index-based rule is replaced by its chunk-based
    analogue (addendum §4.5) - the null is sampled from random CROSS-CHUNK pairs, the
    proximity gate is "different chunk assignment", and NMS keeps the single best pair
    per (chunk, chunk) combination. Same 99.9th-percentile rule either way.
    """
    sims = Z @ Z.T
    N = len(Z)
    fc = np.asarray(frame_chunk)
    if sim_thresh is None:
        if unordered:
            iu = np.triu_indices(N, k=1)
            cross = sims[iu][fc[iu[0]] != fc[iu[1]]]
            sim_thresh = float(np.percentile(cross, 99.9)) if len(cross) else 1.0
        else:
            iu = np.triu_indices(N, k=t_excl + 1)
            far = sims[iu]
            sim_thresh = float(np.percentile(far, 99.9)) if len(far) else 1.0
    pairs = {}
    for i in range(N):
        row = sims[i].copy()
        if unordered:
            row[fc == fc[i]] = -1                     # intra-chunk pairs carry no new constraint
        else:
            lo, hi = max(0, i - t_excl), min(N, i + t_excl + 1)
            row[lo:hi] = -1
        for j in np.argsort(row)[-top_k:]:
            j = int(j)
            if row[j] < sim_thresh or fc[i] == fc[j]:
                continue
            a, b = min(i, j), max(i, j)
            pairs[(a, b)] = max(pairs.get((a, b), -1), float(row[j]))
    ranked = sorted(((a, b, s) for (a, b), s in pairs.items()), key=lambda x: -x[2])
    kept, seen_cc = [], set()
    for a, b, s in ranked:
        if unordered:                                  # NMS: best pair per chunk-pair
            cc = (min(fc[a], fc[b]), max(fc[a], fc[b]))
            if cc in seen_cc:
                continue
            seen_cc.add(cc)
        elif any(abs(a - ka) <= nms and abs(b - kb) <= nms for ka, kb, _ in kept):
            continue                                   # greedy NMS in frame index
        kept.append((a, b, s))
    return kept

def gps_candidates(enu, frame_chunk, radius=15.0, t_excl=60, unordered=False):
    """GPS proposals are distance-based and survive unordered ingestion unchanged
    (addendum §4.5); only the index-separation gate becomes a chunk gate."""
    ok = np.isfinite(enu).all(axis=1)
    idx = np.where(ok)[0]
    kept = []
    for ii, i in enumerate(idx):
        d = np.linalg.norm(enu[idx[ii + 1:]] - enu[i], axis=1)
        for jj in np.where(d < radius)[0]:
            j = int(idx[ii + 1 + jj])
            far_enough = (frame_chunk[i] != frame_chunk[j]) if unordered else (j - i > t_excl)
            if far_enough and frame_chunk[i] != frame_chunk[j]:
                if not any(abs(i - ka) <= 25 and abs(j - kb) <= 25 for ka, kb, _ in kept):
                    kept.append((int(i), j, 1.0))
    return kept

# ─── Loop-edge composition, verification gate ────────────────────────────────
def _gate_ok(med_a, med_b, seam_median, scale=3.0):
    if not (np.isfinite(med_a) and np.isfinite(med_b)):
        return False                       # builtin max() silently drops a 2nd-arg NaN
    return max(med_a, med_b) < scale * max(seam_median, 1e-9)

def _consensus_ok(info, min_frames=2, allow_weak=True):
    """May this edge enter the pose graph?

    Only GOOD (and WEAK, deliberately - a single shared frame is usually a legitimate
    long-range constraint) are admitted. AMBIGUOUS/NONRIGID are refused on principle: a
    partial map with explicit disconnected components beats one ghosted conglomeration,
    and the pose graph must never be handed inconsistency to average away."""
    st = info.get('state')
    if st is None:                         # legacy info dicts (v1 fallback path)
        nf, nc = info.get('n_frames', 0), info.get('n_consensus', 0)
        return nf <= 1 or nc >= min_frames
    return st == EDGE_GOOD or (allow_weak and st == EDGE_WEAK)

def _window(center_local, n_frames, half=10):
    lo = max(0, center_local - half)
    hi = min(n_frames, center_local + half)
    return list(range(lo, hi))

def _desc_window(Z, chunk, center_local, n=20):
    """Unordered loop window (addendum §4.5): the chunk's n members nearest to the
    endpoint in whitened descriptor space (temporal ±10 has no meaning without order)."""
    rows = chunk.g_rows
    cand = [li for li in range(len(chunk.files)) if not chunk.bridge_mask[li]]
    sims = np.array([float(Z[rows[li]] @ Z[rows[center_local]]) for li in cand])
    top = np.argsort(-sims)[:n]
    return sorted(int(cand[t]) for t in top)

def build_loop_edge(pair, chunks, frame_chunk, frame_local, cfg, seam_median, Z=None, exe_sha=''):
    i, j, sim = pair
    ci, cj = int(frame_chunk[i]), int(frame_chunk[j])
    if ci == cj:
        return None
    if ci > cj:
        ci, cj, i, j = cj, ci, j, i
    A, B = chunks[ci], chunks[cj]
    if Z is None:                        # serial: temporal window around each endpoint
        la = _window(int(frame_local[i]), len(A.files))
        lb = _window(int(frame_local[j]), len(B.files))
    else:                                # unordered: descriptor-nearest members
        la = _desc_window(Z, A, int(frame_local[i]))
        lb = _desc_window(Z, B, int(frame_local[j]))
    loop_files = [A.files[k] for k in la] + [B.files[k] for k in lb]
    key = chunk_key(loop_files, cfg.img_size, cfg.loop_steps, exe_sha)
    out = Path(cfg.out) / 'loops' / f"loop_{i}_{j}_{key}"
    run_dvlt(cfg.dvlt, loop_files, out / 'out', cfg.img_size, cfg.loop_steps)
    L = Chunk(-1, 0, loop_files, out / 'out', role='loop')
    lp, _, lc = L.pts_grid()
    na = len(la)

    def half_align(local_idx, parent, half_pts, half_conf):
        pp, _, pc = parent.pts_grid()
        P = pp[local_idx].reshape(-1, 3); C = pc[local_idx].reshape(-1)
        Q = half_pts.reshape(-1, 3);      D = half_conf.reshape(-1)
        ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        th = 0.1 * min(np.median(C[ok]), np.median(D[ok]))
        ok &= (C > th) & (D > th)
        P, Q, w = P[ok], Q[ok], np.sqrt(C[ok] * D[ok])
        if len(P) > 100_000:
            sel = np.linspace(0, len(P) - 1, 100_000).astype(int)
            P, Q, w = P[sel], Q[sel], w[sel]
        return irls_sim3(Q, P, w)                       # loop coords -> parent coords

    Sa, ia = half_align(la, A, lp[:na], lc[:na])
    Sb, ib = half_align(lb, B, lp[na:], lc[na:])
    if not (ia['converged'] and ib['converged'] and
            _gate_ok(ia['med_resid'], ib['med_resid'], seam_median, cfg.gate_scale)):
        return None
    S_ab = sim3_compose(Sa, sim3_inv(Sb))               # chunk-cj coords -> chunk-ci coords
    # ghost sample via the loop chunk: the same physical points expressed in ci coords
    # (through Sa) and cj coords (through Sb) - post-graph validation for loop edges too.
    l_pts = lp.reshape(-1, 3)
    l_fin = l_pts[np.isfinite(l_pts).all(1)]
    ghost_sample = None
    if len(l_fin):
        gs = np.linspace(0, len(l_fin) - 1, min(2048, len(l_fin))).astype(int)
        ghost_sample = (sim3_apply(Sa, l_fin[gs]).astype(np.float32),
                        sim3_apply(Sb, l_fin[gs]).astype(np.float32))
    return ci, cj, S_ab, {'med_a': ia['med_resid'], 'med_b': ib['med_resid'],
                          'gate_ref': seam_median, 'sim': sim, 'ghost_sample': ghost_sample}

def edge_weight(med_resid, seam_median):
    """Confidence weight from a measured alignment residual: 1.0 at or below
    the corpus median, decaying as the residual grows. Gentle 1/r (not 1/r^2)
    so bad seams are downweighted, not severed - a severed chain leaves the
    optimizer nothing to correct from."""
    m = max(seam_median, 1e-12)
    return float(m / max(med_resid, m))

def edge_ghosts(edge_list, S_by_idx):
    """POST-graph ghost per edge: apply each chunk's FINAL optimized Sim(3) to the edge's
    cached correspondence sample and measure how far apart the same physical pixels land,
    as a fraction of scene extent (p95 radius over all samples in world coords).

    [MEASURED on a 723-frame indoor set] Every other gate runs in chunk-local coordinates BEFORE
    the graph moves anything: edges passed the residual gate and still placed the same
    pixel 2.28 scene-widths apart after optimization (rim 8<-9, ghost 228% of extent). An
    edge can be locally excellent and globally catastrophic; this is the only metric in
    the pipeline that looks at the product. Invariant to the global gauge/ENU transform
    (both sides move identically and the extent scales with them).

    edge_list: [(i, j, est, info)] with chunk idxs. Returns ([ghost_frac | None], extent).
    """
    world = []
    for i, j, _e, info in edge_list:
        smp = info.get('ghost_sample')
        if smp is not None and i in S_by_idx and j in S_by_idx:
            world.append(sim3_apply(S_by_idx[i], smp[0]))
    if not world:
        return [None] * len(edge_list), 0.0
    allp = np.concatenate(world)
    extent = float(np.percentile(np.linalg.norm(allp - allp.mean(0), axis=1), 95))
    out = []
    for i, j, _e, info in edge_list:
        smp = info.get('ghost_sample')
        if smp is None or i not in S_by_idx or j not in S_by_idx:
            out.append(None)
            continue
        d = np.linalg.norm(sim3_apply(S_by_idx[i], smp[0])
                           - sim3_apply(S_by_idx[j], smp[1]), axis=1)
        out.append(float(np.median(d)) / max(extent, 1e-12))
    return out, extent

def _reachable_from_gauge(pairs, root=0):
    """Chunk ids reachable from the gauge over undirected edge pairs."""
    adj = {}
    for i, j in pairs:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    seen = {root}
    q = deque([root])
    while q:
        for v in adj.get(q.popleft(), ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen

def _init_from_edges(n_chunks, edges):
    """Spanning-tree init from the gauge node (BFS over measured relatives).

    Chain, hub and wheel all initialize through this one path, but the *shape* of the
    graph decides how much composition happens: a chain composes k relatives to reach
    chunk k (one bad seam multiplies into every later chunk — the measured 29x
    cascade), while a hub reaches every chunk in one hop, so a bad edge corrupts exactly
    one node before the optimizer has even run. That is the containment claim of
    addendum §4.4, and it lives here, not in the solver.

    Edge convention (unchanged from v1): (i, j, m) constrains m ~= inv(S_i) . S_j.
    """
    S = [None] * n_chunks
    S[0] = (1.0, np.eye(3), np.zeros(3))       # gauge (skeleton in hub mode, chunk 0 in chain)
    adj = {}
    for i, j, m in edges:
        adj.setdefault(i, []).append((j, m, False))
        adj.setdefault(j, []).append((i, m, True))
    q = deque([0])
    while q:
        u = q.popleft()
        for v, m, inverted in adj.get(u, ()):
            if S[v] is not None:
                continue
            #  S_j = S_i . m   (forward)      S_i = S_j . inv(m)   (reverse)
            S[v] = sim3_compose(S[u], sim3_inv(m)) if inverted else sim3_compose(S[u], m)
            q.append(v)
    unreached = [k for k in range(n_chunks) if S[k] is None]
    for k in unreached:
        S[k] = (1.0, np.eye(3), np.zeros(3))   # caller gates these (orphans, §4.5)
    return S, unreached

def optimize_pose_graph(n_chunks, seq_edges, loop_edges, gps=None,
                        seq_w=None, loop_w=None):
    for i, j, meas in seq_edges:
        if not (np.isfinite(meas[0]) and meas[0] > 0):
            sys.exit(f"pose graph: invalid seq-edge scale {meas[0]} on {i}->{j} (degenerate seam fit?)")
    for i, j, meas, _info in loop_edges:
        if not (np.isfinite(meas[0]) and meas[0] > 0):
            sys.exit(f"pose graph: invalid loop-edge scale {meas[0]} on {i}->{j}")
    S, unreached = _init_from_edges(n_chunks, [(i, j, m) for i, j, m in seq_edges]
                                    + [(i, j, m) for i, j, m, _ in loop_edges])
    if unreached:
        print(f"  WARNING: chunks {unreached} unreachable from gauge - initialized to identity "
              f"(orphan handling should have removed them)")
    if n_chunks == 1:
        return S
    # Gauge-consistency for the GPS term. Raw ENU is in METERS while the chained
    # map is in the model's arbitrary units with chunk 0 pinned to identity.
    # Feeding metric targets straight into the residual lets meter-sized numbers
    # crush the ~0.01 seam residuals: the optimizer stretches free chunks toward
    # metric scale while chunk 0 cannot move, TEARING the map (the 2026-07-15
    # "fanning/lasagna separation"; root-caused in fastvggt-win 2026-08-06,
    # per-chunk scales 1.4..99 on a 172-frame strip). So ENU anchors are first
    # mapped INTO the visual gauge with one robust fit against the chained init
    # - inside the graph GPS only corrects drift. Metric scale is restored by
    # main()'s final map->ENU refit.
    if gps is not None:
        m_local, enu, w_gps = gps
        ok = np.isfinite(enu).all(axis=1) & np.isfinite(m_local).all(axis=1)
        if ok.sum() >= 3:
            anchors_init = np.stack([sim3_apply(S[k], m_local[k][None])[0]
                                     for k in np.where(ok)[0]])
            sv = np.linalg.svd(enu[ok] - enu[ok].mean(0), compute_uv=False)
            if sv[1] < 0.05 * sv[0]:
                gps = None          # collinear anchors: rotation unconstrained
            else:
                # Huber-IRLS, not plain Umeyama: a drifted chunk is an OUTLIER
                # anchor, and a least-squares fit would lean toward it -
                # moving the GPS target with the drift it exists to correct.
                spread = float(np.linalg.norm(
                    anchors_init - anchors_init.mean(0), axis=1).mean())
                G0, _ = irls_sim3(enu[ok], anchors_init, np.ones(int(ok.sum())),
                                  delta=max(0.05 * spread, 1e-9))  # ENU -> gauge
                enu_gauge = np.full_like(enu, np.nan)
                enu_gauge[ok] = sim3_apply(G0, enu[ok])
                gps = (m_local, enu_gauge, w_gps)
        else:
            gps = None
    x0 = np.concatenate([sim3_log(S[k]) for k in range(1, n_chunks)])
    sw = seq_w  if seq_w  is not None else [1.0] * len(seq_edges)
    lw = loop_w if loop_w is not None else [1.0] * len(loop_edges)
    edges = [(i, j, m, w) for (i, j, m), w in zip(seq_edges, sw)] + \
            [(i, j, m, w) for (i, j, m, _), w in zip(loop_edges, lw)]

    def unpack(x):
        out = [(1.0, np.eye(3), np.zeros(3))]
        for k in range(n_chunks - 1):
            out.append(sim3_exp(x[7 * k: 7 * k + 7]))
        return out

    def residuals(x):
        Sx = unpack(x)
        rs = []
        for i, j, meas, w in edges:
            d = sim3_compose(sim3_inv(meas), sim3_compose(sim3_inv(Sx[i]), Sx[j]))
            rs.append(w * sim3_log(d))
        if gps is not None:
            m_local, enu, w_gps = gps
            for k in range(n_chunks):
                if np.isfinite(enu[k]).all() and np.isfinite(m_local[k]).all():
                    rs.append(w_gps * (sim3_apply(Sx[k], m_local[k][None])[0] - enu[k]))
        return np.concatenate(rs)

    nfev_cap = max(2000, 30 * (len(x0) + 1))            # 2-pt jacobian: ~(n+1) evals/LM iter;
    try:                                                 # keep >=30 iterations at any graph size
        sol = least_squares(residuals, x0, method='lm', max_nfev=nfev_cap)
    except ValueError:                                  # lm needs m>=n
        sol = least_squares(residuals, x0, method='trf', max_nfev=nfev_cap)
    optimize_pose_graph.last_nfev = int(sol.nfev)       # manifest graph.iters (spec §4.7)
    return unpack(sol.x)

# ─── CLI orchestration ────────────────────────────────────────────────────────
def _rows_of(c):
    return c.g_rows if c.g_rows is not None else list(range(c.g_start, c.g_start + len(c.files)))

def _owner_map(chunks, n_images):
    """global frame row -> chunk owning it (emitting its points and its pose).

    [INVARIANT §4.3] Injected bridge copies never own: the same physical frame is
    inferred in two contexts, and without this rule v1's most-central-local-index rule
    would let a frame-0 bridge outrank the skeleton and double-emit it. Skeleton frames
    are owned by the skeleton chunk unconditionally (d = -1 beats every local's centrality).
    """
    owner = np.full(n_images, -1, int); best = np.full(n_images, np.inf)
    for c in chunks:
        rows, n = _rows_of(c), len(c.files)
        for li, g in enumerate(rows):
            if li < len(c.bridge_mask) and c.bridge_mask[li]:
                continue
            d = -1.0 if c.role == 'skeleton' else abs(li - (n - 1) / 2)
            if d < best[g]:
                best[g], owner[g] = d, c.idx
    return owner

def _owned_points(c, owner, conf_keep_coef):
    """Points of the frames this chunk owns, confidence-filtered (v1 merge rule)."""
    pts, rgb, conf = c.pts_grid()
    rows = _rows_of(c)
    own = [li for li in range(len(c.files))
           if owner[rows[li]] == c.idx and not (li < len(c.bridge_mask) and c.bridge_mask[li])]
    if not own:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    p = pts[own].reshape(-1, 3); r = rgb[own].reshape(-1, 3); cf = conf[own].reshape(-1)
    m = np.isfinite(p).all(1) & (cf > conf_keep_coef * cf.mean())
    return p[m], r[m]

def _open_overwrite(path, attempts=4, wait=2.0):
    """Open for binary write, retrying briefly when a viewer/indexer holds the previous
    file (observed: inspecting a merged PLY while a re-run replaces it). After the
    retries, fall back to a sibling '<stem>.new.ply' rather than dying on the very last
    step of a long run. Returns (file_object, actual_path)."""
    path = Path(path)
    for a in range(attempts):
        try:
            return open(path, 'wb'), path
        except PermissionError:
            if a + 1 < attempts:
                time.sleep(wait)
    alt = path.with_name(path.stem + '.new' + path.suffix)
    print(f"  WARNING: {path.name} locked by another process - writing {alt.name} instead")
    return open(alt, 'wb'), alt

PLY_HEADER = ("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")

def _ply_records(p, r):
    rec = np.zeros((len(p), 15), np.uint8)
    rec[:, :12] = np.ascontiguousarray(p, np.float32).view(np.uint8).reshape(len(p), 12)
    rec[:, 12:] = r
    return rec.tobytes()

class PingPly:
    """Incremental PLY for --profile live --ping-out: append-only body with a
    fixed-width vertex count patched in place, so a ping costs O(new points) rather
    than re-copying the whole cloud (the whole point of a <10 s ping). The final
    `merged/scene.ply` still uses the exact two-pass writer below."""
    def __init__(self, path):
        self.path = Path(path); self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, 'wb')
        self.f.write(PLY_HEADER.format(n=f"{0:012d}").encode())
        self.hdr_pos = PLY_HEADER.format(n="").encode().index(b"element vertex ") + len("element vertex ")
    def add(self, p, r):
        self.f.write(_ply_records(p, r)); self.n += len(p)
        self.f.flush()
        self.f.seek(self.hdr_pos); self.f.write(f"{self.n:012d}".encode())
        self.f.seek(0, os.SEEK_END); self.f.flush()
    def close(self):
        self.f.close()

def plan_chunks(cfg, images, Z, kept_rows, order, topology):
    """Decide the run's chunk set (addendum §4.2/§4.3/§4.5).

    chain     -> v1 windows over the surviving frames, no skeleton, no bridges.
    hub       -> skeleton chunk 0 (FPS in whitened space) + locals, each local carrying
                 2-4 injected skeleton frames so its hub edge is the existing Stage-3
                 pixel-indexed IRLS, verbatim. Serial locals stay capture-order windows
                 (their free adjacent overlap becomes the wheel's rim); unordered locals
                 are retrieval clusters around skeleton anchors.
    Returns (chunks, skeleton | None).
    """
    n_bridges = max(2, min(4, cfg.bridges))
    if topology == 'chain':
        sub = [images[r] for r in kept_rows]
        chunks = make_chunks(sub, cfg.chunk_size, cfg.overlap)
        for c in chunks:
            c.g_rows = kept_rows[c.g_start:c.g_start + len(c.files)]
            c.k = cfg.loop_steps
        return chunks, None

    # ── skeleton: FPS over survivors in whitened space (pose-free anchor selection,
    #    ZPressor Table B); frozen thereafter [D10]
    n_skel = max(3, min(cfg.skeleton_size, VRAM_FRAME_CAP, len(kept_rows) - 1))
    skel_local = (fps_order(Z)[:n_skel] if cfg.skeleton_mode == 'fps'
                  else skeleton_rows_clustered(Z, n_skel, cfg.skeleton_group))
    skel_rows = sorted(int(kept_rows[i]) for i in skel_local)
    skel_pos = {int(kept_rows[i]): i for i in skel_local}      # global row -> row in Z
    zrow = {int(g): i for i, g in enumerate(kept_rows)}

    skeleton = Chunk(0, 0, [images[r] for r in skel_rows], "", role='skeleton')
    skeleton.g_rows = list(skel_rows)
    skeleton.k = cfg.skeleton_k

    cap = cfg.chunk_size - n_bridges          # effective new content per chunk (§4.3)
    groups = []                               # (anchor_global_row | None, [member rows])
    if order == 'unordered':
        skel_z = [zrow[r] for r in skel_rows]
        for anchor_i, mem_i in assign_unordered_chunks(Z, skel_z, cap):
            groups.append((int(kept_rows[anchor_i]), [int(kept_rows[m]) for m in mem_i]))
    else:
        sub = [images[r] for r in kept_rows]
        for w in make_chunks(sub, cap, min(cfg.overlap, cap - 1)):
            groups.append((None, [int(kept_rows[r]) for r in
                                  range(w.g_start, w.g_start + len(w.files))]))

    chunks = [skeleton]
    for gi, (anchor, mem) in enumerate(groups):
        mem_z = [zrow[m] for m in mem]
        skel_z = [zrow[r] for r in skel_rows]
        picks_z = select_bridges(Z, mem_z, skel_z, n_bridges,
                                 must_include=(zrow[anchor] if anchor is not None else None))
        bridge_rows = [int(kept_rows[p]) for p in picks_z]
        files, mask, rows = build_chunk_files(images, mem, bridge_rows)
        c = Chunk(gi + 1, min(mem), files, "", role='local')
        c.g_rows, c.bridge_mask, c.k = rows, mask, cfg.loop_steps
        chunks.append(c)
    return chunks, skeleton


def _resolve_profile(cfg):
    """--profile live|final sets the defaults; an explicit flag always wins (the
    profile-controlled options default to None so 'given' is distinguishable).

    live  (§4.6/D9): chain @ K=8 - the bottom of DVLT's trained K range, so in
          distribution by construction - no dedup/skeleton/loops, GPS unaries on,
          postprocess off. Drift is accepted and GPS-bounded; the product is a
          situational map while flying, not the deliverable.
    final: dedup -> skeleton -> hub (wheel if serial) @ K=16, loops on, full merge.
    """
    live = cfg.profile == 'live'
    # Live defaults are MEASURED, not assumed (§8.10 grid, 4060 Ti, 60-frame ping windows,
    # 2026-07-19): K is nearly free (8->16 costs 0.5 s at 336 px) while RESOLUTION is the
    # latency lever (336->504 px costs ~3.5 s). Every K meets the <10 s ping at 336 px;
    # none does at 392+. So live runs K=16 @ 336 - full loop quality, resolution paying
    # for the ping - overturning D9's K=8 @ 504 guess. NOTE live 336-px chunks can never seed a final
    # 504-px pass (pixel-indexed correspondence needs one grid) - the §4.6 honest-cache
    # statement already priced that in, and the content key keeps them separate.
    defaults = {'loop_steps': 16,
                'img_size': 336 if live else 504,
                'topology': 'chain' if live else 'auto',
                'order': 'serial' if live else 'auto',
                'loop': 'off' if live else 'auto',
                'dedup': 'off'}    # off in BOTH profiles pending the §8.6 recalibration;
                                   # an explicit --dedup auto always wins, even under live
    for k, v in defaults.items():
        if getattr(cfg, k) is None:
            setattr(cfg, k, v)
    if cfg.no_postprocess is None:
        cfg.no_postprocess = live
    return cfg


def _gc_cache_report(cfg, chunks):
    """Stale content-keyed dirs are reported, never auto-deleted (§6)."""
    live_keys = {c.out_dir.parent.name for c in chunks}
    root = cfg.out / 'chunks'
    stale = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name not in live_keys] \
        if root.exists() else []
    print(f"  --gc-cache: {len(stale)} stale chunk dir(s) under {root} "
          f"(not deleted; remove manually if the disk matters)")
    for d in stale:
        print(f"    stale {d.name}")
    return [str(d) for d in stale]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('image_dir', help='image folder or .txt list; every path must fit the '
                                      'ANSI code page (dvlt.exe opens paths through it)')
    ap.add_argument('output_dir')
    ap.add_argument('--chunk-size', type=int, default=60)
    ap.add_argument('--overlap', type=int, default=30)
    ap.add_argument('--loop-steps', type=int, default=None)      # profile-controlled
    ap.add_argument('--img-size', type=int, default=None)        # profile-controlled:
                                                                 # live 336, final 504 (§8.10)
    ap.add_argument('--conf-keep-coef', type=float, default=0.75)
    ap.add_argument('--max-points', type=int, default=25_000_000)
    ap.add_argument('--loop', choices=['auto', 'off'], default=None)
    ap.add_argument('--loop-sim', default='auto')  # 'auto' -> 99.9th pct; or numeric
    ap.add_argument('--gate-scale', type=float, default=3.0)
    ap.add_argument('--gps', choices=['auto', 'off'], default='auto')
    ap.add_argument('--dvlt', default=str(Path(__file__).parent / 'dvlt.exe'))
    ap.add_argument('--formats', default='ply,glb,las',
                    help='passed through to dvlt_postprocess.py (default: ply,glb,las)')
    ap.add_argument('--no-postprocess', action='store_true', default=None)
    # ── v1.5 (addendum §5)
    ap.add_argument('--profile', choices=['live', 'final'], default='final')
    ap.add_argument('--topology', choices=['hub', 'chain', 'auto'], default=None)
    ap.add_argument('--order', choices=['serial', 'unordered', 'auto'], default=None)
    # dedup defaults OFF pending a spec decision: the transferred 0.95 was REFUTED by the
    # §8.6 calibration on a 723-frame indoor set. In power-normalized space our
    # far pairs average 0.789 (p95 0.903) against adjacent 0.982 - only 0.193 of separation -
    # so every threshold in 0.90..0.99 saturates the min-keep clamp at exactly 50% dropped.
    # The clamp, not the threshold, was selecting. 0.999 is the measured-conservative point
    # (0.6% dropped) but sits on a cliff (0.995 -> 43.7%), so an absolute cosine cut looks
    # unsuited to DVLT descriptors. Opt in with --dedup auto; see §10 D4 (needs a new entry).
    ap.add_argument('--dedup', choices=['auto', 'off'], default=None)   # profile-resolved; off
    ap.add_argument('--dedup-sim', type=float, default=0.999)
    ap.add_argument('--dedup-min-keep', type=float, default=0.5)
    ap.add_argument('--skeleton-size', type=int, default=48)
    ap.add_argument('--skeleton-mode', choices=['clustered', 'fps'], default='clustered',
                    help="'clustered' (default): FPS anchors filled out with their nearest "
                         "neighbours, so skeleton frames have co-visible partners to "
                         "triangulate against. 'fps': v1.5 pure farthest-point (coverage "
                         "only) - measurably produces split-scale hub edges on a 723-frame indoor set.")
    ap.add_argument('--skeleton-group', type=int, default=4)
    ap.add_argument('--skeleton-k', type=int, default=16)        # skeleton always K=16 (§4.2)
    ap.add_argument('--bridges', type=int, default=3)
    ap.add_argument('--keep-orphans', action='store_true')
    ap.add_argument('--ping-out', action='store_true')
    ap.add_argument('--gc-cache', action='store_true')
    ap.add_argument('--consensus-tol', type=float, default=0.05,
                    help='frame-consensus tolerance: |log scale ratio| + rotation (rad); '
                         '0.05 ~= 5%% scale or 2.9 deg')
    ap.add_argument('--ghost-max', type=float, default=0.15,
                    help='post-graph validation: reject edges whose shared pixels land more '
                         'than this fraction of scene extent apart AFTER optimization, then '
                         're-solve (measured on a 723-frame indoor set: healthy edges ghost 1-3%%, broken '
                         'ones 30-228%% and they still passed the pre-graph gates). '
                         '<=0 disables')
    # Intermediates are ~170 MB per chunk (a 24-chunk run is ~4 GB). RETAINED BY DEFAULT
    # [2026-07-19]: per-chunk pointmaps, poses, descriptors and bridge identities
    # are the only artifacts that can distinguish bad matching / bad transform estimation /
    # contextual model inconsistency / genuine non-rigid error after the fact. Deleting them
    # turns a diagnosable failure into an anecdote. Opt in with --drop-chunks once a run is
    # understood and you only want the merged product.
    ap.add_argument('--drop-chunks', action='store_true',
                    help='delete per-chunk dvlt outputs after a successful merge to reclaim '
                         'disk (default: keep - they are the alignment evidence)')
    ap.add_argument('--keep-chunks', action='store_true',
                    help=argparse.SUPPRESS)   # deprecated no-op: keeping is now the default
    cfg = _resolve_profile(ap.parse_args())
    # dvlt.exe runs in its own folder, so every path it is handed (-o, -w, list) is absolute
    cfg.out = Path(cfg.output_dir).resolve()
    cfg.dvlt = str(Path(cfg.dvlt).resolve())
    t_start = time.time()
    timings = {}

    if not Path(cfg.dvlt).exists():
        sys.exit(f"dvlt.exe not found at {cfg.dvlt} (--dvlt PATH)")
    exe_sha = file_sha1(cfg.dvlt)
    images = list_images(cfg.image_dir)

    order = detect_order(images) if cfg.order == 'auto' else cfg.order
    topology = cfg.topology
    if topology == 'auto':
        # Hub only where it is REQUIRED (unordered input has no chain to fall back on).
        # The wheel LOST its serial A/B on a 723-frame indoor set (chain 2.30x vs wheel 3.46x scale
        # spread), so serial corpora default to chain until a
        # §10 decision revives it. Explicit --topology hub remains available for A/Bs.
        topology = 'hub' if (order == 'unordered' and len(images) > cfg.chunk_size) else 'chain'
    if order == 'unordered' and topology == 'chain' and len(images) > cfg.chunk_size:
        # a corpus that fits ONE window is order-free inside a single dvlt run, so small
        # unordered dumps go through chain as a single chunk instead of erroring out
        sys.exit("--order unordered requires --topology hub (chain topology assumes capture order)")
    print(f"dvlt_long v1.5: {len(images)} images | profile={cfg.profile} topology={topology} "
          f"order={order} K={cfg.loop_steps} img-size={cfg.img_size}")

    manifest = {'config': {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in vars(cfg).items()},
                'exe_sha': exe_sha, 'resolved': {'order': order, 'topology': topology},
                'frames': [], 'chunks': [], 'edges': [], 'graph': {}, 'timings': {}}

    # ── 1. corpus descriptors (before any chunking decision exists, §4.0) + dedup (§4.1)
    kept_rows, dropped, Z = list(range(len(images))), {}, None
    need_desc = (cfg.dedup != 'off') or topology == 'hub' or order == 'unordered'
    if need_desc:
        t0 = time.time()
        # content-keyed like every other cached dir (§4.7; confirmed review finding: a bare
        # 'encode' dir would silently satisfy a run with different images/img_size/exe)
        enc_dir = cfg.out / f"encode_{chunk_key(images, cfg.img_size, -1, exe_sha)}"
        raw = run_dvlt_encode(cfg.dvlt, images, enc_dir, cfg.img_size)
        timings['encode_s'] = round(time.time() - t0, 1)
        print(f"  encode: {len(raw)} descriptors ({timings['encode_s']}s)")
        if cfg.dedup != 'off':
            scan = None
            if order == 'unordered':      # no capture order to scan along -> FPS visit order,
                scan = fps_order(power_normalize(raw))    # computed in the dedup space itself
            kept_rows, dropped, clamped = dedup_frames(raw, cfg.dedup_sim,
                                                       cfg.dedup_min_keep, scan_order=scan)
            print(f"  dedup(sim>{cfg.dedup_sim}): kept {len(kept_rows)}/{len(images)}, "
                  f"dropped {len(dropped)}" + (" [MIN-KEEP CLAMP FIRED]" if clamped else ""))
            manifest['dedup'] = {'sim': cfg.dedup_sim, 'min_keep': cfg.dedup_min_keep,
                                 'kept': len(kept_rows), 'dropped': len(dropped),
                                 'clamped': bool(clamped)}
        Z = prep_descriptors(raw[kept_rows])          # whitened space: retrieval/FPS/assignment
    for g, img in enumerate(images):
        d = ('dedup_dropped' if g in dropped else 'kept')
        rec = {'path': str(img), 'disposition': d}
        if g in dropped:
            rec['duplicate_of'] = str(images[dropped[g][0]]); rec['sim'] = round(dropped[g][1], 4)
        manifest['frames'].append(rec)

    # ── 2. chunk plan + inference (content-keyed cache, §4.7)
    chunks, skeleton = plan_chunks(cfg, images, Z, kept_rows, order, topology)
    for c in chunks:
        c.key = chunk_key(c.files, cfg.img_size, c.k, exe_sha)
        c.out_dir = cfg.out / 'chunks' / f"{'skel' if c.role == 'skeleton' else f'chunk_{c.idx:03d}'}_{c.key}" / 'out'
    print(f"  plan: {len(chunks)} chunks"
          + (f" (skeleton {len(skeleton.files)} frames @ K={skeleton.k} + "
             f"{len(chunks)-1} locals)" if skeleton else f" (chain, L={cfg.chunk_size} O={cfg.overlap})"))
    if cfg.gc_cache:
        manifest['stale_cache_dirs'] = _gc_cache_report(cfg, chunks)

    # ping assumes chain layout (fresh frames = everything past the overlap); in hub mode
    # slot 0 is an injected bridge and the tail holds bridge copies, so the slice would
    # drop real geometry and emit copies (confirmed review finding) - chain-only.
    if cfg.ping_out and topology != 'chain':
        print("  --ping-out ignored: pings are chain-only (hub staging breaks the "
              "fresh-frame slice)")
    ping = PingPly(cfg.out / 'merged' / 'latest.ply') if (cfg.ping_out and
                                                          topology == 'chain') else None
    ping_S = (1.0, np.eye(3), np.zeros(3))
    t0 = time.time()
    for ci, c in enumerate(chunks):
        run_dvlt(cfg.dvlt, c.files, c.out_dir, cfg.img_size, c.k)
        print(f"  {c.role} {c.idx:03d}: {len(c.files)} frames (K={c.k}) ok")
        if c.role == 'skeleton':          # health gate (a) before anything depends on it
            ok, stats = skeleton_health(c)
            manifest['skeleton_health'] = dict(stats, passed=bool(ok))
            print(f"  skeleton health: finite={stats['finite_frac']:.3f} "
                  f"conf_med={stats['conf_median']:.2f} -> {'OK' if ok else 'FAIL'}")
            if not ok:
                if order == 'unordered':
                    sys.exit("skeleton health gate FAILED and assignment depended on it "
                             "(--order unordered): aborting rather than anchoring to a bad gauge")
                print("  demoting to chain topology (skeleton unhealthy, serial input)")
                topology = 'chain'
                chunks, skeleton = plan_chunks(cfg, images, Z, kept_rows, order, 'chain')
                for c2 in chunks:
                    c2.key = chunk_key(c2.files, cfg.img_size, c2.k, exe_sha)
                    c2.out_dir = cfg.out / 'chunks' / f"chunk_{c2.idx:03d}_{c2.key}" / 'out'
                manifest['resolved']['topology'] = 'chain (demoted: skeleton health)'
                for c2 in chunks:
                    run_dvlt(cfg.dvlt, c2.files, c2.out_dir, cfg.img_size, c2.k)
                break
        if ping is not None and c.role != 'skeleton':
            # in-flight ping: chain-compose against the previous chunk and append only the
            # new frames. Approximate by construction (pre-graph, provisional ownership) -
            # situational awareness, not the deliverable (§4.6).
            if ci > 0:
                est, _ = align_shared(chunks[ci - 1], c)
                ping_S = sim3_compose(ping_S, est)
            pts, rgb, conf = c.pts_grid()
            fresh = list(range(0 if ci == 0 else min(cfg.overlap, len(c.files) - 1), len(c.files)))
            p = pts[fresh].reshape(-1, 3); r = rgb[fresh].reshape(-1, 3); cf = conf[fresh].reshape(-1)
            m = np.isfinite(p).all(1) & (cf > cfg.conf_keep_coef * cf.mean())
            ping.add(sim3_apply(ping_S, p[m]).astype(np.float32), r[m])
            print(f"    ping: {ping.n:,} pts -> {ping.path}")
    if ping is not None:
        ping.close()
    timings['inference_s'] = round(time.time() - t0, 1)

    # ── 3. edges: hub (local <-> skeleton) and/or rim (adjacent locals)
    t0 = time.time()
    edges, edge_recs = [], []
    hub_meds, rim_meds = [], []
    hub_info = {}

    def _record(kind, i, j, est, info, gate=None):
        edge_recs.append({'type': kind, 'i': int(i), 'j': int(j), 's': round(float(est[0]), 6),
                          'rot_deg': round(_rot_deg(est[1]), 3),
                          'med_resid': (None if not np.isfinite(info['med_resid'])
                                        else round(float(info['med_resid']), 6)),
                          'n': int(info['n']), 'fallback': bool(info.get('fallback')),
                          'n_frames': info.get('n_frames'),
                          'n_consensus': info.get('n_consensus'),
                          'rival_frames': info.get('rival_frames'),
                          'consensus_frac': (None if info.get('consensus_frac') is None
                                             else round(info['consensus_frac'], 3)),
                          'frame_scale_spread': (None if info.get('scale_spread') is None
                                                 else round(info['scale_spread'], 4)),
                          'state': info.get('state'),
                          'per_frame': info.get('per_frame'),   # the evidence, per §diagnostics
                          'ghost': None,                        # filled by post-graph validation
                          'gate': gate})
        info['_rec'] = edge_recs[-1]      # so ghost validation can annotate its record

    def _add_rim(i, j, est, info):
        """A seam that found no usable correspondence is NOT an edge. Appending it as an
        identity transform would mark the chunk 'connected' and quietly place it at the
        wrong pose - the exact silent-merge failure orphan semantics exist to prevent
        (§4.5). Serial windows always overlap, so this fires only on real breakage."""
        usable = info['converged'] and np.isfinite(info['med_resid']) and _consensus_ok(info)
        st = info.get('state', '?')
        _record('rim', i, j, est, info, 'pass' if usable else f'fail({st})')
        tag = ' FALLBACK' if info.get('fallback') else ''
        if usable:
            rim_meds.append(info['med_resid'])
            edges.append((i, j, est, info))
        else:
            tag = f' REJECTED [{st}]'
        print(f"  rim seam {i:03d}->{j:03d}: s={est[0]:.4f} "
              f"med_resid={info['med_resid']:.5f} n={info['n']} "
              f"cons={info.get('n_consensus','?')}/{info.get('n_frames','?')} "
              f"[{st}]{tag if not usable else ('' if not info.get('fallback') else tag)}")

    locals_ = [c for c in chunks if c.role == 'local']
    if topology == 'hub':
        for c in locals_:
            est, info = align_shared(skeleton, c)          # chunk-k coords -> skeleton coords
            hub_info[c.idx] = (est, info)
            if np.isfinite(info['med_resid']):
                hub_meds.append(info['med_resid'])
            print(f"  hub edge skel<-chunk{c.idx:03d}: s={est[0]:.4f} "
                  f"med_resid={info['med_resid']:.5f} n={info['n']}"
                  f"{' FALLBACK' if info.get('fallback') else ''}")
        if order == 'serial':          # wheel: rim seams are free - and computed FIRST so
            for a, b in zip(locals_, locals_[1:]):         # the hub gate calibrates against
                if not shared_frame_indices(a, b):         # an INDEPENDENT baseline
                    continue
                _add_rim(a.idx, b.idx, *align_shared(a, b))
        # Hub-gate baseline per spec §5: rim-seam median when serial, hub median only when
        # unordered (no rims exist). Self-calibration against the hub residuals themselves
        # was a confirmed review finding: a uniformly-sick skeleton inflates its own
        # baseline, so neither this gate nor health gate (b) could ever fire on it.
        base = (float(np.median(rim_meds)) if rim_meds else
                (float(np.median(hub_meds)) if hub_meds else 0.0))
        # health gate (b): >=2 of the first 3 hub edges failing means the gauge is sick
        first3 = [c.idx for c in locals_[:3]]
        bad3 = sum(1 for k in first3
                   if not _gate_ok(hub_info[k][1]['med_resid'], hub_info[k][1]['med_resid'],
                                   base, cfg.gate_scale))
        if len(first3) == 3 and bad3 >= 2:
            msg = (f"skeleton health gate (b) FAILED: {bad3}/3 first hub edges rejected "
                   f"(residuals {[round(hub_info[k][1]['med_resid'], 5) for k in first3]}, "
                   f"baseline {base:.5f})")
            if order == 'unordered':
                sys.exit(msg + " - assignment depended on the skeleton; aborting")
            print("  " + msg + " -> demoting to chain topology")
            # Demotion must actually REMOVE the skeleton: it was just judged an unreliable
            # gauge, so it may not remain chunk 0, may not anchor the graph, and may not
            # emit points. The bridged locals stay (their runs are valid; injected copies
            # are masked); the rim seams already measured carry the chain. Chunk idxs are
            # NOT renumbered - recorded edges reference them - so the gauge becomes the
            # first surviving local (graph root = chunks[0].idx below).
            topology = 'chain'
            manifest['resolved']['topology'] = 'chain (demoted: hub-edge gate)'
            chunks = locals_
            skeleton = None
    if topology == 'hub':
        for c in locals_:
            est, info = hub_info[c.idx]
            ok = _gate_ok(info['med_resid'], info['med_resid'], base, cfg.gate_scale)
            reason = 'residual'
            if ok and not _consensus_ok(info):
                ok, reason = False, info.get('state', 'unverified')
            _record('hub', 0, c.idx, est, info, 'pass' if ok else f'fail({reason})')
            if ok:
                edges.append((0, c.idx, est, info))
            else:
                print(f"    hub edge skel<-chunk{c.idx:03d} REJECTED [{reason}] "
                      f"cons={info.get('n_consensus')}/{info.get('n_frames')} "
                      f"rival={info.get('rival_frames')}")
    if topology == 'chain' and not rim_meds:               # demoted runs already have rims
        for a, b in zip(chunks, chunks[1:]):
            _add_rim(a.idx, b.idx, *align_shared(a, b))
    # loop verification baseline: the run's own healthy-alignment distribution (§4.5)
    seam_median = float(np.median(rim_meds)) if rim_meds else (
        float(np.median(hub_meds)) if hub_meds else 0.0)
    timings['edges_s'] = round(time.time() - t0, 1)

    # ── 4. orphans: a chunk with no surviving edge is dropped, never silently merged
    gauge_idx = chunks[0].idx              # 0 normally; first local after a demotion
    connected = _reachable_from_gauge([(i, j) for i, j, _e, _i2 in edges], root=gauge_idx)
    orphans = [c for c in chunks if c.idx not in connected]
    if orphans:
        print(f"  ORPHANS: {[c.idx for c in orphans]} have no accepted edge to the gauge "
              f"- excluded from merge and poses (--keep-orphans writes a sidecar PLY)")
    live_chunks = [c for c in chunks if c.idx in connected]

    # ── 5. loop closure (retained in every topology; in hub mode it is redundant
    #      verification rather than the sole long-range constraint, §4.4)
    frame_chunk = np.full(len(images), -1, int)
    frame_local = np.full(len(images), -1, int)
    owner = _owner_map(live_chunks, len(images))
    for c in live_chunks:
        for li, g in enumerate(_rows_of(c)):
            if owner[g] == c.idx:
                frame_chunk[g] = c.idx; frame_local[g] = li
    loop_edges = []
    enu = None
    gps_lla = read_exif_gps(images) if cfg.gps == 'auto' else None
    if gps_lla is not None:
        enu = gps_to_enu(gps_lla)
    if cfg.loop == 'auto' and len(live_chunks) > 2:
        t0 = time.time()
        if Z is None:
            # chain + dedup off runs no corpus encode, so harvest descriptors from the
            # chunk runs the way v1 did (one row per owned frame) - without this, loop
            # closure would silently disable itself and v1 would quietly regress.
            raw_rows, have = np.zeros((len(images), 768), np.float32), []
            for c in live_chunks:
                d = c.desc()
                for li, g in enumerate(_rows_of(c)):
                    if owner[g] == c.idx:
                        raw_rows[g] = d[li]; have.append(g)
            have = sorted(have)
            Z = prep_descriptors(raw_rows[have])
            kept_rows = have
        Zf = np.zeros((len(images), Z.shape[1]), np.float32)
        for i, g in enumerate(kept_rows):
            Zf[g] = Z[i]
        rows = np.where(frame_chunk >= 0)[0]
        sim_th = None if cfg.loop_sim == 'auto' else float(cfg.loop_sim)
        cand_local = loop_candidates(Zf[rows], frame_chunk[rows], sim_th,
                                     t_excl=cfg.overlap, unordered=(order == 'unordered'))
        cand = [(int(rows[a]), int(rows[b]), s) for a, b, s in cand_local]
        if enu is not None:
            for p in gps_candidates(enu, frame_chunk, unordered=(order == 'unordered')):
                if frame_chunk[p[0]] >= 0 and frame_chunk[p[1]] >= 0 and not any(
                        abs(p[0] - a) <= 25 and abs(p[1] - b) <= 25 for a, b, _ in cand):
                    cand.append(p)
        print(f"  loop candidates: {len(cand)}")
        for pair in cand:
            e = build_loop_edge(pair, {c.idx: c for c in live_chunks}, frame_chunk, frame_local,
                                cfg, seam_median, Z=(Zf if order == 'unordered' else None),
                                exe_sha=exe_sha)
            if e:
                loop_edges.append(e)
                e[3]['med_resid'] = max(e[3]['med_a'], e[3]['med_b'])   # gate/weight scalar
                e[3]['n'] = 0
                _record('loop', e[0], e[1], e[2], e[3], 'pass')
                print(f"  loop edge chunk{e[0]}<->chunk{e[1]} accepted "
                      f"(med {e[3]['med_a']:.5f}/{e[3]['med_b']:.5f})")
            else:
                print(f"  loop pair {pair[:2]} rejected by gate")
        timings['loops_s'] = round(time.time() - t0, 1)

    # ── 6. pose graph + POST-GRAPH GHOST VALIDATION (leave-one-out reject, re-solve)
    #      Every gate so far ran in chunk-local coordinates BEFORE the graph moved anything;
    #      this loop is the first stage that looks at the PRODUCT. Edges whose realized
    #      placement puts the same physical pixels further apart than --ghost-max of scene
    #      extent mark the graph as contaminated. Ghost magnitude alone cannot NAME the
    #      culprit - the joint solve dilutes a corrupt edge into its neighbours, and
    #      measurably the victims can ghost harder than the culprit - so the culprit is
    #      identified by leave-one-out: dropping it heals the rest; dropping a victim does
    #      not. Chunks disconnected by a rejection become orphans: a partial map with
    #      explicit components beats one ghosted conglomeration [2026-07-19;
    #      measured: gate-passing edges ghosting up to 228% of extent].
    t0 = time.time()
    ghost_summary = {'rounds': 0, 'rejected': 0, 'max_allowed': cfg.ghost_max}

    def _solve_round(es=None, ls=None, annotate=True):
        es = edges if es is None else es
        ls = loop_edges if ls is None else ls
        node = {c.idx: n for n, c in enumerate(live_chunks)}
        g_edges = [t for t in es if t[0] in node and t[1] in node]
        g_loops = [t for t in ls if t[0] in node and t[1] in node]
        seq_e = [(node[i], node[j], e) for i, j, e, _ in g_edges]
        seq_w = [edge_weight(inf['med_resid'], seam_median) for _, _, _, inf in g_edges]
        lp_e = [(node[i], node[j], e, inf) for i, j, e, inf in g_loops]
        loop_w = [edge_weight(max(inf['med_a'], inf['med_b']), seam_median)
                  for _, _, _, inf in g_loops]
        gps_term = None
        if enu is not None:
            m_local = np.full((len(live_chunks), 3), np.nan)
            enu_mean = np.full((len(live_chunks), 3), np.nan)
            for c in live_chunks:
                po = c.poses()
                own_li = [li for li, g in enumerate(_rows_of(c)) if owner[g] == c.idx]
                if not own_li:
                    continue
                m_local[node[c.idx]] = np.mean([po[li]['c2w'][:3, 3] for li in own_li], axis=0)
                rws = enu[[_rows_of(c)[li] for li in own_li]]
                if np.isfinite(rws).all(axis=1).any():
                    enu_mean[node[c.idx]] = np.nanmean(rws, axis=0)
            gps_term = (m_local, enu_mean, 0.1)
        S_nodes = optimize_pose_graph(len(live_chunks), seq_e, lp_e, gps_term,
                                      seq_w=seq_w, loop_w=loop_w)
        if gps_term is not None:
            m_local, enu_mean, _ = gps_term
            ok = np.isfinite(enu_mean).all(axis=1) & np.isfinite(m_local).all(axis=1)
            if ok.sum() >= 3:
                src = np.stack([sim3_apply(S_nodes[k], m_local[k][None])[0]
                                for k in np.where(ok)[0]])
                sv = np.linalg.svd(src - src.mean(0), compute_uv=False)
                if sv[1] < 0.05 * sv[0]:
                    if annotate:
                        print("  ENU alignment SKIPPED: anchors near-collinear (straight "
                              "flight pass) - rotation about the line would be "
                              "unconstrained; relative geometry kept")
                else:
                    G = weighted_umeyama(src, enu_mean[ok], np.ones(int(ok.sum())))
                    S_nodes = [sim3_compose(G, S) for S in S_nodes]
                    if annotate:
                        print(f"  ENU alignment: global scale {G[0]:.4f} over "
                              f"{int(ok.sum())} anchors")
        S_abs = {c.idx: S_nodes[node[c.idx]] for c in live_chunks}
        all_e = g_edges + g_loops
        ghosts, ext = edge_ghosts(all_e, S_abs)
        if annotate:
            for (_i2, _j2, _e2, inf), gv in zip(all_e, ghosts):
                if '_rec' in inf:
                    inf['_rec']['ghost'] = None if gv is None else round(gv, 4)
        return S_abs, all_e, ghosts, ext, len(seq_e), len(lp_e)

    MAX_REJECTS = 4
    while True:
        S_abs, all_e, ghosts, g_extent, n_seq, n_lp = _solve_round()
        ghost_summary['rounds'] += 1
        offenders = ([] if cfg.ghost_max <= 0 else
                     [k for k, gv in enumerate(ghosts) if gv is not None and gv > cfg.ghost_max])
        if not offenders:
            break
        if ghost_summary['rejected'] >= MAX_REJECTS:
            # Endgame: one-at-a-time LOO stopped converging, so the contradiction is not a
            # single culprit. Do NOT ship a merge whose surviving edges still disagree
            # (measured before this branch existed: 22M points at 36% median ghost) -
            # hard-drop every remaining offender and let disconnected chunks fall off as
            # orphans. A partial map with explicit components beats one ghosted
            # conglomeration [2026-07-19].
            print(f"  GHOST ENDGAME: {len(offenders)} edge(s) still over --ghost-max after "
                  f"{ghost_summary['rejected']} LOO rejections - dropping all of them; "
                  f"disconnected chunks become orphans")
            drop = set()
            for k in offenders:
                inf_k = all_e[k][3]
                drop.add(id(inf_k))
                if '_rec' in inf_k:
                    inf_k['_rec']['gate'] = f'fail(ghost {100 * ghosts[k]:.0f}%)'
            ghost_summary['rejected'] += len(offenders)
            edges = [t for t in edges if id(t[3]) not in drop]
            loop_edges = [t for t in loop_edges if id(t[3]) not in drop]
            connected = _reachable_from_gauge([(t[0], t[1]) for t in edges]
                                              + [(t[0], t[1]) for t in loop_edges],
                                              root=gauge_idx)
            newly = [c for c in live_chunks if c.idx not in connected]
            if newly:
                print(f"  ORPHANED by ghost endgame: {[c.idx for c in newly]}")
                orphans += newly
                live_chunks = [c for c in live_chunks if c.idx in connected]
                owner = _owner_map(live_chunks, len(images))
            S_abs, all_e, ghosts, g_extent, n_seq, n_lp = _solve_round()
            break
        # leave-one-out over the worst few offenders: probe-solve without each, score by the
        # max remaining ghost. Candidates whose removal would disconnect a chunk from the
        # gauge are skipped (orphaning is a last resort, not a probe side effect).
        cands = sorted(offenders, key=lambda kk: -ghosts[kk])[:3]
        best = None
        for k in cands:
            inf_k = all_e[k][3]
            es2 = [t for t in edges if t[3] is not inf_k]
            ls2 = [t for t in loop_edges if t[3] is not inf_k]
            reach = _reachable_from_gauge([(t[0], t[1]) for t in es2]
                                          + [(t[0], t[1]) for t in ls2], root=gauge_idx)
            if any(c.idx not in reach for c in live_chunks):
                continue
            _S2, _ae2, gh2, _x2, _a2, _b2 = _solve_round(es2, ls2, annotate=False)
            score = max((gv for gv in gh2 if gv is not None), default=0.0)
            # tie-break by the candidate's own HELD-OUT ghost (cross-validation: how badly
            # does the rest of the graph contradict this edge?). Needed because a single
            # remaining cycle cannot localize blame - every removal of its edges "heals" -
            # and the held-out disagreement is the only per-edge evidence left.
            held = edge_ghosts([all_e[k]], _S2)[0][0] or 0.0
            key2 = (round(score / 0.02), -held)
            if best is None or key2 < best[1]:
                best = (k, key2, score)
        k = best[0] if best is not None else max(offenders, key=lambda kk: ghosts[kk])
        i, j, _e, inf = all_e[k]
        if '_rec' in inf:
            inf['_rec']['gate'] = f'fail(ghost {100 * ghosts[k]:.0f}%)'
        heal = (f"healing the rest to {100 * best[2]:.1f}% max" if best is not None
                else "argmax fallback (every candidate would disconnect a chunk)")
        print(f"  GHOST REJECT {(inf['_rec']['type'] if '_rec' in inf else '?')} "
              f"{i}<->{j}: ghost {100 * ghosts[k]:.0f}% of extent post-graph; "
              f"leave-one-out {heal} - dropped, re-solving")
        ghost_summary['rejected'] += 1
        edges = [t for t in edges if t[3] is not inf]
        loop_edges = [t for t in loop_edges if t[3] is not inf]
        connected = _reachable_from_gauge([(t[0], t[1]) for t in edges]
                                          + [(t[0], t[1]) for t in loop_edges], root=gauge_idx)
        newly = [c for c in live_chunks if c.idx not in connected]
        if newly:
            print(f"  ORPHANED by ghost validation: {[c.idx for c in newly]}")
            orphans += newly
            live_chunks = [c for c in live_chunks if c.idx in connected]
            owner = _owner_map(live_chunks, len(images))
    gvals = [gv for gv in ghosts if gv is not None]
    if gvals:
        ghost_summary.update(extent=round(g_extent, 4),
                             median=round(float(np.median(gvals)), 4),
                             p90=round(float(np.percentile(gvals, 90)), 4),
                             max=round(float(max(gvals)), 4))
        print(f"  ghost check: med={100 * np.median(gvals):.1f}% "
              f"p90={100 * np.percentile(gvals, 90):.1f}% max={100 * max(gvals):.1f}% "
              f"of extent over {len(gvals)} edges | {ghost_summary['rejected']} rejected "
              f"in {ghost_summary['rounds']} round(s)")
    scales = np.array([S_abs[c.idx][0] for c in live_chunks])
    spread = float(scales.max() / max(scales.min(), 1e-12))
    print(f"  pose graph: {n_seq} rigid + {n_lp} loop edges | scale spread "
          f"[{scales.min():.3f}, {scales.max():.3f}] median {np.median(scales):.3f} "
          f"(ratio {spread:.2f}x)")
    timings['graph_s'] = round(time.time() - t0, 1)

    # ── 7. streaming merge (two passes: count, then write with exact global stride)
    t0 = time.time()
    merged = cfg.out / 'merged'; merged.mkdir(parents=True, exist_ok=True)
    total = sum(len(_owned_points(c, owner, cfg.conf_keep_coef)[0]) for c in live_chunks)
    step = max(1, -(-total // cfg.max_points)) if cfg.max_points > 0 else 1
    body = merged / f'scene.body.{os.getpid()}.tmp'   # per-PID: concurrent runs can't collide
    n_out, carry = 0, 0
    with open(body, 'wb') as bf:
        for c in live_chunks:
            p, r = _owned_points(c, owner, cfg.conf_keep_coef)
            p = sim3_apply(S_abs[c.idx], p).astype(np.float32)
            sel = np.arange((step - carry) % step, len(p), step)
            carry = (carry + len(p)) % step
            p, r = p[sel], r[sel]
            bf.write(_ply_records(p, r)); n_out += len(p)
            print(f"  merge {c.role} {c.idx:03d}: +{len(p):,} pts")
    f, scene_path = _open_overwrite(merged / 'scene.ply')
    with f:
        f.write(PLY_HEADER.format(n=n_out).encode())
        with open(body, 'rb') as bf:
            shutil.copyfileobj(bf, f, 16 << 20)
    body.unlink()
    print(f"  merged: {n_out:,} pts (of {total:,}; stride {step}) -> {scene_path}")
    if orphans and cfg.keep_orphans:
        f, orph_path = _open_overwrite(merged / 'orphans.ply')
        with f:
            recs, n_orph = [], 0
            for c in orphans:                              # untransformed, for inspection only
                p, r = _owned_points(c, _owner_map(orphans, len(images)), cfg.conf_keep_coef)
                recs.append(_ply_records(p, r)); n_orph += len(p)
            f.write(PLY_HEADER.format(n=n_orph).encode())
            for rec in recs:
                f.write(rec)
        print(f"  orphans: {n_orph:,} pts (UNTRANSFORMED) -> {orph_path}")
    timings['merge_s'] = round(time.time() - t0, 1)

    # ── 8. per-frame poses from the owning chunk
    frames, kept = [], []
    for g, img in enumerate(images):
        if owner[g] < 0:
            continue
        kept.append(Path(img).resolve())
        c = next(c for c in live_chunks if c.idx == owner[g])
        li = _rows_of(c).index(g)
        p = c.poses()[li]
        s, R, t = S_abs[c.idx]
        c2w = p['c2w'].copy()
        c2w[:3, 3] = s * (R @ c2w[:3, 3]) + t
        c2w[:3, :3] = R @ c2w[:3, :3]
        frames.append({'extrinsics_c2w': c2w.flatten().tolist(),
                       'intrinsics': p['K'].flatten().tolist(),
                       'filename': img.name})
    (merged / 'poses.json').write_text(json.dumps({'frames': frames}, indent=1))
    # Orphan/dedup frames are absent from poses.json, and postprocess pairs images
    # to poses BY POSITION (hard error on a count mismatch) - so hand it exactly the
    # kept frames, in pose order, rather than the original input.
    (merged / 'images.txt').write_text(''.join(f"{q}\n" for q in kept), encoding='utf-8')

    # ── 9. manifest + report (machine channel; human logs stay, §4.7)
    live_ids = {c.idx for c in live_chunks}
    for c in chunks:
        manifest['chunks'].append({'id': c.idx, 'key': c.key, 'role': c.role, 'k': c.k,
                                   'size': len(c.files), 'bridges': int(sum(c.bridge_mask)),
                                   'orphan': c.idx not in live_ids,
                                   'files': [str(f) for f in c.files]})
    for g, rec in enumerate(manifest['frames']):
        if rec['disposition'] == 'kept':
            rec['owner_chunk'] = int(owner[g]) if owner[g] >= 0 else None
            if owner[g] < 0:
                rec['disposition'] = 'orphaned'
        rec['bridge_into'] = [c.idx for c in chunks
                              if g in _rows_of(c) and c.bridge_mask[_rows_of(c).index(g)]]
    manifest['edges'] = edge_recs
    manifest['graph'] = {'gauge': ('skeleton' if topology == 'hub'
                                   else f'chunk_{live_chunks[0].idx if live_chunks else 0}'),
                         'per_chunk_sRt': {str(c.idx): {'s': float(S_abs[c.idx][0]),
                                                        'R': S_abs[c.idx][1].tolist(),
                                                        't': S_abs[c.idx][2].tolist()}
                                           for c in live_chunks},
                         'scale_spread': [float(scales.min()), float(scales.max())],
                         'scale_ratio': spread, 'ghost': ghost_summary,
                         'iters': getattr(optimize_pose_graph, 'last_nfev', None),
                         'merged_points': int(n_out)}
    timings['total_s'] = round(time.time() - t_start, 1)
    manifest['timings'] = timings
    (cfg.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))

    (merged / 'report.txt').write_text(
        "dvlt_long v1.5 run summary\n"
        f"images {len(images)}  chunks {len(chunks)} (orphans {len(orphans)})  "
        f"profile={cfg.profile} topology={topology} order={order} K={cfg.loop_steps}\n"
        + "".join(f"{e['type']} {e['i']}->{e['j']}: s={e['s']:.4f} med={e['med_resid']} "
                  f"n={e['n']} {e['gate']}\n" for e in edge_recs)
        + f"loop edges accepted {n_lp}\n"
        + f"scale spread [{scales.min():.3f}, {scales.max():.3f}] ratio {spread:.2f}x\n"
        + f"ghost: {ghost_summary}\n"
        + f"merged points {n_out:,} of {total:,} (stride {step})\n"
        + f"timings {timings}\n", encoding='utf-8')

    # ── 10. postprocess
    if not cfg.no_postprocess:
        subprocess.run([sys.executable, str(Path(__file__).parent / 'dvlt_postprocess.py'),
                        str(merged), '--images', str(merged / 'images.txt'),
                        '--formats', cfg.formats], check=True)

    # ── 11. reclaim the intermediates (opt-in, see --drop-chunks). Still refuses whenever the
    #        run left something to investigate - that is exactly when the chunks are evidence.
    if cfg.drop_chunks:
        hold = []
        if orphans:
            hold.append(f"{len(orphans)} orphan chunk(s)")
        if n_out == 0:
            hold.append("merge produced no points")
        if any(e['gate'] != 'pass' for e in edge_recs):
            hold.append(f"{sum(1 for e in edge_recs if e['gate'] != 'pass')} rejected edge(s)")
        if hold:
            print(f"  chunks KEPT for inspection ({'; '.join(hold)}) - "
                  f"delete {cfg.out / 'chunks'} manually when done")
        else:
            freed = 0
            for d in (cfg.out / 'chunks', cfg.out / 'loops'):
                if not d.exists():
                    continue
                freed += sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
            print(f"  chunks removed, {freed / 2**30:.1f} GB reclaimed (omit --drop-chunks to retain)")
    print(f"dvlt_long: done in {timings['total_s']}s -> {cfg.out / 'manifest.json'}")

# ─── self tests ──────────────────────────────────────────────────────────────
class _FakeChunk:
    def __init__(self, g_start, pts, rgb, conf, poses):
        self.g_start, self._p, self._r, self._c, self._po = g_start, pts, rgb, conf, poses
        self.files = [Path(f"{g_start+i:04d}.jpg") for i in range(len(pts))]
    def pts_grid(self): return self._p, self._r, self._c
    def poses(self): return self._po

def test_align_adjacent_synthetic():
    rng = np.random.default_rng(4)
    S, H, W = 6, 20, 30
    world = rng.normal(0, 3, (S, H, W, 3)).astype(np.float32)   # shared "scene" for overlap frames
    gt = sim3_exp(_rand_sim3(rng))
    # chunk A sees frames 0..5 in its own frame == world; chunk B sees frames 2..7,
    # where B's coords = inv(gt) applied to world (so mapping B->A must recover gt)
    pts_b = np.concatenate([world[2:], rng.normal(0, 3, (2, H, W, 3)).astype(np.float32)])
    pts_b = sim3_apply(sim3_inv(gt), pts_b.reshape(-1, 3)).reshape(S, H, W, 3).astype(np.float32)
    conf = np.full((S, H, W), 3.0, np.float32)
    conf[:, :2, :] = 1.01                                        # a low-conf band (soft weighting only; hard-discard covered in the next test)
    rgb = np.zeros((S, H, W, 3), np.uint8)
    poses = [{'c2w': np.eye(4), 'K': np.eye(3)}] * S
    A = _FakeChunk(0, world, rgb, conf, poses)
    B = _FakeChunk(2, pts_b, rgb, conf, poses)
    est, info = align_adjacent(A, B)
    assert not info.get('fallback')
    assert abs(est[0] - gt[0]) / gt[0] < 1e-3
    ang = math.degrees(math.acos(np.clip((np.trace(est[1] @ gt[1].T) - 1) / 2, -1, 1)))
    assert ang < 0.1 and np.linalg.norm(est[2] - gt[2]) < 0.01

def test_align_adjacent_discard_and_fallback():
    rng = np.random.default_rng(9)
    S, H, W = 6, 20, 30
    world = rng.normal(0, 3, (S, H, W, 3)).astype(np.float32)
    gt = sim3_exp(_rand_sim3(rng))
    rgb = np.zeros((S, H, W, 3), np.uint8)
    eye_poses = [{'c2w': np.eye(4), 'K': np.eye(3)}] * S

    # (a) hard discard actually filters: band conf 0.02 < 0.1*median(3.0)
    pts_b = np.concatenate([world[2:], rng.normal(0, 3, (2, H, W, 3)).astype(np.float32)])
    pts_b = sim3_apply(sim3_inv(gt), pts_b.reshape(-1, 3)).reshape(S, H, W, 3).astype(np.float32)
    conf = np.full((S, H, W), 3.0, np.float32)
    conf[:, :2, :] = 0.02
    est, info = align_adjacent(_FakeChunk(0, world, rgb, conf, eye_poses),
                               _FakeChunk(2, pts_b, rgb, conf, eye_poses))
    full = 4 * H * W
    assert info['n'] < full - 100, f"hard discard filtered nothing: n={info['n']} of {full}"
    assert not info.get('fallback')
    assert abs(est[0] - gt[0]) / gt[0] < 1e-3

    # (b) fallback: poison pointmaps so <100 correspondences survive -> camera-center Umeyama
    world_nan = np.full_like(world, np.nan)
    world_nan[:, 0, :20, :] = world[:, 0, :20, :]          # 4 shared frames x 20 px = 80 < 100
    pts_b_nan = np.concatenate([world_nan[2:], world_nan[:2]])
    pts_b_nan = sim3_apply(sim3_inv(gt), pts_b_nan.reshape(-1, 3)).reshape(S, H, W, 3).astype(np.float32)
    conf2 = np.full((S, H, W), 3.0, np.float32)

    def c2w_from_center(c):
        m = np.eye(4); m[:3, 3] = c; return m
    cents = rng.normal(0, 2, (8, 3))                        # global frames 0..7, A/world coords
    posesA = [{'c2w': c2w_from_center(cents[g]), 'K': np.eye(3)} for g in range(6)]
    posesB = [{'c2w': c2w_from_center(sim3_apply(sim3_inv(gt), cents[g + 2][None])[0]),
               'K': np.eye(3)} for g in range(6)]
    est2, info2 = align_adjacent(_FakeChunk(0, world_nan, rgb, conf2, posesA),
                                 _FakeChunk(2, pts_b_nan, rgb, conf2, posesB))
    assert info2.get('fallback') is True
    assert abs(est2[0] - gt[0]) / gt[0] < 1e-6
    ang = math.degrees(math.acos(np.clip((np.trace(est2[1] @ gt[1].T) - 1) / 2, -1, 1)))
    assert ang < 1e-3 and np.linalg.norm(est2[2] - gt[2]) < 1e-6

def _rand_sim3(rng, max_ang=1.5, max_sig=0.8):
    v = np.concatenate([rng.normal(0, 1, 3), rng.normal(0, max_ang / 3, 3), [rng.uniform(-max_sig, max_sig)]])
    return v

def test_sim3_exp_matches_matrix_expm():
    from scipy.linalg import expm
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = _rand_sim3(rng)
        s, R, t = sim3_exp(v)
        X = np.zeros((4, 4)); X[:3, :3] = _hat(v[3:6]) + v[6] * np.eye(3); X[:3, 3] = v[:3]
        M = expm(X)
        assert np.allclose(s * R, M[:3, :3], atol=1e-8), (s * R, M[:3, :3])
        assert np.allclose(t, M[:3, 3], atol=1e-8)

def test_sim3_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = _rand_sim3(rng)
        assert np.allclose(sim3_log(sim3_exp(v)), v, atol=1e-7)

def test_umeyama_exact():
    rng = np.random.default_rng(2)
    src = rng.normal(0, 2, (500, 3))
    gt = sim3_exp(_rand_sim3(rng))
    s, R, t = weighted_umeyama(src, sim3_apply(gt, src), np.ones(500))
    assert abs(s - gt[0]) < 1e-9 and np.allclose(R, gt[1], atol=1e-9) and np.allclose(t, gt[2], atol=1e-9)

def test_irls_outliers():
    rng = np.random.default_rng(3)
    src = rng.normal(0, 2, (2000, 3))
    gt = sim3_exp(_rand_sim3(rng))
    tgt = sim3_apply(gt, src) + rng.normal(0, 0.002, (2000, 3))
    out = rng.random(2000) < 0.3                       # 30% gross outliers
    tgt[out] += rng.normal(0, 2.0, (out.sum(), 3))
    w0 = np.where(out, 0.3, 1.0)                       # confidence loosely correlated
    est, info = irls_sim3(src, tgt, w0)
    assert info['converged']
    assert abs(est[0] - gt[0]) / gt[0] < 1e-2
    ang = math.degrees(math.acos(np.clip((np.trace(est[1] @ gt[1].T) - 1) / 2, -1, 1)))
    assert ang < 0.5 and np.linalg.norm(est[2] - gt[2]) < 0.05

def test_ply_grid_roundtrip(tmp=None):
    import tempfile
    xyz = np.arange(36, dtype=np.float32).reshape(12, 3)
    rgb = np.arange(36, dtype=np.uint8).reshape(12, 3)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.ply"
        buf = bytearray()
        for i in range(12):
            buf += xyz[i].tobytes() + rgb[i].tobytes()
        p.write_bytes(("ply\nformat binary_little_endian 1.0\nelement vertex 12\n"
                       "property float x\nproperty float y\nproperty float z\n"
                       "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                       "end_header\n").encode() + bytes(buf))
        x2, r2 = read_ply_grid(p)
    assert np.array_equal(x2, xyz) and np.array_equal(r2, rgb)

def test_list_images_txt_preserves_order():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for n in ('b.jpg', 'a.jpg', 'c.jpg'):
            (td / n).write_bytes(b'x')
        # unsorted order + comment + blank + relative path: order must be kept
        # as written (dvlt.exe expand_list: line 1 = reference camera)
        lst = td / 'sel.txt'
        lst.write_text(f"# vfold selection\n{td / 'b.jpg'}\n\na.jpg\n{td / 'c.jpg'}\n")
        imgs = list_images(lst)
        assert [p.name for p in imgs] == ['b.jpg', 'a.jpg', 'c.jpg'], imgs
        assert all(p.exists() and p.is_absolute() for p in imgs)
        # a directory still sorts
        assert [p.name for p in list_images(td)] == ['a.jpg', 'b.jpg', 'c.jpg']
        # missing file fails fast with the line number
        lst.write_text("a.jpg\nmissing.jpg\n")
        try:
            list_images(lst)
            assert False, "missing list entry not rejected"
        except SystemExit:
            pass
        # a name outside the exe's code page is refused up front, by name - forced to a
        # cp1252 laptop's, as the dev box's ACP (65001) holds everything
        global ANSI
        (td / '写真.jpg').write_bytes(b'x')
        lst.write_text("写真.jpg\na.jpg\n", encoding='utf-8')
        ansi, ANSI = ANSI, 'cp1252'
        try:
            list_images(lst)
            assert False, "non-ANSI name not rejected"
        except SystemExit as e:
            assert '写真.jpg' in str(e), e
        finally:
            ANSI = ansi

def test_list_images_dir_sort_is_bytewise():
    # dvlt.cu expand_dir std::sorts path strings bytewise (case-SENSITIVE:
    # uppercase < lowercase), NOT Python's default case-insensitive WindowsPath
    # order. Frame pairing is positional, so the order must match the exe's.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for n in ('IMG_001.jpg', 'aaa.jpg', 'B.jpg', 'a.jpg'):
            (td / n).write_bytes(b'x')
        got = [p.name for p in list_images(td)]
        assert got == ['B.jpg', 'IMG_001.jpg', 'a.jpg', 'aaa.jpg'], got

def test_list_images_rejects_utf16():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'a.jpg').write_bytes(b'x')
        lst = td / 'sel.txt'
        lst.write_bytes('﻿a.jpg\n'.encode('utf-16-le'))  # PowerShell Out-File default
        try:
            list_images(lst)
            assert False, "UTF-16 list not rejected"
        except SystemExit as e:
            assert 'UTF-8' in str(e)

def test_list_is_abs_matches_exe():
    # mirrors dvlt.cu:71 bool abs = s[0]=='/'||s[0]=='\\'||(s.size()>1&&s[1]==':')
    for s in ('/data/x.jpg', '\\data\\x.jpg', 'D:\\x.jpg', 'D:x.jpg', '//srv/s/x.jpg'):
        assert _list_is_abs(s), s
    for s in ('x.jpg', 'sub/x.jpg', 'a.jpg'):
        assert not _list_is_abs(s), s

def test_make_chunks():
    imgs = [Path(f"{i:04d}.jpg") for i in range(723)]
    ch = make_chunks(imgs, 60, 30)
    assert ch[0].g_start == 0 and len(ch[0].files) == 60
    assert ch[1].g_start == 30
    assert ch[-1].g_start + len(ch[-1].files) == 723
    starts = [c.g_start for c in ch]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)

def test_make_chunks_rejects_bad_overlap():
    imgs = [Path(f"{i:04d}.jpg") for i in range(10)]
    for L, O in ((60, 60), (60, 90), (0, 0)):
        try:
            make_chunks(imgs, L, O)
            assert False, f"make_chunks accepted L={L} O={O}"
        except SystemExit:
            pass

def test_prep_descriptors():
    rng = np.random.default_rng(5)
    raw = rng.normal(0, 1, (40, 768)).astype(np.float32)
    Z = prep_descriptors(raw)
    assert Z.shape == (40, 38) or Z.shape[1] <= 512
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)

def test_loop_candidates():
    rng = np.random.default_rng(6)
    base = rng.normal(0, 1, (100, 768)).astype(np.float32)
    raw = base + rng.normal(0, 0.05, base.shape).astype(np.float32)
    raw[80:85] = raw[10:15] + rng.normal(0, 0.02, (5, 768)).astype(np.float32)  # revisit of frames 10-14
    Z = prep_descriptors(raw)
    fc = np.arange(100) // 30                       # 4 chunks
    cand = loop_candidates(Z, fc)
    assert cand, "revisit not detected"
    i, j, _ = cand[0]
    assert 8 <= i <= 16 and 78 <= j <= 86           # matches the planted revisit
    # NMS: no two kept pairs within +/-25 of each other on both endpoints
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            assert not (abs(cand[a][0] - cand[b][0]) <= 25 and abs(cand[a][1] - cand[b][1]) <= 25)

def test_prep_descriptors_duplicates_survive_whitening():
    rng = np.random.default_rng(11)
    raw = (rng.normal(0, 1, (100, 768)) + rng.normal(0, 0.05, (100, 768))).astype(np.float32)
    raw[80:85] = raw[10:15] + rng.normal(0, 0.02, (5, 768)).astype(np.float32)
    Z = prep_descriptors(raw)
    dup = float(np.mean([(Z[10 + k] @ Z[80 + k]) for k in range(5)]))
    assert dup > 0.9, f"whitening destroyed planted duplicates: {dup}"

def test_prep_descriptors_degenerate_inputs():
    Z1 = prep_descriptors(np.ones((1, 768), np.float32))
    Z2 = prep_descriptors(np.ones((4, 768), np.float32))   # identical rows -> zero spectrum
    assert np.isfinite(Z1).all() and np.isfinite(Z2).all()

def test_loop_candidates_auto_threshold_beats_hardcoded():
    rng = np.random.default_rng(12)
    v = rng.normal(0, 1, 64); v /= np.linalg.norm(v)
    rows = [v]
    for _ in range(199):                                   # smooth video-like descriptor walk
        v = v + 0.3 * rng.normal(0, 1, 64) / 8.0
        v /= np.linalg.norm(v)
        rows.append(v)
    Z = np.array(rows, np.float32)
    plant = Z[20] + 0.6 * rng.normal(0, 1, 64) / 8.0       # revisit at ~0.86 cosine
    Z[150] = (plant / np.linalg.norm(plant)).astype(np.float32)
    s_plant = float(Z[20] @ Z[150])
    assert 0.65 < s_plant < 0.88, f"construction drifted: {s_plant}"
    fc = np.arange(200) // 50
    hits_auto = [(i, j) for i, j, _ in loop_candidates(Z, fc)]
    assert any(abs(i - 20) <= 2 and abs(j - 150) <= 2 for i, j in hits_auto), hits_auto
    hits_088 = [(i, j) for i, j, _ in loop_candidates(Z, fc, sim_thresh=0.88)]
    assert not any(abs(i - 20) <= 2 and abs(j - 150) <= 2 for i, j in hits_088), hits_088

def test_gps_candidates():
    enu = np.zeros((200, 3)); enu[:, 0] = np.arange(200) * 2.0   # 2 m spacing straight line
    enu[150:160] = enu[20:30] + 1.0                              # returns to 40-60 m region
    fc = np.arange(200) // 60
    cand = gps_candidates(enu, fc)
    assert any(5 <= i <= 45 and 140 <= j <= 170 for i, j, _ in cand)

def test_loop_edge_composition():
    rng = np.random.default_rng(7)
    Sa = sim3_exp(_rand_sim3(rng))       # loop-chunk -> chunk A coords
    Sb = sim3_exp(_rand_sim3(rng))       # loop-chunk -> chunk B coords
    Sab = sim3_compose(Sa, sim3_inv(Sb))  # B -> A
    p = rng.normal(0, 1, (10, 3))
    via = sim3_apply(Sa, sim3_apply(sim3_inv(Sb), p))
    assert np.allclose(sim3_apply(Sab, p), via, atol=1e-9)

def test_gate_rejects_bad_alignment():
    assert _gate_ok(med_a=0.01, med_b=0.02, seam_median=0.01, scale=3.0)
    assert not _gate_ok(med_a=0.05, med_b=0.02, seam_median=0.01, scale=3.0)

def test_gate_rejects_nan_and_boundary():
    assert not _gate_ok(float('nan'), 0.01, 0.01)
    assert not _gate_ok(0.01, float('nan'), 0.01)   # 2nd-position NaN must not slip through
    assert not _gate_ok(0.03, 0.03, 0.01, scale=3.0)  # equality is rejected (strict <)
    assert _gate_ok(0.0299, 0.01, 0.01, scale=3.0)

def test_irls_degenerate_returns_nonconverged():
    est, info = irls_sim3(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    assert info['converged'] is False and info['med_resid'] == float('inf')
    est2, info2 = irls_sim3(np.ones((2, 3)), np.ones((2, 3)), np.ones(2))
    assert info2['converged'] is False

def test_irls_budget_exhaustion_is_usable():
    rng = np.random.default_rng(14)
    src = rng.normal(0, 2, (5000, 3))
    gt = sim3_exp(_rand_sim3(rng))
    tgt = sim3_apply(gt, src) + rng.normal(0, 0.02, (5000, 3))
    tgt[:50] += rng.normal(0, 0.5, (50, 3))    # genuine outliers: Huber reweighting changes the
    # refit, so the tolerance early-stop provably cannot fire within 1 iteration (verified:
    # dparam~2.5e-3) — this pins the budget-exhaustion-is-usable semantics against reverts
    est, info = irls_sim3(src, tgt, np.ones(5000), max_iters=1)
    assert info['converged'] is True and np.isfinite(info['med_resid'])
    assert abs(est[0] - gt[0]) / gt[0] < 0.02

def test_enu():
    lla = np.array([[-118.0, 34.0, 10.0], [-117.999, 34.0, 10.0]])[:, [1, 0, 2]]  # lat,lon,alt
    enu = gps_to_enu(lla)
    assert 80 < abs(enu[1, 0] - enu[0, 0]) < 105   # ~92 m per 0.001° lon at 34°N

def test_pose_graph_recovers_drift():
    rng = np.random.default_rng(8)
    K = 6
    gt = [ (1.0, np.eye(3), np.zeros(3)) ]
    for _ in range(K - 1):
        gt.append(sim3_compose(gt[-1], sim3_exp(_rand_sim3(rng, max_ang=0.4, max_sig=0.2))))
    seq = []
    for k in range(K - 1):
        rel = sim3_compose(sim3_inv(gt[k]), gt[k + 1])
        noise = sim3_exp(_rand_sim3(rng, max_ang=0.02, max_sig=0.05))  # scale-drifting noise
        seq.append((k, k + 1, sim3_compose(rel, noise)))
    loop_rel = sim3_compose(sim3_inv(gt[0]), gt[K - 1])                 # perfect loop 0<->K-1
    est = optimize_pose_graph(K, seq, [(0, K - 1, loop_rel, {})])
    err = sim3_log(sim3_compose(sim3_inv(est[K - 1]), gt[K - 1]))
    S = (1.0, np.eye(3), np.zeros(3)); chain = [S]
    for k, _, rel in seq: chain.append(sim3_compose(chain[-1], rel))
    chained_err = sim3_log(sim3_compose(sim3_inv(chain[K - 1]), gt[K - 1]))
    assert np.linalg.norm(err) < 0.5 * np.linalg.norm(chained_err), (err, chained_err)

def test_pose_graph_weights_downweight_bad_seam():
    # one corrupted seam + one perfect loop: residual-derived weights must
    # push the error into the bad seam instead of averaging it into the chain
    rng = np.random.default_rng(15)
    K = 4
    gt = [(1.0, np.eye(3), np.zeros(3))]
    for _ in range(K - 1):
        gt.append(sim3_compose(gt[-1], sim3_exp(_rand_sim3(rng, max_ang=0.4, max_sig=0.2))))
    seq = []
    for k in range(K - 1):
        rel = sim3_compose(sim3_inv(gt[k]), gt[k + 1])
        if k == 1:                                     # corrupt seam 1->2
            rel = sim3_compose(rel, sim3_exp(_rand_sim3(rng, max_ang=0.6, max_sig=0.4)))
        seq.append((k, k + 1, rel))
    loop = [(0, K - 1, sim3_compose(sim3_inv(gt[0]), gt[K - 1]), {})]
    w_bad = edge_weight(0.5, 0.05)                     # measured-bad seam
    assert w_bad == 0.1 and edge_weight(0.01, 0.05) == 1.0
    est_u = optimize_pose_graph(K, seq, loop)
    est_w = optimize_pose_graph(K, seq, loop, seq_w=[1.0, w_bad, 1.0], loop_w=[1.0])
    err_u = np.linalg.norm(sim3_log(sim3_compose(sim3_inv(est_u[K - 1]), gt[K - 1])))
    err_w = np.linalg.norm(sim3_log(sim3_compose(sim3_inv(est_w[K - 1]), gt[K - 1])))
    assert err_w < 0.5 * err_u, (err_w, err_u)

def test_pose_graph_gps_stays_in_gauge():
    # m_local is CHUNK-LOCAL (anchor at each chunk's origin); true chunk
    # positions trace a non-collinear L. ENU anchors are 10x the visual scale:
    # the graph must NOT stretch chunks toward metric - that tears the map.
    # Metric comes from main()'s final refit, emulated here.
    t_true = np.array([[0., 0, 0], [10., 0, 0], [10., 10., 0], [10., 20., 0]])
    n = len(t_true)
    m_local = np.zeros((n, 3))
    enu = t_true * 10.0
    seq = [(k, k + 1, (1.0, np.eye(3), t_true[k + 1] - t_true[k]))
           for k in range(n - 1)]
    S = optimize_pose_graph(n, seq, [], gps=(m_local, enu, 1.0))
    scales = np.array([s for s, _, _ in S])
    assert (np.abs(np.log(scales)) < 0.2).all(), f"graph left the gauge: {scales}"
    src = np.stack([sim3_apply(S[k], m_local[k][None])[0] for k in range(n)])
    G = weighted_umeyama(src, enu, np.ones(n))
    assert np.allclose(sim3_apply(G, src), enu, atol=0.5)

def test_pose_graph_gps_corrects_drift():
    # One drifting seam among five chunks: gauge-mapped GPS must pull the tail
    # back. A GPS-less chunk (NaN row) must be tolerated. (Three anchors with
    # one outlier would be degenerate for the IRLS pre-fit - two clean points
    # cannot pin a Sim(3); a real flight has an anchor per chunk.)
    t_true = np.array([[0., 0, 0], [10., 0, 0], [20., 0, 0],
                       [20., 10., 0], [20., 20., 0], [20., 30., 0]])
    n = len(t_true)
    m_local = np.zeros((n, 3))
    enu = t_true.copy()                      # gauge == metric here
    enu[1] = np.nan                          # GPS-less chunk tolerated
    rels = [t_true[k + 1] - t_true[k] for k in range(n - 1)]
    rels[4] = rels[4] + np.array([3., 0, 0])            # 3 m drift on last seam
    seq = [(k, k + 1, (1.0, np.eye(3), rels[k])) for k in range(n - 1)]
    S = optimize_pose_graph(n, seq, [], gps=(m_local, enu, 2.0))
    got = sim3_apply(S[n - 1], m_local[n - 1][None])[0]
    assert np.linalg.norm(got - enu[n - 1]) < 1.5, got

def test_pose_graph_rejects_bad_edge_scale():
    for bad in (0.0, -1.0, float('nan')):
        seq = [(0, 1, (bad, np.eye(3), np.zeros(3)))]
        try:
            optimize_pose_graph(2, seq, [])
            assert False, f"accepted edge scale {bad}"
        except SystemExit:
            pass

# ─── v1.5 self tests (addendum §8) ───────────────────────────────────────────
def test_hub_init_contains_bad_edge():
    """§8.1 star-vs-chain containment. One corrupted edge: chain init multiplies it into
    every downstream chunk (the measured 29x cascade); hub init reaches every chunk
    in one hop, so exactly one node is wrong. This is the whole argument for v1.5."""
    rng = np.random.default_rng(20)
    K, BAD = 8, 3
    gt = [(1.0, np.eye(3), np.zeros(3))]
    for _ in range(K - 1):
        gt.append(sim3_compose(gt[-1], sim3_exp(_rand_sim3(rng, max_ang=0.3, max_sig=0.15))))
    corrupt = sim3_exp(np.array([0., 0., 0., 0., 0., 0., math.log(2.0)]))   # 2x scale error
    chain_edges, hub_edges = [], []
    for k in range(K - 1):
        rel = sim3_compose(sim3_inv(gt[k]), gt[k + 1])
        chain_edges.append((k, k + 1, sim3_compose(rel, corrupt) if k + 1 == BAD else rel))
    for k in range(1, K):
        rel = sim3_compose(sim3_inv(gt[0]), gt[k])
        hub_edges.append((0, k, sim3_compose(rel, corrupt) if k == BAD else rel))

    def bad_nodes(S):
        return [k for k in range(K) if abs(S[k][0] / gt[k][0] - 1.0) > 0.1]
    chain_bad = bad_nodes(_init_from_edges(K, chain_edges)[0])
    hub_bad = bad_nodes(_init_from_edges(K, hub_edges)[0])
    assert hub_bad == [BAD], f"hub must confine to the bad spoke, got {hub_bad}"
    assert set(range(BAD, K)) <= set(chain_bad), f"chain must compound downstream, got {chain_bad}"
    assert len(chain_bad) > len(hub_bad)
    # and the containment survives the optimizer: hub's scale spread stays bounded
    S_hub = optimize_pose_graph(K, [(i, j, m) for i, j, m in hub_edges], [])
    sc = np.array([s for s, _, _ in S_hub]) / np.array([g[0] for g in gt])
    assert sc.max() / sc.min() < 2.5, f"hub post-graph scale ratio {sc.max()/sc.min():.2f}"

def test_shared_frames_identity():
    """§8.2 identity-based discovery: serial overlap reproduces v1 behaviour, injected
    bridges (non-contiguous global identity) are found, and the run is fed the originals
    themselves (list file, absolute, in Chunk.files order)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        imgs = []
        for i in range(10):
            p = root / f"IMG_{i:03d}.jpg"; p.write_bytes(b"x" * (i + 1)); imgs.append(p)
        # (a) serial overlap: windows [0,6) and [3,9) share frames 3,4,5
        A = Chunk(0, 0, imgs[0:6], ""); A.g_rows = list(range(0, 6))
        B = Chunk(1, 3, imgs[3:9], ""); B.g_rows = list(range(3, 9))
        assert shared_frame_indices(A, B) == [(3, 0), (4, 1), (5, 2)]
        # (b) injected bridges: non-contiguous identities (7, 0, 9) staged into a chunk
        skel = Chunk(0, 0, [imgs[0], imgs[7], imgs[9]], "", role='skeleton')
        skel.g_rows = [0, 7, 9]
        files, mask, rows = build_chunk_files(imgs, [2, 3, 4], [7, 0, 9])
        loc = Chunk(1, 2, files, ""); loc.g_rows, loc.bridge_mask = rows, mask
        assert mask[0] is True and rows[0] == 7, "best bridge must occupy frame 0 (§4.3)"
        pairs = shared_frame_indices(skel, loc)
        assert sorted(rows[lj] for _, lj in pairs) == [0, 7, 9], pairs
        # (c) the exe reads exactly Chunk.files: line 1 = the frame-0 bridge, every line
        #     absolute (it runs with cwd = its own folder)
        lines = _write_list(loc.files, root / 'frames.txt').read_text(ANSI).splitlines()
        assert lines == [_ident(f) for f in loc.files] and lines[0] == _ident(imgs[7])
        assert all(_list_is_abs(s) for s in lines)

def test_dedup_rejects_whitened_space():
    """§8.3 two-space guard [INVARIANT D5]. Whitening is corpus-relative contrast
    enhancement (adjacent frames measure 0.18-0.39 there), so an absolute 0.95 threshold
    deletes nothing; dedup must never receive whitened vectors."""
    rng = np.random.default_rng(21)
    raw = rng.normal(0, 1, (30, 768)).astype(np.float32)
    Z = prep_descriptors(raw)
    assert Z.shape[1] != 768
    try:
        dedup_frames(Z)
        assert False, "dedup accepted whitened vectors"
    except ValueError as e:
        assert 'raw' in str(e).lower()

def test_dedup_synthetic():
    """§8.4 planted near-duplicates removed, distinct-but-similar survivors kept."""
    rng = np.random.default_rng(22)
    base = rng.normal(0, 1, (20, 768)).astype(np.float32)
    raw = np.repeat(base, 2, axis=0)                       # each frame duplicated
    raw[1::2] += rng.normal(0, 1e-3, (20, 768)).astype(np.float32)
    kept, dropped, clamped = dedup_frames(raw, sim_thresh=0.95)
    assert not clamped and len(kept) == 20, (len(kept), clamped)
    assert all(k % 2 == 0 for k in kept), kept
    assert all(dropped[d][0] == d - 1 for d in dropped)
    P = power_normalize(base)
    off = float(np.max(P @ P.T - np.eye(20)))
    assert off < 0.95, f"distinct rows are not distinct enough to test ({off:.3f})"

def test_dedup_min_keep_clamp():
    """§8.4 the min-keep guard bounds the damage when a corpus aliases hard (repetitive
    industrial structure)."""
    rng = np.random.default_rng(23)
    raw = np.tile(rng.normal(0, 1, (1, 768)), (10, 1)).astype(np.float32)
    raw += rng.normal(0, 1e-4, raw.shape).astype(np.float32)   # all near-identical
    kept, dropped, clamped = dedup_frames(raw, sim_thresh=0.95, min_keep=0.5)
    assert clamped is True and len(kept) >= 5, (len(kept), clamped)
    assert len(kept) + len(dropped) == 10

def test_bridge_copies_never_own():
    """§4.3 [INVARIANT] merge/ownership: an injected bridge at frame 0 is maximally
    central, so without the mask it would outrank the skeleton and double-emit."""
    imgs = [Path(f"{i:04d}.jpg") for i in range(12)]
    skel = Chunk(0, 0, [imgs[0], imgs[6]], "", role='skeleton'); skel.g_rows = [0, 6]
    files, mask, rows = build_chunk_files(imgs, [2, 3, 4], [6, 0])
    loc = Chunk(1, 2, files, ""); loc.g_rows, loc.bridge_mask = rows, mask
    owner = _owner_map([skel, loc], 12)
    assert owner[6] == 0 and owner[0] == 0, "skeleton must own its frames"
    assert all(owner[g] == 1 for g in (2, 3, 4))
    assert (owner[[1, 5, 7, 11]] == -1).all(), "unstaged frames must stay unowned"

def test_orphan_exclusion():
    """§8.5 a chunk whose every edge failed the gate is excluded from merge and poses -
    silently merging an unplaced chunk is the one failure worse than dropping it."""
    imgs = [Path(f"{i:04d}.jpg") for i in range(9)]
    cs = []
    for k in range(3):
        c = Chunk(k, 3 * k, imgs[3 * k:3 * k + 3], "", role='skeleton' if k == 0 else 'local')
        c.g_rows = list(range(3 * k, 3 * k + 3)); cs.append(c)
    edges = [(0, 1, (1.0, np.eye(3), np.zeros(3)))]          # chunk 2's hub edge failed the gate
    connected = {0}
    q = deque([0])
    adj = {}
    for i, j, _ in edges:
        adj.setdefault(i, set()).add(j); adj.setdefault(j, set()).add(i)
    while q:
        for v in adj.get(q.popleft(), ()):
            if v not in connected:
                connected.add(v); q.append(v)
    assert connected == {0, 1}
    live = [c for c in cs if c.idx in connected]
    owner = _owner_map(live, 9)
    assert (owner[6:9] == -1).all(), "orphan frames must not be owned/emitted"
    assert set(owner[0:6]) == {0, 1}
    S, unreached = _init_from_edges(3, edges)
    assert unreached == [2]

def test_chunk_key_separates_k_and_content():
    """§4.7/D8: the live/final split makes K-poisoning concrete - a K=8 live chunk must
    never satisfy a K=16 final pass, and bridge-modified frame lists must not collide."""
    f = [Path(f"{i}.jpg") for i in range(4)]
    a = chunk_key(f, 504, 8, 'sha')
    assert a != chunk_key(f, 504, 16, 'sha')
    assert a != chunk_key(f, 448, 8, 'sha')
    assert a != chunk_key(f, 504, 8, 'other')
    assert a != chunk_key(f[::-1], 504, 8, 'sha'), "order is part of the content address"
    assert a != chunk_key(f + [Path('bridge.jpg')], 504, 8, 'sha')
    assert a == chunk_key(list(f), 504, 8, 'sha') and len(a) == 12

def test_unordered_assignment_and_bridges():
    """§4.5 retrieval assignment respects the frame cap and always yields a bridge, so
    every chunk has a hub edge by construction."""
    rng = np.random.default_rng(24)
    cent = rng.normal(0, 1, (4, 32))
    rows = np.concatenate([cent[i] + rng.normal(0, 0.05, (25, 32)) for i in range(4)])
    Z = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
    skel = fps_order(Z)[:6]
    groups = assign_unordered_chunks(Z, skel, cap=20)
    assert groups and all(len(m) <= 20 for _, m in groups)
    assigned = sorted(i for _, m in groups for i in m)
    assert assigned == sorted(set(range(100)) - set(skel)), "every non-anchor frame assigned once"
    anchor, mem = groups[0]
    picks = select_bridges(Z, mem, skel, 3, must_include=anchor)
    assert anchor in picks and 2 <= len(picks) <= 4 and len(set(picks)) == len(picks)

def test_loop_candidates_unordered_uses_chunk_null():
    """§4.5 every serial-index rule replaced by its chunk-based analogue: the AUTO-tau
    null comes from cross-chunk pairs and NMS keeps one pair per chunk-pair."""
    rng = np.random.default_rng(25)
    rows = rng.normal(0, 1, (60, 48))
    rows[40:45] = rows[5:10] + rng.normal(0, 0.02, (5, 48))    # revisit, no temporal meaning
    Z = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
    fc = np.arange(60) // 15
    cand = loop_candidates(Z, fc, unordered=True)
    assert cand, "cross-chunk revisit not detected"
    seen = [(min(fc[a], fc[b]), max(fc[a], fc[b])) for a, b, _ in cand]
    assert len(seen) == len(set(seen)), f"NMS must keep one pair per chunk-pair: {seen}"
    assert any(fc[a] == 0 and fc[b] == 2 for a, b, _ in cand), cand

def test_detect_order():
    assert detect_order([Path(f"IMG_{i:04d}.jpg") for i in range(20)]) == 'serial'
    shuffled = [Path(f"IMG_{i:04d}.jpg") for i in [5, 1, 9, 3, 7, 0, 2, 8, 4, 6]]
    assert detect_order(shuffled) == 'unordered'
    assert detect_order([Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]) == 'unordered'

def test_fps_order_covers_clusters():
    """§4.2 FPS in descriptor space optimizes visual diversity: the first picks must
    land in distinct clusters, not all inside the densest one."""
    rng = np.random.default_rng(26)
    cent = np.eye(4, 16) * 5
    rows = np.concatenate([cent[i] + rng.normal(0, 0.1, (30, 16)) for i in range(4)])
    X = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
    first4 = fps_order(X)[:4]
    assert len({i // 30 for i in first4}) == 4, f"FPS missed a cluster: {first4}"

def test_frame_consensus_beats_pixel_pooling():
    """[MEASURED on a 723-frame indoor set] The v1 estimator pools pixels across shared frames, so a frame
    whose geometry is systematically displaced is read as inlier (its pixels agree with each
    other and are numerous). Reproduce that failure synthetically and pin the fix: 3 frames
    agree, 2 sit at a different scale+offset; pixel-pooled IRLS must be dragged off, and
    frame consensus must recover the majority's transform."""
    rng = np.random.default_rng(30)
    gt = sim3_exp(_rand_sim3(rng, max_ang=0.4, max_sig=0.3))
    bad = sim3_compose(gt, sim3_exp(np.array([0.4, -0.3, 0.2, 0.05, 0.05, 0.05, math.log(1.35)])))
    blocks = []
    for k in range(5):
        q = rng.normal(0, 1.0, (4000, 3))
        T = gt if k < 3 else bad
        p = sim3_apply(T, q) + rng.normal(0, 1e-3, (4000, 3))
        blocks.append((p, q, np.ones(4000), k, k))
    P = np.concatenate([b[0] for b in blocks]); Q = np.concatenate([b[1] for b in blocks])
    W = np.concatenate([b[2] for b in blocks])
    pooled, _ = irls_sim3(Q, P, W)
    cons, info = frame_consensus_sim3(blocks)
    e_pool = abs(pooled[0] - gt[0]) / gt[0]
    e_cons = abs(cons[0] - gt[0]) / gt[0]
    assert info['n_consensus'] == 3 and info['n_frames'] == 5, info
    assert set(info['consensus_frames']) == {0, 1, 2}, info['consensus_frames']
    assert e_cons < 0.01, f"consensus scale error {e_cons:.4f}"
    assert e_pool > 5 * e_cons, f"pooling was not actually harmed ({e_pool:.4f} vs {e_cons:.4f})"
    # the measured signature: pooling can land OUTSIDE the span of the per-frame solutions
    assert _rot_deg(cons[1] @ gt[1].T) < 0.5

def test_consensus_gate_rejects_uncorroborated_edge():
    """A single-frame consensus is unverified, not merely noisy: with nothing to corroborate
    it, a displaced frame is accepted with an excellent residual (measured on hub 0<-22,
    consensus 1 of 4 frames yet med_resid 0.048)."""
    ok = lambda **kw: _consensus_ok(dict({'med_resid': 0.01}, **kw) |
                                    {'state': _edge_state(dict({'med_resid': 0.01}, **kw))})
    assert ok(n_frames=4, n_consensus=3, consensus_frac=0.75, rival_frames=0)
    assert not ok(n_frames=4, n_consensus=1, consensus_frac=0.25, rival_frames=0)
    assert not ok(n_frames=30, n_consensus=1, consensus_frac=0.03, rival_frames=0)
    assert ok(n_frames=1, n_consensus=1, consensus_frac=1.0, rival_frames=0)
    assert _consensus_ok({})                                    # legacy info dicts

def test_edge_states_name_the_failure():
    """A binary pass/fail cannot express 'two coherent populations' - the state that
    silently corrupts a merge because it looks like a converged fit."""
    base = {'med_resid': 0.01}
    assert _edge_state(dict(base, n_frames=10, n_consensus=9, consensus_frac=0.9,
                            rival_frames=0)) == EDGE_GOOD
    assert _edge_state(dict(base, n_frames=10, n_consensus=4, consensus_frac=0.4,
                            rival_frames=1)) == EDGE_WEAK
    # 5 frames splitting 3/2 into two self-consistent populations => AMBIGUOUS, not GOOD
    assert _edge_state(dict(base, n_frames=5, n_consensus=3, consensus_frac=0.6,
                            rival_frames=2)) == EDGE_AMBIGUOUS
    # ...but a 2-frame blip against a 27/30 consensus is outvoted, not a rival hypothesis
    # (the consensus refit ignores it and ghost validation still measures its region)
    assert _edge_state(dict(base, n_frames=30, n_consensus=27, consensus_frac=0.9,
                            rival_frames=2)) == EDGE_GOOD
    assert _edge_state(dict(base, n_frames=30, n_consensus=20, consensus_frac=0.67,
                            rival_frames=8)) == EDGE_AMBIGUOUS
    assert _edge_state(dict(base, n_frames=4, n_consensus=1, consensus_frac=0.25,
                            rival_frames=1)) == EDGE_NONRIGID
    assert _edge_state({'med_resid': float('inf'), 'n_frames': 0}) == EDGE_ORPHAN
    assert _edge_state(dict(base, n_frames=1, n_consensus=1, consensus_frac=1.0,
                            rival_frames=0)) == EDGE_WEAK
    # AMBIGUOUS and NONRIGID must never reach the graph
    for st in (EDGE_AMBIGUOUS, EDGE_NONRIGID, EDGE_ORPHAN):
        assert not _consensus_ok({'state': st})
    assert _consensus_ok({'state': EDGE_GOOD}) and _consensus_ok({'state': EDGE_WEAK})

def test_ambiguous_edge_detected_on_split_geometry():
    """End-to-end on the measured signature: 3 frames at one scale, 2 at another.
    The estimator must recover the majority AND flag the edge AMBIGUOUS rather than
    quietly returning a compromise."""
    rng = np.random.default_rng(31)
    gt = sim3_exp(_rand_sim3(rng, max_ang=0.3, max_sig=0.2))
    other = sim3_compose(gt, sim3_exp(np.array([.3, -.2, .1, .04, .04, .04, math.log(1.28)])))
    blocks = []
    for k in range(5):
        q = rng.normal(0, 1.0, (3000, 3))
        T = gt if k < 3 else other
        blocks.append((sim3_apply(T, q) + rng.normal(0, 1e-3, (3000, 3)), q,
                       np.ones(3000), k, k))
    est, info = frame_consensus_sim3(blocks)
    assert info['state'] == EDGE_AMBIGUOUS, info['state']
    assert info['rival_frames'] == 2 and info['n_consensus'] == 3
    assert abs(est[0] - gt[0]) / gt[0] < 0.01           # still recovers the majority
    assert not _consensus_ok(info), "AMBIGUOUS must not enter the pose graph"
    assert len(info['per_frame']) == 5 and sum(f['in_consensus'] for f in info['per_frame']) == 3

def test_sim3_discrepancy_is_unit_free():
    """One tolerance must work at any scene scale, so the metric excludes translation."""
    a = (1.0, np.eye(3), np.zeros(3))
    assert _sim3_discrepancy(a, a) == 0.0
    assert abs(_sim3_discrepancy(a, (1.05, np.eye(3), np.zeros(3))) - math.log(1.05)) < 1e-9
    R = Rotation.from_rotvec([0, 0, math.radians(3)]).as_matrix()
    assert abs(_sim3_discrepancy(a, (1.0, R, np.zeros(3))) - math.radians(3)) < 1e-6
    far = (1.0, np.eye(3), np.array([1e4, 0, 0]))               # huge translation, same s/R
    assert _sim3_discrepancy(a, far) < 1e-9

def test_edge_ghosts_measures_post_graph_placement():
    """Pre-graph gates cannot see a graph that ignored an edge; edge_ghosts can, and it is
    invariant to the global gauge (a pure gauge change moves both chunks identically)."""
    rng = np.random.default_rng(32)
    q = rng.normal(0, 1, (500, 3)).astype(np.float32)
    rel = sim3_exp(_rand_sim3(rng, max_ang=0.4, max_sig=0.3))    # maps j-coords -> i-coords
    p = sim3_apply(rel, q).astype(np.float32)
    edge = [(0, 1, rel, {'ghost_sample': (p, q)})]
    I = (1.0, np.eye(3), np.zeros(3))
    g_ok, ext = edge_ghosts(edge, {0: I, 1: rel})       # graph realized the edge
    g_bad, _ = edge_ghosts(edge, {0: I, 1: I})          # graph ignored it entirely
    assert ext > 0 and g_ok[0] < 0.01, (g_ok, ext)
    assert g_bad[0] > 0.05 and g_bad[0] > 5 * max(g_ok[0], 1e-9), (g_ok, g_bad)
    G = sim3_exp(_rand_sim3(rng))
    g_gauge, _ = edge_ghosts(edge, {0: G, 1: sim3_compose(G, rel)})
    assert g_gauge[0] < 0.01, g_gauge
    g_none, _ = edge_ghosts([(0, 1, rel, {})], {0: I, 1: rel})
    assert g_none == [None]                              # sample-less edges are exempt, not 0

def test_ghost_reject_and_reoptimize_recovers():
    """The measured failure mode end-to-end: a corrupt loop edge drags every chunk when
    solved jointly. Crucially, dilution can make the VICTIMS ghost harder than the culprit
    (observed in this very synthetic), so ghost-argmax must NOT be trusted to name the
    culprit - leave-one-out must: dropping the culprit heals the rest, dropping a victim
    does not. This is main step 6's rejection rule in miniature."""
    rng = np.random.default_rng(33)
    K = 4
    gt = [(1.0, np.eye(3), np.zeros(3))]
    for _ in range(K - 1):
        gt.append(sim3_compose(gt[-1], sim3_exp(_rand_sim3(rng, max_ang=0.3, max_sig=0.15))))

    def mk_edge(i, j, rel):
        q = rng.normal(0, 1, (400, 3)).astype(np.float32)
        return (i, j, rel, {'med_resid': 0.01,
                            'ghost_sample': (sim3_apply(rel, q).astype(np.float32), q)})

    es = [mk_edge(k, k + 1, sim3_compose(sim3_inv(gt[k]), gt[k + 1])) for k in range(K - 1)]
    es.append(mk_edge(0, K - 1, sim3_compose(sim3_inv(gt[0]), gt[K - 1])))   # good loop
    bad = sim3_compose(sim3_compose(sim3_inv(gt[0]), gt[K - 2]),
                       sim3_exp(np.array([3.0, 0., 0., 0., 0., 0., math.log(3.0)])))
    es.append(mk_edge(0, K - 2, bad))                                        # corrupt loop
    CULPRIT = len(es) - 1
    S1 = optimize_pose_graph(K, [(i, j, e) for i, j, e, _ in es], [])
    g1, _ = edge_ghosts(es, dict(enumerate(S1)))
    assert all(g is not None for g in g1)
    assert max(g1) > 0.15, f"corruption not visible in ghosts: {g1}"
    assert sorted(g1[:CULPRIT])[-2] > 0.02, f"expected contamination of good edges, got {g1}"
    # leave-one-out: in a graph with REDUNDANT cycles (like real wheel graphs), only
    # removing the culprit heals the remaining edges. (A single-cycle graph cannot
    # localize blame - any removal leaves a tree, and trees fit anything exactly; that
    # case is what the held-out tie-break in main step 6 exists for.)
    scores = []
    for k in range(len(es)):
        keep = [t for kk, t in enumerate(es) if kk != k]
        Sk = optimize_pose_graph(K, [(i, j, e) for i, j, e, _ in keep], [])
        gk, _ = edge_ghosts(keep, dict(enumerate(Sk)))
        scores.append(max(gk))
    assert int(np.argmin(scores)) == CULPRIT, f"LOO failed to name the culprit: {scores}"
    assert scores[CULPRIT] < 0.05, f"dropping the culprit did not heal: {scores}"
    assert min(scores[:CULPRIT]) > 0.10, f"dropping a victim should NOT heal: {scores}"
    keep = es[:CULPRIT]
    still = _reachable_from_gauge([(t[0], t[1]) for t in keep])
    assert still == set(range(K)), "graph must stay connected after dropping the culprit"

def test_reachable_from_gauge():
    assert _reachable_from_gauge([(0, 1), (1, 2)]) == {0, 1, 2}
    assert _reachable_from_gauge([(0, 1), (2, 3)]) == {0, 1}     # island {2,3} unreached
    assert _reachable_from_gauge([]) == {0}

def test_list_images_txt():
    """.txt list files [2026-07-19]: order preserved AS WRITTEN (line 1 = the
    reference camera), relative paths resolve against the LIST's directory, '#' comments
    and blanks skipped, files may span any directories, missing files fail fast with the
    line number. Directories keep their sorted behaviour."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a").mkdir(); (root / "b" / "deep").mkdir(parents=True)
        for rel in ("a/img_002.jpg", "a/img_001.jpg", "b/deep/zzz.png", "b/other.jpeg"):
            (root / rel).write_bytes(b"x")
        lst = root / "views.txt"
        lst.write_text("# a comment\n"
                       "a/img_002.jpg\n"
                       "\n"
                       f"{(root / 'b' / 'deep' / 'zzz.png')}\n"     # absolute, other dir
                       "b/other.jpeg\n"
                       "a/img_001.jpg\n", encoding='utf-8')
        got = list_images(lst)
        assert [p.name for p in got] == ["img_002.jpg", "zzz.png", "other.jpeg", "img_001.jpg"], \
            "list order must be preserved as written, never sorted"
        assert all(p.exists() for p in got)
        assert len({_ident(p) for p in got}) == 4, "cross-directory identity must stay unique"
        # directory input keeps v1 sorted semantics
        assert [p.name for p in list_images(root / "a")] == ["img_001.jpg", "img_002.jpg"]
        # missing file fails fast, naming the line
        bad = root / "bad.txt"
        bad.write_text("a/img_001.jpg\nnope/missing.jpg\n", encoding='utf-8')
        try:
            list_images(bad)
            assert False, "missing listed file must fail fast"
        except SystemExit as e:
            assert "line 2" in str(e), e

def test_plan_chunks_hub_and_chain():
    """Exercises the planner end-to-end without dvlt.exe: every local must carry 2-4
    bridges and share frames with the skeleton (so a hub edge exists BY CONSTRUCTION,
    §4.5), and every surviving frame must be owned exactly once."""
    rng = np.random.default_rng(28)
    N = 120
    imgs = [Path(f"IMG_{i:04d}.jpg") for i in range(N)]
    rows = np.cumsum(rng.normal(0, 0.3, (N, 24)), axis=0)      # smooth capture-like walk
    Z = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
    cfg = argparse.Namespace(chunk_size=30, overlap=10, loop_steps=16, bridges=3,
                             skeleton_size=12, skeleton_k=16,
                             skeleton_mode='clustered', skeleton_group=4)
    kept = list(range(N))
    for order in ('serial', 'unordered'):
        chunks, skel = plan_chunks(cfg, imgs, Z, kept, order, 'hub')
        assert skel is not None and skel.role == 'skeleton' and skel.idx == 0
        assert len(skel.files) == 12 and skel.k == 16
        locs = [c for c in chunks if c.role == 'local']
        assert locs, order
        for c in locs:
            assert len(c.files) == len(c.g_rows) == len(c.bridge_mask)
            assert len(set(c.g_rows)) == len(c.g_rows), f"{order} chunk {c.idx} stages a frame twice"
            assert len(c.files) <= cfg.chunk_size + 1, (order, len(c.files))
            assert c.bridge_mask[0] is True, "best bridge belongs at the reference slot"
            # The invariant is the EDGE, not the injection count: a picked bridge that is
            # already a natural member of a serial window is not staged twice (that would
            # feed the same image to one run twice), so injections can be as few as 1 while
            # the hub edge still rests on >= --bridges shared frames.
            n_shared = len(shared_frame_indices(skel, c))
            assert n_shared >= 2, f"{order} chunk {c.idx}: hub edge on {n_shared} frames"
            assert 1 <= sum(c.bridge_mask) <= 4, (order, c.idx, sum(c.bridge_mask))
        owner = _owner_map(chunks, N)
        assert (owner >= 0).all(), f"{order}: {(owner < 0).sum()} frames unowned"
        for c in chunks:                                       # no double emission
            for li, g in enumerate(c.g_rows):
                if c.bridge_mask[li]:
                    assert owner[g] != c.idx or c.role == 'skeleton'
    chunks, skel = plan_chunks(cfg, imgs, Z, kept, 'serial', 'chain')
    assert skel is None and all(not any(c.bridge_mask) for c in chunks)
    assert chunks[0].g_rows[:3] == [0, 1, 2] and chunks[0].k == 16

def test_align_shared_no_overlap_is_flagged():
    """Anchorless pairs should not exist (bridges by construction) - but if one appears,
    it must fail loudly, not return identity as though it had aligned."""
    rng = np.random.default_rng(27)
    S, H, W = 3, 8, 8
    p = rng.normal(0, 1, (S, H, W, 3)).astype(np.float32)
    conf = np.full((S, H, W), 3.0, np.float32); rgb = np.zeros((S, H, W, 3), np.uint8)
    po = [{'c2w': np.eye(4), 'K': np.eye(3)}] * S
    A = _FakeChunk(0, p, rgb, conf, po); B = _FakeChunk(100, p, rgb, conf, po)
    assert shared_frame_indices(A, B) == []
    est, info = align_shared(A, B)
    assert info['converged'] is False and info['n'] == 0

def run_self_tests():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print(f"  PASS {name}")
            except Exception as e:
                fails += 1; print(f"  FAIL {name}: {e}")
    print(f"self-test: {'OK' if not fails else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    if '--self-test' in sys.argv:
        run_self_tests()
    else:
        main()
