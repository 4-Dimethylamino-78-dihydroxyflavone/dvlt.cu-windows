#!/usr/bin/env python3
"""
dvlt_postprocess.py — DVLT output → full photogrammetry/3DGS ecosystem export

Usage:
    python dvlt_postprocess.py <out_dvlt_dir> [--images <input_image_dir_or_list.txt>]
                               [--formats ply,glb,...]

Formats (comma list; default ply,glb,las — the rest cost time, ask for what you need):
    ply         scene_upright.ply   ground-corrected point cloud (scene.ply from
                                    the reconstruction is the not-upright one)
    glb         scene.glb + scene_upright.glb   viewers (original axes / Y-up)
    las         scene_upright.las   LAS 1.2 with RGB, Z-up (QGIS / CloudCompare / PDAL)
    xyz         scene_upright.xyz   XYZ-RGB text (CloudCompare / ArcGIS / QGIS; SLOW)
    pcd         scene_upright.pcd   Open3D/PCL binary (ROS / robotics)
    npz         scene_upright.npz   numpy archive: xyz, rgb, poses, intrinsics, rotation
    colmap      colmap/sparse/0/ + colmap/images/   brush.exe / gaussian-splatting
    transforms  transforms.json     Nerfstudio / instant-ngp / many 3DGS trainers
    all         everything above
poses_upright.json (reference upright poses) is always written.
"""

import json
import argparse
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps


IMG_EXTS = {'.jpg', '.jpeg', '.png'}      # compared against suffix.lower()


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def _list_is_abs(s):
    """Mirror dvlt.cu expand_list's absolute test (dvlt.cu:71): a leading '/' or
    '\\' (rooted) OR a drive colon at position 1. Python's Path.is_absolute() is
    False for rooted-but-driveless paths on Windows, so joining them to the list's
    directory would graft the WRONG drive on — diverging from what the exe read."""
    return s[0] in '/\\' or (len(s) > 1 and s[1] == ':')


def _read_list(src):
    """Parse a dvlt.exe .txt list file into image paths, order preserved.
    Mirrors dvlt.cu expand_list: UTF-8 BOM tolerated, '#' comments and blank
    lines skipped (after trimming), relative paths resolve against the list's
    own directory, missing files fail fast with the line number."""
    try:
        raw_text = src.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        sys.exit(f"ERROR: {src} is not UTF-8 (UTF-16?). Re-save as UTF-8 — "
                 f"PowerShell: `Out-File -Encoding utf8`, or `cmd /c type ... > list.txt`.")
    imgs = []
    for ln, line in enumerate(raw_text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        p = Path(s) if _list_is_abs(s) else src.parent / s
        if not p.exists():
            sys.exit(f"ERROR: {src} line {ln}: file not found: {p}")
        imgs.append(p)
    return imgs


def collect_images(src):
    """Image paths in dvlt.exe frame order — poses.json pairs by position (it
    carries no filenames), so this must reproduce exactly what the exe consumed.
    A directory is sorted bytewise by name (mirrors dvlt.cu expand_dir's
    std::sort — case-SENSITIVE, unlike Python's default WindowsPath ordering).
    A .txt list keeps the order written (line 1 = reference camera)."""
    if src.is_file() and src.suffix.lower() == '.txt':
        return _read_list(src)
    return sorted((p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS),
                  key=lambda p: p.name)


def exif_orientation(im):
    """EXIF orientation (tag 0x0112) exactly as dvlt.cu's exif_orientation reads
    it: JPEG only, 1..8 else 1. The exe loads the image upright, so 5-8 swap W/H."""
    o = im.getexif().get(0x0112, 1) if im.format in ('JPEG', 'MPO') else 1
    return o if o in range(1, 9) else 1


def load_poses(path):
    with open(path) as f:
        data = json.load(f)
    return [{'c2w': np.array(fr['extrinsics_c2w'], dtype=np.float64).reshape(4, 4),
             'K':   np.array(fr['intrinsics'],     dtype=np.float64).reshape(3, 3)}
            for fr in data['frames']]


def load_ply(path):
    """Return (xyz float32 [N,3], rgb uint8 [N,3])."""
    pc = trimesh.load(str(path))
    xyz = np.asarray(pc.vertices, dtype=np.float32)
    rgba = np.asarray(pc.visual.vertex_colors, dtype=np.uint8) if hasattr(pc.visual, 'vertex_colors') else None
    rgb  = rgba[:, :3] if rgba is not None else np.zeros((len(xyz), 3), dtype=np.uint8)
    return xyz, rgb


def rot_to_quat(R):
    """3×3 → quaternion [qw, qx, qy, qz] — Shepperd method."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


# ─── orientation correction ───────────────────────────────────────────────────

def _rodrigues(src, tgt):
    """3×3 rotation that maps unit vector src → unit vector tgt."""
    v = np.cross(src, tgt)
    c, s = float(np.dot(src, tgt)), float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.]).astype(float)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def detect_ground_rotation(xyz, poses, seed=42):
    """
    Primary method: gravity from the null space of the camera X-axes.
    Image-right stays horizontal under any camera PITCH (the dominant motion
    in handheld/drone capture), so gravity is the direction most orthogonal
    to every c2w[:,0]. Averaging c2w[:,1] instead (the old method) mixes
    pitch into the estimate and — DVLT cameras being OpenCV c2w, +Y = image
    DOWN — points the wrong way for an upright camera.

    Sign disambiguation: prefer "most points sit below the cameras"
    (handheld/drone shoots from above scene mass); when that is ambiguous
    (e.g. mid-tunnel captures) fall back to the photographer's up, -avg cam-Y.

    Fallback: RANSAC ground plane (roll-heavy captures where the X-axis
    null space degenerates), with the same camera-above-ground check.
    """
    cam_x   = np.array([p['c2w'][:3, 0] for p in poses])   # image-right [N,3]
    cam_y   = np.array([p['c2w'][:3, 1] for p in poses])   # image-down  [N,3]
    cam_z   = np.array([p['c2w'][:3, 2] for p in poses])   # optical axis [N,3]
    cam_pos = np.array([p['c2w'][:3, 3] for p in poses])

    # ── 0. Nadir grid (aerial survey) ─────────────────────────────────────────
    # Looking straight down, BOTH image axes are horizontal: the cam-X null
    # space is a 2-D ambiguity (true-up + a horizontal direction) and can pick
    # a horizontal "up" with low residual, so the resid gate below never
    # catches it. A nadir grid is identified by planar camera positions whose
    # plane normal agrees with a consistent mean optical axis; that normal IS
    # gravity. Orbits and walkthroughs fail |mean cam_z| (axes cancel or turn);
    # linear strafes fail the planarity test.
    ctr = cam_pos - cam_pos.mean(0)
    psv = np.linalg.svd(ctr, compute_uv=False)
    normal = np.linalg.svd(ctr)[2][-1]
    mz = cam_z.mean(0); mzn = float(np.linalg.norm(mz))
    if (psv[1] > 0.5 * psv[0] and psv[2] < 0.25 * psv[1]
            and mzn > 0.6 and abs(float(normal @ (mz / mzn))) > 0.8):
        if (cam_pos @ normal).mean() < np.median(xyz @ normal):
            normal = -normal                        # cameras sit above the scene
        R = _rodrigues(normal, np.array([0., 1., 0.]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        print(f"  nadir grid: up = camera-plane normal  {angle:.1f} deg  "
              f"(psv={psv[1]/psv[0]:.2f}/{psv[2]/psv[1]:.2f}, |mean cam_z|={mzn:.2f})")
        return R

    # ── 1. Gravity = null space of image-right axes ──────────────────────────
    _, svals, vecs = np.linalg.svd(cam_x)
    up    = vecs[-1]                                # least-spanned direction
    resid = float(np.abs(cam_x @ up).mean())        # 0 = perfectly horizontal rights

    if resid < 0.25:
        ph_up = -cam_y.mean(axis=0)                 # photographer's up (OpenCV Y-down)
        # sign: photographer's up decides when decisive - nobody shoots a scene
        # upside down. The scene-mass prior ("most points below cameras") is
        # scene-DEPENDENT: sky-heavy terrestrial shots put most points ABOVE
        # the cameras (building dataset: 68% sky) and it votes inverted, so it
        # only breaks the near-vertical-pitch ambiguity where ph_up goes flat.
        def frac_above(u):
            return float(((xyz @ u) > (cam_pos @ u).mean()).mean())
        d = float(np.dot(up, ph_up))
        if abs(d) > 0.2:
            if d < 0:
                up = -up
        elif frac_above(up) > 0.5:
            up = -up
        R = _rodrigues(up, np.array([0., 1., 0.]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        print(f"  cam-right null-space up: {angle:.1f}°  resid={resid:.2f}  "
              f"pts-above-cams={frac_above(up):.2f}  up=[{up[0]:.2f},{up[1]:.2f},{up[2]:.2f}]")
        return R

    # ── 2. RANSAC fallback (roll-heavy capture) ───────────────────────────────
    print(f"  cam-right axes not horizontal (resid={resid:.2f}), using RANSAC ground plane")
    rng = np.random.default_rng(seed)
    thresh = np.percentile(xyz[:, 1], 15)
    ground = xyz[xyz[:, 1] < thresh]
    if len(ground) < 100:
        print("  [warn] too few ground points — identity")
        return np.eye(3)

    tol = max(np.percentile(np.abs(ground[:, 1] - ground[:, 1].mean()), 75) * 0.4, 0.005)
    best_normal, best_n = None, 0
    for _ in range(500):
        i0, i1, i2 = rng.integers(0, len(ground), 3)
        p0, p1, p2 = ground[i0], ground[i1], ground[i2]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n);
        if nn < 1e-9: continue
        n /= nn
        if int(np.sum(np.abs((ground - p0) @ n) < tol)) > best_n:
            best_n = int(np.sum(np.abs((ground - p0) @ n) < tol)); best_normal = n.copy()

    if best_normal is None:
        return np.eye(3)

    mask = np.abs((ground - ground.mean(0)) @ best_normal) < tol
    if mask.sum() >= 3:
        g2 = ground[mask]
        _, vecs = np.linalg.eigh(np.cov((g2 - g2.mean(0)).T))
        best_normal = vecs[:, 0]
    if best_normal[1] < 0:
        best_normal = -best_normal

    R = _rodrigues(best_normal, np.array([0., 1., 0.]))

    # Sanity: cameras should be above ground (positive Y after rotation)
    cam_pos = np.array([p['c2w'][:3, 3] for p in poses])
    if (R @ cam_pos.T).T[:, 1].mean() < 0:
        R = np.diag([1., -1., -1.]).astype(float) @ R

    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"  RANSAC ground: {angle:.1f}°  (inliers: {best_n}/{len(ground)})")
    return R


def apply_rotation(xyz, poses, R3):
    R4 = np.eye(4); R4[:3, :3] = R3
    return (R3 @ xyz.T).T, [{'c2w': R4 @ p['c2w'], 'K': p['K'].copy()} for p in poses]


# ─── exporters ────────────────────────────────────────────────────────────────

def export_glb(xyz, rgb, path):
    """Point-cloud GLB written directly. glTF defines COLOR_0 as LINEAR: trimesh
    stores the photos' sRGB bytes as-is, so spec-following importers (Blender,
    three.js) brightened every point a second time (0.73 shown as 0.87). Colours
    are linearised and kept as uint16 - 8-bit linear bands the shadows away, and
    these are night captures. VEC4, not VEC3: vertex attributes need 4-byte strides."""
    pos = np.ascontiguousarray(xyz, dtype=np.float32)
    c = rgb[:, :3].astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    col = np.full((len(pos), 4), 65535, np.uint16)
    col[:, :3] = np.round(lin * 65535)
    pb, cb = pos.tobytes(), col.tobytes()          # 12n and 8n bytes: 4-aligned
    acc_pos = {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"}
    if len(pos):
        acc_pos.update(min=pos.min(0).tolist(), max=pos.max(0).tolist())
    gltf = {"asset": {"version": "2.0", "generator": "dvlt_postprocess"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}],
            "buffers": [{"byteLength": len(pb) + len(cb)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pb), "target": 34962},
                            {"buffer": 0, "byteOffset": len(pb), "byteLength": len(cb), "target": 34962}],
            "accessors": [acc_pos, {"bufferView": 1, "componentType": 5123, "normalized": True,
                                    "count": len(pos), "type": "VEC4"}]}
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(pb) + len(cb)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(pb) + len(cb), 0x004E4942) + pb + cb)


def export_las(xyz, rgb, path):
    """LAS 1.2 point format 2 (XYZ + RGB), written directly (laspy is not bundled).
    Stored Z-up as (x, -z, y), a rotation of the Y-up upright frame: every LAS tool
    reads Z as height. LAS keeps i32 = (v - offset) / scale, and DVLT's units are
    the scene's own, where a lidar-style fixed 1 mm would band a 2-unit room. Scale
    is the largest power of ten no coarser than float32 at this extent (2^-24 of
    it), leaving i32 headroom for tools that re-offset. Colours are 16-bit by spec."""
    v = np.column_stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1])).astype(np.float64)
    lo, hi = (v.min(0), v.max(0)) if len(v) else (np.zeros(3), np.zeros(3))
    extent = float((hi - lo).max())
    scale = 10.0 ** np.floor(np.log10(extent / 2**24)) if extent > 0 else 1e-3
    offset = (lo + hi) / 2
    rec = np.zeros(len(v), [('xyz', '<i4', 3), ('intensity', '<u2'), ('returns', 'u1'),
                            ('class', 'u1'), ('angle', 'i1'), ('user', 'u1'),
                            ('source', '<u2'), ('rgb', '<u2', 3)])
    rec['xyz'] = np.round((v - offset) / scale)
    rec['returns'] = 0b001001                         # return 1 of 1; 0 is not a valid return
    rec['rgb'] = rgb[:, :3].astype(np.uint16) * 257   # 255 -> 65535
    lo, hi = (offset + np.round((b - offset) / scale) * scale for b in (lo, hi))   # as stored
    now = time.gmtime()
    head = struct.pack('<4sHHIHH8sBB32s32sHHHIIBHI5I12d', b'LASF', 0, 0, 0, 0, 0, bytes(8), 1, 2,
                       b'OTHER', b'dvlt_postprocess', now.tm_yday, now.tm_year, 227, 227, 0,
                       2, 26, len(rec), len(rec), 0, 0, 0, 0, scale, scale, scale, *offset,
                       hi[0], lo[0], hi[1], lo[1], hi[2], lo[2])
    with open(path, 'wb') as f:
        f.write(head)
        rec.tofile(f)


def export_xyz(xyz, rgb, path):
    """Space-separated XYZ-RGB text. CloudCompare / MeshLab / ArcGIS / QGIS compatible."""
    print(f"  writing {path.name} ({len(xyz):,} pts) …")
    chunk = 50_000
    with open(path, 'w') as f:
        for i in range(0, len(xyz), chunk):
            xc, rc = xyz[i:i+chunk], rgb[i:i+chunk]
            lines = [f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}"
                     for (x,y,z),(r,g,b) in zip(xc, rc)]
            f.write('\n'.join(lines) + '\n')


def export_pcd(xyz, rgb, path):
    """Binary PCD v0.7 written directly (open3d is not bundled) — PCL / ROS / Open3D
    native. rgb is PCL's packed float: the bits of uint32 0x00RRGGBB."""
    rec = np.empty(len(xyz), [('xyz', '<f4', 3), ('rgb', '<u4')])
    rec['xyz'] = xyz
    rec['rgb'] = rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(f"VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
                f"WIDTH {len(rec)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(rec)}\n"
                f"DATA binary\n".encode())
        rec.tofile(f)


def export_npz(xyz, rgb, poses, path, R_fix):
    """
    Numpy archive — direct ML / research use.
    Keys: xyz [N,3], rgb [N,3] uint8, poses_c2w [M,4,4], intrinsics_K [M,3,3],
          upright_R [3,3] (applied rotation)
    """
    c2w_arr = np.stack([p['c2w'] for p in poses]).astype(np.float32)
    K_arr   = np.stack([p['K']   for p in poses]).astype(np.float32)
    np.savez_compressed(str(path),
                        xyz=xyz.astype(np.float32),
                        rgb=rgb,
                        poses_c2w=c2w_arr,
                        intrinsics_K=K_arr,
                        upright_R=R_fix.astype(np.float32))


def export_transforms_json(poses, image_names, dvlt_wh, orig_wh, path):
    """
    Nerfstudio / instant-ngp / many 3DGS trainers.
    DVLT c2w (OpenCV) → Nerfstudio c2w (OpenGL) by flipping Y+Z.
    Convention: X right, Y up, -Z forward (OpenGL / NeRF standard).
    """
    # OpenCV → OpenGL per-camera: flip Y and Z columns of the rotation part
    flip_yz = np.diag([1., -1., -1., 1.])

    dW, dH = dvlt_wh
    oW, oH = orig_wh
    sx, sy = oW / dW, oH / dH

    K0 = poses[0]['K']
    fx = K0[0, 0] * sx; fy = K0[1, 1] * sy
    cx = K0[0, 2] * sx; cy = K0[1, 2] * sy

    frames = []
    for i, (p, name) in enumerate(zip(poses, image_names)):
        ns_c2w = (p['c2w'] @ flip_yz).tolist()
        frames.append({'file_path': f'colmap/images/{name}',
                       'transform_matrix': ns_c2w})

    doc = {
        'camera_model': 'OPENCV',
        'fl_x': fx, 'fl_y': fy,
        'cx':   cx, 'cy':   cy,
        'w': oW, 'h': oH,
        '_convention': 'OpenGL (Y-up, -Z forward); DVLT c2w flipped via diag(1,-1,-1,1)',
        '_colmap_compatible': True,
        'frames': frames
    }
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)


def write_colmap(colmap_dir, poses, xyz, rgb, image_names, dvlt_wh, orig_wh):
    sparse = colmap_dir / 'sparse' / '0'
    sparse.mkdir(parents=True, exist_ok=True)

    dW, dH = dvlt_wh
    oW, oH = orig_wh
    sx, sy = oW / dW, oH / dH

    with open(sparse / 'cameras.txt', 'w') as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
        for i, p in enumerate(poses):
            K = p['K']
            f.write(f"{i+1} PINHOLE {oW} {oH} "
                    f"{K[0,0]*sx:.6f} {K[1,1]*sy:.6f} "
                    f"{K[0,2]*sx:.6f} {K[1,2]*sy:.6f}\n")

    with open(sparse / 'images.txt', 'w') as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i, (p, name) in enumerate(zip(poses, image_names)):
            Rc = p['c2w'][:3, :3]; tc = p['c2w'][:3, 3]
            Rw = Rc.T; tw = -Rw @ tc
            qw, qx, qy, qz = rot_to_quat(Rw)
            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tw[0]:.9f} {tw[1]:.9f} {tw[2]:.9f} {i+1} {name}\n\n")

    # Sub-sample points for COLMAP init file (brush only uses this as starting cloud)
    step = max(1, len(xyz) // 300_000)
    xw, rw = xyz[::step], rgb[::step]
    print(f"  points3D.txt: {len(xw):,} pts (step={step})")
    with open(sparse / 'points3D.txt', 'w') as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for j, (pt, col) in enumerate(zip(xw, rw)):
            f.write(f"{j+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{col[0]} {col[1]} {col[2]} 0.0\n")


# ─── main ─────────────────────────────────────────────────────────────────────

FORMATS = {'ply', 'glb', 'las', 'xyz', 'pcd', 'npz', 'colmap', 'transforms'}


def main():
    sys.stdout.reconfigure(errors='replace')  # a redirected run on a non-UTF-8 code page
                                              # must not die printing '→' (stderr already
                                              # uses backslashreplace)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--images', '-i', default=None,
                    help='Source image dir OR .txt list file, exactly as given to the '
                         'reconstruction (enables intrinsic rescaling + COLMAP images/). '
                         'For a dvlt_long merged/ folder pass merged/images.txt: the '
                         'frames it kept, as dropped frames have no pose')
    ap.add_argument('--formats', '-f', default='ply,glb,las',
                    help="Comma list of outputs (default: ply,glb,las). "
                         f"Choices: {', '.join(sorted(FORMATS))}, all")
    args = ap.parse_args()

    fmts = {f.strip().lower() for f in args.formats.split(',') if f.strip()}
    if 'all' in fmts:
        fmts = set(FORMATS)
    bad = fmts - FORMATS
    if bad:
        sys.exit(f"ERROR: unknown format(s) {', '.join(sorted(bad))} "
                 f"(choose from: {', '.join(sorted(FORMATS))}, all)")

    out  = Path(args.out_dir)
    ply  = out / 'scene.ply'
    pjsn = out / 'poses.json'
    for p in (ply, pjsn):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    print(f"\n=== dvlt_postprocess: {out} ===")
    print(f"    formats: {', '.join(sorted(fmts))}\n")

    # ── load ──────────────────────────────────────────────────────────────────
    print("[poses ] loading poses.json …")
    poses = load_poses(pjsn)
    print(f"         {len(poses)} frames")

    print("[points] loading scene.ply …")
    xyz, rgb = load_ply(ply)
    print(f"         {len(xyz):,} points")

    # ── source images ─────────────────────────────────────────────────────────
    if args.images:
        imgs_src = collect_images(Path(args.images))
        if len(imgs_src) != len(poses):
            sys.exit(f"ERROR: --images yields {len(imgs_src)} images but poses.json has "
                     f"{len(poses)} poses. Pose/image pairing is positional, so these "
                     f"must match: pass the same directory or .txt list the "
                     f"reconstruction ran on."
                     + (f" This is a dvlt_long merge, which may drop frames - pass "
                        f"{out / 'images.txt'} instead of the original input."
                        if (out / 'images.txt').exists() else ""))
        image_names = [f"frame_{i:05d}{p.suffix.lower().replace('.jpeg','.jpg')}"
                       for i, p in enumerate(imgs_src)]
        with Image.open(imgs_src[0]) as im:
            orig_wh = im.size[::-1] if exif_orientation(im) >= 5 else im.size
        print(f"         source: {len(imgs_src)} images @ {orig_wh[0]}×{orig_wh[1]}")
    else:
        imgs_src    = []
        image_names = [f"frame_{i:05d}.jpg" for i in range(len(poses))]
        orig_wh     = None

    K0 = poses[0]['K']
    dvlt_wh = (int(round(K0[0,2] * 2)), int(round(K0[1,2] * 2)))
    if orig_wh is None:
        orig_wh = dvlt_wh
    elif fmts & {'colmap', 'transforms'} and \
            abs(np.log(orig_wh[0] * dvlt_wh[1] / (orig_wh[1] * dvlt_wh[0]))) > 0.2:
        print(f"  [warn] upright source {orig_wh[0]}x{orig_wh[1]} but the reconstruction saw "
              f"{dvlt_wh[0]}x{dvlt_wh[1]}: colmap/transforms intrinsics will be wrong. "
              f"Re-run it with a dvlt.exe that applies EXIF orientation; mixed "
              f"portrait/landscape input is not supported.")

    # ── orientation ───────────────────────────────────────────────────────────
    print("[orient] ground orientation correction …")
    R_fix = detect_ground_rotation(xyz, poses)
    xyz_up, poses_up = apply_rotation(xyz, poses, R_fix)

    written = []                                   # (filename, note) for the summary

    if 'ply' in fmts:
        print("[ply   ] scene_upright.ply …")
        trimesh.PointCloud(vertices=xyz_up.astype(np.float32), colors=rgb).export(
            str(out / 'scene_upright.ply'))
        written.append(('scene_upright.ply', 'upright PLY (trimesh / MeshLab)'))

    if 'glb' in fmts:
        print("[glb   ] scene.glb + scene_upright.glb …")
        export_glb(xyz,    rgb, out / 'scene.glb')
        export_glb(xyz_up, rgb, out / 'scene_upright.glb')
        written.append(('scene.glb', 'GLB — original axes'))
        written.append(('scene_upright.glb', 'GLB — Y-up, ground-aligned'))

    if 'las' in fmts:
        print("[las   ] scene_upright.las …")
        export_las(xyz_up, rgb, out / 'scene_upright.las')
        written.append(('scene_upright.las', 'LAS 1.2, Z-up (QGIS / CloudCompare / PDAL)'))

    if 'xyz' in fmts:
        print("[xyz   ] scene_upright.xyz …")
        export_xyz(xyz_up, rgb, out / 'scene_upright.xyz')
        written.append(('scene_upright.xyz', 'XYZ-RGB text (CloudCompare / GIS)'))

    if 'pcd' in fmts:
        print("[pcd   ] scene_upright.pcd …")
        export_pcd(xyz_up, rgb, out / 'scene_upright.pcd')
        written.append(('scene_upright.pcd', 'binary PCD (Open3D / PCL / ROS)'))

    if 'npz' in fmts:
        print("[npz   ] scene_upright.npz …")
        export_npz(xyz_up, rgb, poses_up, out / 'scene_upright.npz', R_fix)
        written.append(('scene_upright.npz', 'numpy (xyz,rgb,poses,K,R) for ML'))

    if 'colmap' in fmts:
        print("[colmap] COLMAP sparse (brush.exe / gaussian-splatting) …")
        colmap_dir = out / 'colmap'
        write_colmap(colmap_dir, poses_up, xyz_up.astype(np.float32), rgb,
                     image_names, dvlt_wh, orig_wh)
        written.append(('colmap/sparse/0/', 'COLMAP text (brush.exe / 3DGS)'))
        if imgs_src:
            images_dir = colmap_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            # Trainers' PIL loaders ignore EXIF orientation, so a rotated frame is
            # stored upright, as the exe saw it - always rewritten, as an older run
            # may have left the raw file under the same name.
            copied = rotated = 0
            for src, name in zip(imgs_src, image_names):
                dst = images_dir / name
                with Image.open(src) as im:
                    if exif_orientation(im) != 1:
                        up = ImageOps.exif_transpose(im)   # drops the orientation tag
                        up.save(dst, quality=95, exif=up.info.get('exif', b''),
                                icc_profile=up.info.get('icc_profile'))
                        copied += 1; rotated += 1
                    elif (not dst.exists() or dst.stat().st_size != src.stat().st_size
                          or int(dst.stat().st_mtime) != int(src.stat().st_mtime)):
                        # rsync's quick check (copy2 keeps the source mtime): a
                        # re-run whose frames changed - a different set, or newly
                        # corrected --ppisp frames - must not keep the old copies,
                        # which would pair the new poses with the old pixels.
                        shutil.copy2(src, dst)
                        copied += 1
            # A smaller set leaves the previous run's tail behind.
            keep = set(image_names)
            stale = [p for p in images_dir.glob('frame_*') if p.name not in keep]
            for p in stale:
                p.unlink()
            print(f"         copied {copied} images → colmap/images/ ({rotated} rotated upright"
                  + (f", {len(stale)} stale removed" if stale else "") + ")")
            written.append(('colmap/images/', f'{len(imgs_src)} source images '
                            f'({orig_wh[0]}x{orig_wh[1]})'))
        else:
            print("         (no --images; colmap/images/ skipped)")

    if 'transforms' in fmts:
        print("[nerf  ] transforms.json (Nerfstudio / instant-ngp / 3DGS trainers) …")
        export_transforms_json(poses_up, image_names, dvlt_wh, orig_wh,
                               out / 'transforms.json')
        written.append(('transforms.json', 'Nerfstudio / instant-ngp / gsplat'))

    # ── upright poses.json ────────────────────────────────────────────────────
    with open(out / 'poses_upright.json', 'w') as f:
        json.dump({'frames': [{'extrinsics_c2w': p['c2w'].flatten().tolist(),
                                'intrinsics':     p['K'].flatten().tolist()}
                               for p in poses_up],
                   'upright_rotation_3x3': R_fix.tolist()}, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    written.append(('poses_upright.json', 'corrected DVLT poses'))
    print(f"\nwritten to {out}:")
    for name, note in written:
        print(f"  {name:<22} {note}")
    if fmts != FORMATS:
        rest = ', '.join(sorted(FORMATS - fmts))
        print(f"  (skipped: {rest} — request via --formats, e.g. -f all)")


if __name__ == '__main__':
    main()
