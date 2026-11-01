# dvlt.cu-windows

Native Windows build of [dvlt.cu](https://github.com/yassa9/dvlt.cu), yassa9's single-binary
CUDA/C++ inference engine for NVIDIA's **DVLT / Déjà View** looped transformer
([arXiv:2605.30215](https://arxiv.org/abs/2605.30215)): a folder of photos in, a dense point cloud
plus camera poses out, in one forward pass, with no per-scene training. On top of the binary sit the
wrappers that make it a field tool: an auto-sizing `reconstruct.bat`, a long-sequence chunk aligner,
and a post-processor that writes upright PLY/GLB and the COLMAP / Nerfstudio layouts 3DGS trainers
expect.

**Every release zip is self-contained.** The exe, the one CUDA DLL it needs, the model weights, a
private Python runtime and a sample scene are all inside. Unzip, run. Nothing to install, nothing
on PATH, no conda.

## Run it

1. Download `dvlt-win64-<version>.zip` from [Releases](../../releases/latest) and unzip it anywhere.
2. In a terminal (cmd or PowerShell) opened in that folder:

   ```
   .\reconstruct.bat inputs\shoe output\shoe
   ```

3. Look at `output\shoe\scene_upright.ply` or `scene.glb` in any point-cloud viewer, or open
   `tools\viewer.html` in a browser and drag `scene.ply` + `poses.json` onto it to see the cameras too.

What it needs from the machine:

- An NVIDIA GPU with bf16 tensor cores: **Ampere or newer** (RTX 30 / 40 / 50, A- and L-series).
  Native code for sm_80/86/89 plus PTX that the driver compiles forward for Hopper and Blackwell.
  Turing (RTX 20) and older cannot run this model; `dvlt.exe` says so instead of crashing.
- A driver from 2023 or later (CUDA 12 runtime support, 527.41+). The CUDA toolkit is **not** needed;
  a missing or unusable `cublasLt64_12.dll` is reported by name.
- VRAM sets the frame budget: ~80 frames at 504 px on 16 GB. `reconstruct.bat` reads the card's
  total VRAM from `nvidia-smi` and scales every frame limit by what is left after ~2 GB of fixed
  overhead (16 GB assumed if `nvidia-smi` is absent; the total snaps to whole GiB, so every 16 GB
  card gets exactly the limits below); the banner prints the limits it uses. Below ~4 GB a set too
  large for one pass stops with a clear message instead of failing inside `dvlt_long`.

## Usage

```
reconstruct.bat <image_dir | list.txt> [output_dir] [conf_drop 0-1] [loop_steps 8-16] [formats]
```

Flags are accepted anywhere: `-o/--out`, `-c/--conf`, `-k/--loop-steps`, `-f/--formats`.
A bare number after the input is never taken as the output folder: one with a `.` (or 0/1) is
`conf_drop`, any other integer is `loop_steps`, so `reconstruct.bat img\scene 0.3 12` keeps the
default output. A folder that is named like a number needs `-o`.

Format names are never the output folder either. cmd splits unquoted commas and PowerShell drops
the quotes that would protect them, so any argument made only of format names joins the list
wherever it stands: `-f colmap,glb`, `-f colmap glb` and `img\scene 0.3 12 colmap,glb` all mean
`colmap,glb`. A folder named like a format (`glb`) needs `-o`.

The script can be called from any folder, by any path; relative data paths are taken from the
program folder (the script runs there), so give full paths for data elsewhere.

A `list.txt` (e.g. a vfold selection) has one image path per line, `#` comments allowed; order is
kept as written and line 1 becomes the reference camera. A directory is sorted by name.

| arg | default | meaning |
|---|---|---|
| `output_dir` | `<input>_out_dvlt` | written **beside** the input, never inside an image folder (where COLMAP / 3DGS trainers would ingest it): `img\scene` → `img\scene_out_dvlt`; a list inside its image folder steps out beside it, `scene\vfold-60.txt` → `scene_vfold-60_out_dvlt`; a list elsewhere gets one beside it, `lists\vfold-60.txt` → `lists\vfold-60_out_dvlt`. Created before anything runs, so an unwritable target (read-only media) fails at once |
| `conf_drop` | 0.5 | drop that **fraction** of lowest-confidence points (a percentile, not a threshold, so it changes density, not geometry; 0.5 matches NVIDIA's demo default and suppresses sky hallucination. Lower toward 0.1 to keep low-texture ground like water or shadow). Single pass only |
| `loop_steps` | 16 | looped-transformer iterations K. Trained range **[8, 16]**; 16 = paper optimum. K>16 degrades poses and collapses past ~30 (paper App. D); K<8 is out-of-distribution. Accepted: 1–64, no leading zeros |
| `formats` | `ply,glb` | post-processor outputs, comma list from `ply,glb,xyz,pcd,npz,colmap,transforms`, or `all`. `xyz` is by far the slowest; `pcd` is written directly, no open3d needed |

- Single pass up to 150 frames on 16 GB: resolution auto-steps 504 → 448 → 392 → 336 px past 80,
  110 and 140 frames. Every limit scales with VRAM: on 8 GB it is 34 / 47 / 60, single pass up to 65.
- Beyond that: auto-routes to `dvlt_long.py` (60/30 windows on 16 GB, 26/13 on 8 GB, at full 504 px;
  per-frame-consensus IRLS Sim3 seams, loop closure from backbone descriptors + EXIF GPS, global
  pose graph with post-graph ghost validation, streaming merge + `manifest.json`). Chain topology
  by default; `--topology hub` (skeleton + bridges) and `--order unordered` exist for unordered
  dumps. Merging is the weak stage: for a long capture,
  a vfold selection of ≤150 views run single-pass is usually the cleaner cloud. `conf_drop` does
  not apply there (dvlt_long keeps points above 0.75× each chunk's mean confidence) and the banner
  says so. Run directly, `dvlt_long.py` resolves relative paths itself and loads the weights from
  beside `dvlt.exe`, whatever the working folder. Each chunk gets a `frames.txt` list (no staged
  copies, so inputs on an SD card or another drive are never duplicated); every image path must
  fit the system code page, as `dvlt.exe` opens paths through it, and an unfit name is reported
  before any GPU work. A merge may drop orphaned frames: re-run post-processing on a merged folder
  with `--images <out>\merged\images.txt`, the frames it kept in pose order.
- Python: the local `python\` when present (release zips ship it, `setup.bat` builds it), else the
  first `python.exe` on PATH, named in the banner. A list whose images cannot be counted (no
  working Python, a UTF-16 file) stops the run instead of single-passing an unknown count.
- Any failure in `dvlt.exe`, `dvlt_long.py` or post-processing ends the run with a non-zero exit code.
- Raw binary: `run.bat` (`dvlt.exe -w model\weights.dvlt`, from any folder; a later `-w` wins
  and the exit code is dvlt.exe's) / `dvlt.exe -h` (`-o -w -c -s -k -i -x -e -b`).

### Optional preprocessing: `--sky`, `--ppisp`

Two detached tools, found beside the script (`vsky.exe`, `vppisp.exe`, or in `vsky\` / `vppisp\`),
else as sibling source checkouts' release builds (`..\vsky`, `..\vppisp`); `VSKY` / `VPPISP`
override. vsky needs `onnxruntime_providers_*.dll` beside it for CUDA (the same files vfold uses).
Single-pass runs only; intermediates land in `<output>\prep\` and are deleted after a successful
run unless `--keep-prep` is given (the COLMAP export already holds the corrected frames).

- `--sky` (vsky): a sky segmenter (SegFormer ADE20K ONNX, ceiling-aware; CUDA else CPU) masks every frame, dvlt runs on
  copies with the sky painted pure black, and every exactly-black point is then stripped from
  `scene.ply` (original kept as `scene_withsky.ply`). The black key carries per-pixel provenance
  through the PLY's colour, so removal is exact rather than voted. On the 60-frame dorm set:
  sky-blue points 21.5% → 3.1%, 0.18% of ground points lost.
- `--ppisp` (vppisp): Program AE and auto white balance re-expose every frame (3.5 EV across the
  same set). A first dvlt pass supplies multi-view correspondences, NVIDIA's PPISP camera model
  is fitted to them (Vulkan GPU, CPU fallback), and every frame is re-rendered at one exposure.
  Cross-view colour error 11.0 → 7.6 /255. `colmap/images/` gets these corrected frames, never
  the keyed ones.
- `-d` / `--default`: a preset (the `D_*` lines near the top of `reconstruct.bat`, shipped as
  `-c 0.1 -k 16 -f all --sky --ppisp`) fills every option not given on the command line;
  anything typed still wins, and `--no-sky` / `--no-ppisp` opt out for one run.

### Outputs

`dvlt.exe`: `scene.ply` (first-camera frame, OpenCV axes, +Y is image-down), `poses.json` (c2w +
intrinsics), `report.txt`. `dvlt_postprocess.py` adds `scene_upright.ply`, `scene.glb`,
`scene_upright.glb`, `poses_upright.json`, and on request `scene_upright.{xyz,pcd,npz}`,
`colmap/sparse/0/` + `colmap/images/` (brush / 3DGS trainers) and `transforms.json` (Nerfstudio).
It needs no open3d for `pcd`, and runs on consoles whose code page is not UTF-8. GLB colours are
linear, as glTF requires (Blender imports them unwashed). Phone photos with an EXIF rotation are
written upright into `colmap/images/` (JPEG q95) with matching camera sizes, since trainers load
pixels without EXIF; if the source and the reconstruction disagree on aspect, colmap/transforms
output warns.

`dvlt.exe` applies EXIF orientation on load, so portrait and landscape frames can share a batch,
but every frame is then padded to the widest and tallest (a square grid at 504 px): ~1.5× the
tokens and noticeably more VRAM and time for the whole set. Keep one orientation per run when
the set is large.

Upright detection: gravity = null space of the camera **X**-axes (image-right stays horizontal under
pitch), sign from a points-below-cameras prior with photographer's-up tiebreak, RANSAC ground-plane
fallback for roll-heavy rigs.

## Build from source

Needs Visual Studio 2022 Build Tools (C++ x64 workload), the CUDA Toolkit 12.1 (any 12.x should
work; 12.1 is what CI builds with) and git.

```
setup.bat
```

fetches the CUTLASS headers, copies `cublasLt64_12.dll` next to the exe, builds `dvlt.exe` and
`convert.exe`, downloads the `nvidia/dvlt` checkpoint and its `model\LICENSE.txt` from Hugging Face,
converts the checkpoint to `model\weights.dvlt`, and builds the local `python\` runtime the
wrappers prefer. Each step is skipped when its output exists. Afterwards:

- `build.bat` rebuilds after source edits (about 2.5 minutes on a 6-core desktop).
- `python\python.exe dvlt_long.py --self-test` runs the aligner's numeric tests.
- `setup_python.ps1` alone rebuilds `python\` (embeddable CPython 3.11.9 + `requirements.txt`,
  `sys.path` pinned inside the folder so no host Python can shadow it), e.g. after editing the pins.
- `package.ps1` assembles the release zip in `dist\`, which is what CI does on a `v*` tag
  (`.github/workflows/build.yml`; plain pushes just build and smoke-test).

The repo layout is upstream's, so a diff against yassa9/dvlt.cu stays readable:

```
dvlt.cu  platform_compat.cpp  include\  kernels\   the engine (upstream + port)
tools\convert.cpp                                  safetensors -> bf16 blob converter
tools\viewer.html                                  point cloud + frustum viewer
reconstruct.bat  run.bat                           wrappers
dvlt_long.py  dvlt_postprocess.py                  chunk aligner, post-processor
setup.bat  build.bat  package.ps1                  fetch / build / bundle
```

`setup.bat` adds `cublasLt64_12.dll`, `model\weights.dvlt` (+ its license) and `python\`
(embeddable CPython 3.11 with numpy, scipy, pillow, trimesh); a release zip is that same folder
with `inputs\shoe\`. Every runtime dependency lives inside it: nothing is installed, nothing on
PATH is changed.

## What the port changes

Based on yassa9/dvlt.cu at `440a9f5` (2026-06-04). Kernels and numerics are upstream's, untouched.

- **Windows**: `std::filesystem` replaces `dirent`/`mkdir`; weight loading goes through
  `platform_compat` (`CreateFileMapping` on Windows, `mmap` elsewhere) so `<windows.h>` never enters
  the CUDA translation unit; static CRT and static cudart, so the only runtime DLL is cuBLASLt.
- **`-k <N>`** exposes the loop count (the paper's inference-time compute knob; default 12).
- **`-x`** writes `conf.bin` (per-pixel confidence) and `desc.bin` (per-frame DINOv2 descriptor)
  next to `scene.ply`; **`-e`** runs the encoder alone over any number of frames at near-constant
  VRAM. Both feed `dvlt_long.py`.
- **List-file input** (`views.txt`): explicit ordering, `#` comments, BOM-tolerant, relative paths
  resolve against the list, and it sidesteps the Windows command-line length cap for 1000+ inputs.
- **EXIF orientation** is applied on load; a batch mixing rotated phone photos otherwise shatters.
- **Bounded decode memory**: image decoding does not grow with the number of photos in a list.
- **Loud failures**: a missing or unloadable cuBLASLt, or a GPU below Ampere, stops with a message
  naming it.
- The converter builds with MSVC (same `platform_compat` shim).

## Model limits worth knowing in the field

- **14-px square tiling in the cloud is inherent to the model**, not a port bug: world point =
  ray_origin + depth·ray_dir, and the ray head is a linear pixel-shuffle over 14×14 DINOv2 patches
  (paper §3.3, App. B; depth got a conv head to kill these artifacts, rays did not). Error scales with
  scene depth, so it is most visible on drone footage. Mitigate with `conf_drop 0.3–0.5`, or treat the
  cloud as a 3DGS/NeRF initializer (training absorbs it).
- Trained with **2–18 views**; more views work but are out-of-distribution. Prefer chunking over
  resolution drops for large sets.
- Global attention is the VRAM wall (~80 frames @ 504 px on 16 GB). Measured: 36 frames @ 504×336
  = 6.0 GB, 80 frames = 11.6 GB, i.e. a fixed base plus a near-constant cost per frame, which is
  why `reconstruct.bat` scales its limits with VRAM minus ~2 GB.
- Confidence is per-pixel from the depth head; `-c` filtering never creates square holes.

## License

- **Code**: Apache License 2.0 ([LICENSE](LICENSE)), upstream dvlt.cu and this port alike. See
  [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) for CUTLASS, stb_image and the reimplemented
  algorithms.
- **Model weights** (`model\weights.dvlt` in release zips): NVIDIA's `nvidia/dvlt` checkpoint under
  the [NVIDIA License](https://huggingface.co/nvidia/dvlt/blob/main/LICENSE.txt), which permits
  redistribution with its text attached (`model\LICENSE.txt`) and restricts use to
  **non-commercial research and evaluation**. They are not in this repository; `setup.bat` fetches
  them, with that license, from NVIDIA.
- **cuBLASLt** (`cublasLt64_12.dll`): NVIDIA CUDA Toolkit EULA, a listed redistributable runtime
  library.
