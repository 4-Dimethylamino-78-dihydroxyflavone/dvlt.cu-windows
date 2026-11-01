# Third-Party Notices and Licenses

dvlt.cu is a bare-metal CUDA/C++ port of NVIDIA's DVLT (Déjà View Looping
Transformer). The port itself is licensed under the Apache License, Version 2.0
(see LICENSE). This file lists the third-party code it bundles or builds
against, and the upstream projects whose algorithms it reimplements.

The DVLT model weights are NOT part of this project; see NOTICE.

## 1. Bundled / linked components

Redistributed with, or fetched and linked by, this project.

### CUTLASS — NVIDIA, BSD-3-Clause

Fetched by `setup.sh` into `ext/cutlass`; used for the fused multi-head
attention kernel. Full license text:

```
Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### stb_image — Sean Barrett, MIT OR Public Domain (dual-licensed)

Vendored at `include/stb_image.h`; used for image decoding in preprocessing.
Full license text (use whichever alternative you prefer):

```
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/
```

### cuBLASLt — NVIDIA, CUDA Toolkit EULA

`cublasLt64_12.dll`, the cuBLAS Lt runtime, is the one CUDA library `dvlt.exe`
links dynamically (cudart is static). Release archives ship it next to the exe;
it is one of the runtime libraries the NVIDIA CUDA Toolkit End User License
Agreement lists as redistributable (Attachment A). License:
https://docs.nvidia.com/cuda/eula/index.html

## 2. Adapted algorithm lineage (reimplemented, not copied)

This port reimplements DVLT's inference path. DVLT in turn adapts code from the
projects below; the corresponding algorithms are reimplemented here from DVLT's
Apache-2.0 sources — no upstream source code is copied into this repository.
License designations are as recorded in DVLT's `THIRD_PARTY_LICENSES.md`, which
is the authoritative per-file attribution map.

| Component | Upstream | License |
|---|---|---|
| DINOv2 ViT backbone (image encoder) | github.com/facebookresearch/dinov2 | Apache-2.0 |
| PyTorch3D (quaternion to rotation matrix) | github.com/facebookresearch/pytorch3d | BSD-3-Clause |
| MoGe (normalized UV grid) | github.com/microsoft/MoGe | MIT |
| MultiNeRF (camera utilities) | github.com/google-research/multinerf | Apache-2.0 |
| Depth-Anything-3 (ray utilities) | github.com/ByteDance-Seed/Depth-Anything-3 | Apache-2.0 |
| AnyCalib (camera manifold) | github.com/javrtg/AnyCalib | Apache-2.0 |

Original work: DVLT — Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES,
licensed under the Apache License, Version 2.0.

Note: DVLT also contains four VGGT-licensed files, but they are training/loss
code only and are NOT part of this inference port, so the VGGT License does not
apply here.
