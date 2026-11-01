#pragma once
// dvlt conv depth head, channels-last (nhwc) so the 3x3 convs run as cutlass implicit-gemm
// with no per-conv layout transposes (only nchw<->nhwc at entry/exit). tensors are [B,H,W,C].

#include <cmath>
#include <vector>
#include <algorithm>
#include "../include/arena.h"
#include "../include/gemm.h"
#include "../include/conv_cutlass.cuh"

namespace dvlt {

// layout transforms (head boundary only). intermediates a_/b_/c_ are bf16; casts fp32<->bf16 here.
__global__ void nchw2nhwc_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                 int B, int C, int H, int W) {
    size_t total = (size_t)B * C * H * W;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int x = idx % W, y = (idx / W) % H, c = (idx / ((size_t)W * H)) % C, b = idx / ((size_t)W * H * C);
    out[(((size_t)b * H + y) * W + x) * C + c] = __float2bfloat16(in[idx]);
}

// transpose conv k2s2 as a gemm: 4 taps stacked into one Y[M,4*Cout] = aug[M,K] @ Wm[K,4*Cout]
// (M=B*H*W pixels, K=Cprev+2 uv channels, padded to mult-4 for tf32), then a coalesced 2x scatter.

// in [B,H,W,Cprev] -> aug [M, Kp] (M=B*H*W), channels [0,Cprev) copied, Cprev=u, Cprev+1=v, rest 0.
__global__ void convT_build_aug_kernel(const __nv_bfloat16* __restrict__ in, float* __restrict__ aug,
                                       int M, int Cprev, int Kp, int W, int H, float sx, float sy) {
    size_t total = (size_t)M * Kp;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int k = idx % Kp; int m = idx / Kp;
    if (k < Cprev) { aug[idx] = __bfloat162float(in[(size_t)m * Cprev + k]); return; }
    if (k == Cprev)     { int iw = m % W; aug[idx] = sx * (float)(2 * iw - (W - 1)) / (float)W; return; }
    if (k == Cprev + 1) { int ih = (m / W) % H; aug[idx] = sy * (float)(2 * ih - (H - 1)) / (float)H; return; }
    aug[idx] = 0.f;
}

// filter [Cprev+2,Cout,2,2] -> Wm [4*Cout, Kp] (column-major [Kp,4*Cout] for gemm_f32). column
// n=t*Cout+oc, row k: Wm[n*Kp+k] = wt[(k*Cout+oc)*4 + t] for k<Cprev+2, else 0 (t=ky*2+kx).
__global__ void convT_permute_w_kernel(const float* __restrict__ wt, float* __restrict__ Wm,
                                       int Cprev, int Cout, int Kp) {
    int N = 4 * Cout; size_t total = (size_t)N * Kp;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int k = idx % Kp; int n = idx / Kp; int oc = n % Cout, t = n / Cout;
    Wm[idx] = (k < Cprev + 2) ? wt[(((size_t)k * Cout + oc) * 4) + t] : 0.f;
}

// Y [M, 4*Cout] -> out [B,2H,2W,Cout] + bias. one thread per output element, oc innermost
// (coalesced on both the Y read and the out write). tap t = (oh&1)*2 + (ow&1).
__global__ void convT_scatter_kernel(const float* __restrict__ Y, const float* __restrict__ bias,
                                     __nv_bfloat16* __restrict__ out, int B, int Cout, int H, int W) {
    int Ho = 2 * H, Wo = 2 * W; size_t total = (size_t)B * Ho * Wo * Cout;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int oc = idx % Cout; size_t s = idx / Cout;
    int ow = s % Wo, oh = (s / Wo) % Ho, b = s / ((size_t)Wo * Ho);
    int ih = oh >> 1, iw = ow >> 1, t = (oh & 1) * 2 + (ow & 1);
    size_t m = ((size_t)b * H + ih) * W + iw;
    out[idx] = __float2bfloat16(Y[m * (size_t)(4 * Cout) + (size_t)t * Cout + oc] + (bias ? bias[oc] : 0.f));
}

// groupnorm nhwc, two-pass multi-block: GN_SPLIT blocks per (b,g) accumulate partial sum/sumsq
// (atomics), a second kernel normalizes.
static constexpr int GN_SPLIT = 16;

// `bias` (optional) folds the producing conv's bias in here, so it sees (conv_out + bias).
__global__ void gn_stats_nhwc_kernel(const __nv_bfloat16* __restrict__ in, const float* __restrict__ bias,
                                     float* __restrict__ sum, float* __restrict__ sqsum,
                                     int B, int C, int H, int W, int groups) {
    int blk = blockIdx.x, sp = blk % GN_SPLIT, bg = blk / GN_SPLIT;
    int g = bg % groups, b = bg / groups, cg = C / groups;
    size_t HW = (size_t)H * W, n = (size_t)cg * HW, base = (size_t)b * HW * C + g * cg;
    size_t lo = n * sp / GN_SPLIT, hi = n * (sp + 1) / GN_SPLIT;
    int lg = __ffs(cg) - 1; size_t mask = cg - 1;   // cg is a power of 2 -> div/mod become shift/and
    __shared__ float sh[256], sh2[256];
    float s = 0.f, s2 = 0.f;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = __bfloat162float(in[base + (i >> lg) * C + (i & mask)]); if (bias) v += bias[g * cg + (i & mask)];
        s += v; s2 += v * v;
    }
    sh[threadIdx.x] = s; sh2[threadIdx.x] = s2; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o) { sh[threadIdx.x] += sh[threadIdx.x + o]; sh2[threadIdx.x] += sh2[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&sum[bg], sh[0]); atomicAdd(&sqsum[bg], sh2[0]); }
}

// gn apply, fusing optional conv-bias (read) + relu (write). when `padded`, writes into the interior
// of a [B,H+2,W+2,C] buffer (the next conv's pre-padded input, bf16) so the conv skips its pad-copy.
// OT picks fp32 (plain) or bf16 (padded) output; cast_out rounds only on bf16.
__device__ __forceinline__ float cast_out(float v, float)          { return v; }
__device__ __forceinline__ __nv_bfloat16 cast_out(float v, __nv_bfloat16) { return __float2bfloat16(v); }
// 3d grid (blockIdx = w,h,b) so the (b,h,w) unpack is free (no per-element div/mod); threads stride c.
template<class OT>
__global__ void gn_apply_nhwc_kernel(const __nv_bfloat16* __restrict__ in, const float* __restrict__ bias,
                                     const float* __restrict__ sum, const float* __restrict__ sqsum,
                                     const float* __restrict__ gamma, const float* __restrict__ beta,
                                     OT* __restrict__ out, int B, int C, int H, int W, int groups,
                                     float eps, bool relu, bool padded) {
    int w = blockIdx.x, h = blockIdx.y, b = blockIdx.z, cg = C / groups, Wp = W + 2;
    float n = (float)cg * H * W;
    size_t sp_in  = ((size_t)(b * H + h) * W + w) * C;
    size_t sp_out = padded ? ((size_t)(b * (H + 2) + (h + 1)) * Wp + (w + 1)) * C : sp_in;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        int bg = b * groups + c / cg;
        float mean = sum[bg] / n, var = sqsum[bg] / n - mean * mean;
        float v = __bfloat162float(in[sp_in + c]); if (bias) v += bias[c];
        v = (v - mean) * rsqrtf(var + eps) * gamma[c] + beta[c];
        if (relu) v = fmaxf(v, 0.f);
        out[sp_out + c] = cast_out(v, OT{});
    }
}

// replicate-fill the 1px border of a [B,H+2,W+2,C] bf16 buffer whose interior [1..H,1..W] is set.
// one thread per (b, ring-pixel, c); ring length = 2(H+2)+2(W+2)-4.
__global__ void pad_border_fill_nhwc_kernel(__nv_bfloat16* __restrict__ pad, int B, int C, int H, int W) {
    int Hp = H + 2, Wp = W + 2, P = 2 * Hp + 2 * Wp - 4;
    size_t total = (size_t)B * P * C;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C; size_t t = idx / C; int ring = t % P, b = t / P;
    int yp, xp;
    if (ring < Wp) { yp = 0; xp = ring; }
    else if (ring < 2 * Wp) { yp = Hp - 1; xp = ring - Wp; }
    else if (ring < 2 * Wp + (Hp - 2)) { yp = 1 + (ring - 2 * Wp); xp = 0; }
    else { yp = 1 + (ring - 2 * Wp - (Hp - 2)); xp = Wp - 1; }
    int iy = yp - 1; iy = iy < 0 ? 0 : (iy >= H ? H - 1 : iy);    // clamp to interior, then +1 offset
    int ix = xp - 1; ix = ix < 0 ? 0 : (ix >= W ? W - 1 : ix);
    pad[(((size_t)b * Hp + yp) * Wp + xp) * C + c] =
        pad[(((size_t)b * Hp + (iy + 1)) * Wp + (ix + 1)) * C + c];
}

// residual add fusing the producing conv's bias: io += t + bias[c]. 3d grid (w,h,b), threads over c.
__global__ void add_bias_nhwc_kernel(__nv_bfloat16* __restrict__ io, const __nv_bfloat16* __restrict__ t,
                                     const float* __restrict__ bias, int C, int H, int W) {
    int w = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    size_t base = ((size_t)(b * H + h) * W + w) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        io[base + c] = __float2bfloat16(__bfloat162float(io[base + c]) + __bfloat162float(t[base + c]) + bias[c]);
}
// conv bias + relu fused (for the output block's 66->32 conv -> relu).
__global__ void bias_relu_nhwc_kernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ bias, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(fmaxf(__bfloat162float(x[i]) + bias[i % C], 0.f));
}

// bilinear resize (align_corners=false) + uv-concat fused: [B,Hi,Wi,Cprev] -> [B,Ho,Wo,Cprev+2].
__global__ void bilinear_concat_uv_nhwc_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                               int B, int Cprev, int Hi, int Wi, int Ho, int Wo,
                                               float sx, float sy) {
    int Co = Cprev + 2; size_t total = (size_t)B * Ho * Wo * Co;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % Co; size_t t = idx / Co;
    int ox = t % Wo, oy = (t / Wo) % Ho, b = t / ((size_t)Wo * Ho);
    if (c == Cprev)      { out[idx] = __float2bfloat16(sx * (float)(2 * ox - (Wo - 1)) / (float)Wo); return; }
    if (c == Cprev + 1)  { out[idx] = __float2bfloat16(sy * (float)(2 * oy - (Ho - 1)) / (float)Ho); return; }
    float fy = ((float)oy + 0.5f) * (float)Hi / (float)Ho - 0.5f;
    float fx = ((float)ox + 0.5f) * (float)Wi / (float)Wo - 0.5f;
    int y0 = (int)floorf(fy), x0 = (int)floorf(fx); float wy = fy - y0, wx = fx - x0;
    int y0c = y0 < 0 ? 0 : (y0 >= Hi ? Hi - 1 : y0), y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= Hi ? Hi - 1 : y0 + 1);
    int x0c = x0 < 0 ? 0 : (x0 >= Wi ? Wi - 1 : x0), x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= Wi ? Wi - 1 : x0 + 1);
    const __nv_bfloat16* ip = in + (size_t)b * Hi * Wi * Cprev;
    float v00 = __bfloat162float(ip[((size_t)y0c * Wi + x0c) * Cprev + c]), v01 = __bfloat162float(ip[((size_t)y0c * Wi + x1c) * Cprev + c]);
    float v10 = __bfloat162float(ip[((size_t)y1c * Wi + x0c) * Cprev + c]), v11 = __bfloat162float(ip[((size_t)y1c * Wi + x1c) * Cprev + c]);
    out[idx] = __float2bfloat16((v00 * (1 - wx) + v01 * wx) * (1 - wy) + (v10 * (1 - wx) + v11 * wx) * wy);
}

// conv 1x1, nhwc in -> NCHW out (fuses the final layout transpose). in [B,H,W,Cin], weight
// [Cout,Cin], out [B,Cout,H,W].
__global__ void conv1x1_nhwc2nchw_kernel(const __nv_bfloat16* __restrict__ in, const float* __restrict__ wt,
                                         const float* __restrict__ bias, float* __restrict__ out,
                                         int B, int Cin, int Cout, int H, int W) {
    size_t total = (size_t)B * H * W * Cout;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // nhwc iteration
    if (idx >= total) return;
    int oc = idx % Cout; size_t t = idx / Cout;
    int w = t % W, h = (t / W) % H, b = t / ((size_t)W * H);
    const __nv_bfloat16* ip = in + t * Cin; const float* wp = wt + (size_t)oc * Cin;
    float acc = bias ? bias[oc] : 0.f;
    for (int ic = 0; ic < Cin; ic++) acc += __bfloat162float(ip[ic]) * wp[ic];
    out[(((size_t)b * Cout + oc) * H + h) * W + w] = acc;
}

__global__ void bias_nhwc_kernel(__nv_bfloat16* __restrict__ out, const float* __restrict__ bias, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(out[i]) + bias[i % C]);
}

// replicate-pad nhwc, optionally widening to Cpad channels (extra channels zeroed). in [B,H,W,Cin]
// -> pad [B,H+2,W+2,Cpad] bf16. lets the cutlass bf16 path (needs Cin%8==0) handle any Cin via Cpad.
__global__ void pad1_nhwc_widen_kernel(const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ pad,
                                       int B, int Cin, int Cpad, int H, int W) {
    int Hp = H + 2, Wp = W + 2;                          // 3d grid: blockIdx = xp,yp,b; threads over c
    int xp = blockIdx.x, yp = blockIdx.y, b = blockIdx.z;
    int y = yp - 1; y = y < 0 ? 0 : (y >= H ? H - 1 : y);   // replicate
    int x = xp - 1; x = x < 0 ? 0 : (x >= W ? W - 1 : x);
    size_t ob = ((size_t)(b * Hp + yp) * Wp + xp) * Cpad, ib = ((size_t)(b * H + y) * W + x) * Cin;
    for (int c = threadIdx.x; c < Cpad; c += blockDim.x)
        pad[ob + c] = (c < Cin) ? in[ib + c] : __float2bfloat16(0.f);
}
// permute filter [Cout,Cin,3,3] -> [Cout,3,3,Cpad] bf16, extra in-channels zeroed.
__global__ void permute_krsc_widen_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out,
                                          int Cout, int Cin, int Cpad) {
    size_t total = (size_t)Cout * Cpad * 9;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ci = idx % Cpad; size_t t = idx / Cpad; int rs = t % 9; int co = t / 9;
    out[idx] = __float2bfloat16((ci < Cin) ? in[((size_t)co * Cin + ci) * 9 + rs] : 0.f);
}

// weights (moge-style conv depth head, fp32 weights)
struct ResBlockWeights {
    const float *gn1_w, *gn1_b;          // GroupNorm(1, c)
    const float *conv1_w, *conv1_b;      // Conv2d(c, hidden, 3)
    const float *gn2_w, *gn2_b;          // GroupNorm(hidden/32, hidden)
    const float *conv2_w, *conv2_b;      // Conv2d(hidden, c, 3)
};
struct UpsampleWeights {
    const float *convT_w, *convT_b;      // ConvTranspose2d(in+2, out, 2, 2)
    const float *conv_w, *conv_b;        // Conv2d(out, out, 3)
    ResBlockWeights res[2];
};
struct ConvDepthHeadWeights {
    UpsampleWeights up[3];
    const float *out_conv1_w, *out_conv1_b;   // Conv2d(64+2, 32, 3)
    const float *out_conv2_w, *out_conv2_b;   // Conv2d(32, 2, 1)
};

struct ConvDepthHeadNHWC {
    int B_, ph_, pw_, H_, W_, outc_;
    __nv_bfloat16 *a_, *b_, *c_;  // nhwc ping-pong (bf16, sized to the largest intermediate)
    __nv_bfloat16 *pad_, *filt_;  // cutlass conv scratch (bf16): replicate-padded input + filter
    float *gn_sum_, *gn_sqsum_;   // groupnorm per-(b,g) reduction accumulators
    void* ws_;
    CublasCtx* cb_;

    static int up8(int c) { return (c + 7) & ~7; }   // bf16 conv wants Cin a multiple of 8
    static int up4(int c) { return (c + 3) & ~3; }   // convT gemm k (tf32) wants K a multiple of 4

    void init(CublasCtx& cb, GpuArena& arena, int B, int ph, int pw, int H, int W, int outc) {
        cb_ = &cb; B_ = B; ph_ = ph; pw_ = pw; H_ = H; W_ = W; outc_ = outc;
        size_t cap = (size_t)B * std::max(std::max((size_t)512 * (2*ph) * (2*pw),
                                                   (size_t)256 * (4*ph) * (4*pw)),
                                          std::max((size_t)128 * (8*ph) * (8*pw),
                                                   (size_t)68 * H * W));
        a_ = arena.alloc<__nv_bfloat16>(cap); b_ = arena.alloc<__nv_bfloat16>(cap); c_ = arena.alloc<__nv_bfloat16>(cap);
        // pad (bf16): largest padded conv input over stages. also reinterpreted as fp32 scratch for
        // the convT gemm's aug (the bf16 pad byte-budget covers it).
        size_t pad_cap = std::max({(size_t)B*(2*ph+2)*(2*pw+2)*512,
                                   (size_t)B*(4*ph+2)*(4*pw+2)*256,
                                   (size_t)B*(8*ph+2)*(8*pw+2)*128,
                                   (size_t)B*(H+2)*(W+2)*72});
        pad_  = arena.alloc<__nv_bfloat16>(pad_cap);
        filt_ = arena.alloc<__nv_bfloat16>((size_t)512 * 512 * 9);
        ws_   = arena.alloc<float>((size_t)16 * 1024 * 1024 / 4);
        gn_sum_   = arena.alloc<float>((size_t)B * 16);    // max groups = 512/32
        gn_sqsum_ = arena.alloc<float>((size_t)B * 16);
    }

    // two-pass groupnorm, nhwc, fusing conv-bias + relu. padded=false writes fp32 to `out`;
    // padded=true writes bf16 into pad_'s interior + border fill (next conv skips its pad-copy).
    void groupnorm(const __nv_bfloat16* in, const float* bias, const float* gamma, const float* beta,
                   __nv_bfloat16* out, int C, int H, int W, int groups, bool relu, bool padded = false) {
        const int th = 256;
        cudaMemsetAsync(gn_sum_,   0, (size_t)B_ * groups * 4);
        cudaMemsetAsync(gn_sqsum_, 0, (size_t)B_ * groups * 4);
        gn_stats_nhwc_kernel<<<B_ * groups * GN_SPLIT, 256>>>(in, bias, gn_sum_, gn_sqsum_, B_, C, H, W, groups);
        dim3 grid(W, H, B_); int blk = C < 256 ? C : 256;   // one block per (b,h,w), threads over c
        if (padded) {
            gn_apply_nhwc_kernel<<<grid, blk>>>(in, bias, gn_sum_, gn_sqsum_, gamma, beta, pad_, B_, C, H, W, groups, 1e-5f, relu, true);
            int P = 2*(H+2) + 2*(W+2) - 4;
            pad_border_fill_nhwc_kernel<<<((size_t)B_*P*C + th-1)/th, th>>>(pad_, B_, C, H, W);
        } else {
            gn_apply_nhwc_kernel<<<grid, blk>>>(in, bias, gn_sum_, gn_sqsum_, gamma, beta, out, B_, C, H, W, groups, 1e-5f, relu, false);
        }
    }

    // pad_ * filt_ -> out via the cutlass core. fatal on failure: cutlass consumes the launch
    // error itself (cudaGetLastError), so the per-stage check in dvlt.cu would never see it.
    void conv_core(__nv_bfloat16* out, int Cin, int Cout, int H, int W) {
        cutlass::Status st = dvltconv::conv3x3_cutlass_core_bf16(pad_, filt_, out, B_, Cin, Cout, H, W, ws_);
        if (st != cutlass::Status::kSuccess) {
            fprintf(stderr, "cutlass conv3x3 failed: %s (B=%d Cin=%d Cout=%d %dx%d)\n",
                    cutlass::cutlassGetStatusString(st), B_, Cin, Cout, W, H);
            exit(1);
        }
    }

    // conv whose input is already in pad_ (written by a preceding padded groupnorm): permute the
    // filter + run the padding=0 bf16 cutlass core.
    void conv3x3_prepadded(const float* wt, __nv_bfloat16* out, int Cin, int Cout, int H, int W) {
        const int th = 256;
        permute_krsc_widen_kernel<<<((size_t)Cout*Cin*9 + th-1)/th, th>>>(wt, filt_, Cout, Cin, Cin);
        conv_core(out, Cin, Cout, H, W);
    }

    // transpose conv k2s2 (uv-concat fused) via gemm (fp32/tf32): in [B,H,W,Cprev] -> out [B,2H,2W,Cout].
    // pad_/filt_/c_ are free here, reinterpreted as the fp32 aug/Wm/Y scratch.
    void convT_uv_gemm(const __nv_bfloat16* in, const float* wt, const float* bias, __nv_bfloat16* out,
                       int Cprev, int Cout, int H, int W, float sx, float sy) {
        const int th = 256; int M = B_ * H * W, Kp = up4(Cprev + 2), N = 4 * Cout;
        float* aug = reinterpret_cast<float*>(pad_);
        float* Wm  = reinterpret_cast<float*>(filt_);
        float* Y   = reinterpret_cast<float*>(c_);
        convT_build_aug_kernel<<<((size_t)M*Kp + th-1)/th, th>>>(in, aug, M, Cprev, Kp, W, H, sx, sy);
        convT_permute_w_kernel<<<((size_t)N*Kp + th-1)/th, th>>>(wt, Wm, Cprev, Cout, Kp);
        gemm_f32(*cb_, aug, Wm, Y, M, Kp, N);     // Y[M,N] = aug[M,Kp] @ Wm[Kp,N]
        convT_scatter_kernel<<<((size_t)B_*(2*H)*(2*W)*Cout + th-1)/th, th>>>(Y, bias, out, B_, Cout, H, W);
    }

    void span(int H, int W, float& sx, float& sy) {
        float ar = (float)W / (float)H; sx = ar / sqrtf(1 + ar * ar); sy = 1.f / sqrtf(1 + ar * ar);
    }

    // 3x3 stride1 replicate conv, nhwc, cutlass bf16. when bias is null the caller folds it into
    // the consumer (groupnorm/add).
    void conv3x3(const __nv_bfloat16* in, const float* wt, const float* bias, __nv_bfloat16* out,
                 int Cin, int Cout, int H, int W) {
        const int th = 256; int Cp = up8(Cin);
        pad1_nhwc_widen_kernel<<<dim3(W+2, H+2, B_), Cp < 256 ? Cp : 256>>>(in, pad_, B_, Cin, Cp, H, W);
        permute_krsc_widen_kernel<<<((size_t)Cout*Cp*9 + th-1)/th, th>>>(wt, filt_, Cout, Cin, Cp);
        conv_core(out, Cp, Cout, H, W);
        if (bias) bias_nhwc_kernel<<<((size_t)B_*H*W*Cout + th-1)/th, th>>>(out, bias, Cout, (size_t)B_*H*W*Cout);
    }

    // resblock: io += conv2(relu(gn2(conv1(relu(gn1(io)))))). relu folds into each gn; conv biases
    // fold into the next gn / the residual add.
    void resblock(const ResBlockWeights& w, __nv_bfloat16* io, int c, int hidden, int H, int W,
                  __nv_bfloat16* t2) {
        // each gn writes its relu'd output straight into pad_, so conv3x3_prepadded skips the pad-copy.
        groupnorm(io, nullptr, w.gn1_w, w.gn1_b, nullptr, c, H, W, 1, /*relu=*/true, /*padded=*/true);
        conv3x3_prepadded(w.conv1_w, t2, c, hidden, H, W);                    // bias -> gn2
        groupnorm(t2, w.conv1_b, w.gn2_w, w.gn2_b, nullptr, hidden, H, W, hidden/32, /*relu=*/true, /*padded=*/true);
        conv3x3_prepadded(w.conv2_w, t2, hidden, c, H, W);                    // bias -> add
        add_bias_nhwc_kernel<<<dim3(W, H, B_), c < 256 ? c : 256>>>(io, t2, w.conv2_b, c, H, W);
    }

    // x_in [B,in_ch,ph,pw] NCHW -> out [B,outc,H,W] NCHW. (boundary transposes in/out.)
    void fwd(const ConvDepthHeadWeights& w, const float* x_in, int in_ch, float* out) {
        const int th = 256; const int dims[3] = {256, 128, 64};
        int Cprev = in_ch, Hc = ph_, Wc = pw_;
        // nchw -> nhwc into a_
        nchw2nhwc_kernel<<<((size_t)B_*in_ch*ph_*pw_ + th-1)/th, th>>>(x_in, a_, B_, in_ch, ph_, pw_);

        for (int s = 0; s < 3; s++) {
            int Cout = dims[s], hidden = 2 * Cout;
            float sx, sy; span(Hc, Wc, sx, sy);
            int Ho = 2*Hc, Wo = 2*Wc;
            // convT with uv-concat fused (no separate concat pass). a_ -> b_, then conv3x3 b_ -> a_.
            convT_uv_gemm(a_, w.up[s].convT_w, w.up[s].convT_b, b_, Cprev, Cout, Hc, Wc, sx, sy);
            conv3x3(b_, w.up[s].conv_w, w.up[s].conv_b, c_, Cout, Cout, Ho, Wo);
            std::swap(a_, c_);
            Hc = Ho; Wc = Wo; Cprev = Cout;
            resblock(w.up[s].res[0], a_, Cout, hidden, Hc, Wc, c_);
            resblock(w.up[s].res[1], a_, Cout, hidden, Hc, Wc, c_);
        }

        // bilinear-to-(H,W) + uv-concat fused (no bilinear intermediate). a_ -> b_ [B,H,W,Cprev+2].
        float sx, sy; span(H_, W_, sx, sy);
        bilinear_concat_uv_nhwc_kernel<<<((size_t)B_*(Cprev+2)*H_*W_ + th-1)/th, th>>>(a_, b_, B_, Cprev, Hc, Wc, H_, W_, sx, sy);
        conv3x3(b_, w.out_conv1_w, nullptr, c_, Cprev+2, 32, H_, W_);           // 66->32 (widened to 72)
        bias_relu_nhwc_kernel<<<((size_t)B_*32*H_*W_ + th-1)/th, th>>>(c_, w.out_conv1_b, 32, (size_t)B_*32*H_*W_);
        // conv1x1 writes NCHW directly (fuses the final transpose).
        conv1x1_nhwc2nchw_kernel<<<((size_t)B_*outc_*H_*W_ + th-1)/th, th>>>(c_, w.out_conv2_w, w.out_conv2_b, out, B_, 32, outc_, H_, W_);
    }
};

} // namespace dvlt
