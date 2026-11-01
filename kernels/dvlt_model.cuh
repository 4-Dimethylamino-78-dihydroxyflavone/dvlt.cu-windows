#pragma once

// full dvlt inference pipeline: encoder -> token assembly -> 12-step looping core ->
// ray/depth/camera heads -> activations -> world points. transformer bf16, output heads fp32.

#include <vector>
#include <random>
#include <numeric>
#include <cstring>
#include "../include/platform_compat.h"   // cross-platform read-only file mapping (mmap / Win32)
#include "../include/arena.h"
#include "../include/cuda_util.h"
#include "dinov2_encoder.cuh"
#include "looped_aa.cuh"
#include "decoder_head.cuh"
#include "conv_head_nhwc.cuh"
#include "ransac_pose.cuh"

namespace dvlt {

__global__ void fill_f32_kernel(float* __restrict__ p, float v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// assemble [camera(1), register(R), patch(Sp)] per frame. camera token: index 0 for the first
// frame of each batch, index 1 for the rest (dvlt _slice_expand_flatten).
__global__ void build_tokens_kernel(
    const __nv_bfloat16* __restrict__ z0,    // [BS, Sp, D]
    const __nv_bfloat16* __restrict__ cam,   // [2, D]
    const __nv_bfloat16* __restrict__ reg,   // [R, D]
    __nv_bfloat16* __restrict__ out,         // [BS, P, D]
    int B, int S, int Sp, int R, int D, int patch_start)
{
    int P = patch_start + Sp, BS = B * S, total = BS * P * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D, t = (idx / D) % P, bs = idx / (D * P);
    int s = bs % S;
    if (t == 0)              out[idx] = cam[(s == 0 ? 0 : 1) * D + d];
    else if (t < patch_start) out[idx] = reg[(t - 1) * D + d];
    else                      out[idx] = z0[((size_t)bs * Sp + (t - patch_start)) * D + d];
}

// activations + world points. ray identity, depth exp_clamped(15), conf exp+1.
// world = ray_origin(3:6) + ray_dir(0:3) * depth.
__global__ void finalize_kernel(
    const float* __restrict__ ray,        // [BS, 6, H, W]
    const float* __restrict__ depth_raw,  // [BS, 2, H, W]
    float* __restrict__ rays_cl,          // [BS, H, W, 6]
    float* __restrict__ depth,            // [BS, H, W]
    float* __restrict__ conf,             // [BS, H, W]
    float* __restrict__ world,            // [BS, H, W, 3]
    int BS, int H, int W)
{
    int total = BS * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w = idx % W, h = (idx / W) % H, b = idx / (W * H);
    size_t hw = (size_t)H * W;
    float r[6];
    for (int c = 0; c < 6; c++) r[c] = ray[((size_t)b * 6 + c) * hw + h * W + w];
    float dr0 = depth_raw[((size_t)b * 2 + 0) * hw + h * W + w];
    float dr1 = depth_raw[((size_t)b * 2 + 1) * hw + h * W + w];
    float d = expf(fminf(dr0, 15.f));
    float cf = expf(dr1) + 1.f;
    for (int c = 0; c < 6; c++) rays_cl[(size_t)idx * 6 + c] = r[c];
    depth[idx] = d; conf[idx] = cf;
    for (int k = 0; k < 3; k++) world[(size_t)idx * 3 + k] = r[3 + k] + r[k] * d;
}

// weights
struct DvltWeights {
    // encoder
    DinoV2EncoderWeights enc;
    DinoV2BlockWeights enc_blocks[24];   // up to 24; vit-b uses 12
    const __nv_bfloat16 *norm_ones, *norm_zeros;
    // tokens
    const __nv_bfloat16 *camera_token, *register_token;
    // loop (shared block)
    DepthScaledBlockWeights frame, global;
    IntervalGateWeights gframe, gglobal;
    // ray decoder
    DecoderHeadWeights ray;  DecoderBlockWeights ray_blocks[2];
    // depth decoder (front-end + conv stage)
    DecoderHeadWeights depth; DecoderBlockWeights depth_blocks[2];
    ConvDepthHeadWeights depth_conv;
    // camera head
    SimpleCameraHeadWeights cam;
    // native (un-interpolated) pos embed kept as f32 for the cli's grid interpolation
    std::vector<float> pos_native;
};

inline __nv_bfloat16* upload_bf16_from_f32(const float* h, size_t n) {
    std::vector<__nv_bfloat16> t(n);
    for (size_t i = 0; i < n; i++) t[i] = __float2bfloat16(h[i]);
    __nv_bfloat16* d; CUDA_CHECK(cudaMalloc(&d, n * 2)); cudaMemcpy(d, t.data(), n * 2, cudaMemcpyHostToDevice);
    return d;
}

// bulk-upload cursor over the packed mmap'd DVL1 blob (self-describing per-tensor
// [dtype u8][count u64][data]). all tensors land in one gpu buffer (a 256-aligned slice each)
// instead of ~288 cudaMallocs; dtype-mismatched records (only pos) fall back to a private alloc.
struct WCursorBulk {
    const uint8_t* p;
    uint8_t* bulk = nullptr;     // single device buffer; every tensor is a 256-aligned slice of it
    size_t off = 0;
    static size_t al(size_t b) { return (b + 255) & ~(size_t)255; }

    WCursorBulk(const uint8_t* base, uint8_t* buf) : p(base), bulk(buf) {}
    void next(int& is_bf16, size_t& cnt, const uint8_t*& data) {
        is_bf16 = p[0]; memcpy(&cnt, p + 1, 8); data = p + 9; p = data + cnt * (is_bf16 ? 2 : 4);
    }
    __nv_bfloat16* bf16(size_t n) {
        int bf; size_t cnt; const uint8_t* d; next(bf, cnt, d);
        if (bf) { auto* dev = (__nv_bfloat16*)(bulk + off); off += al(n * 2);
                  cudaMemcpyAsync(dev, d, n * 2, cudaMemcpyHostToDevice); return dev; }
        __nv_bfloat16* dev; CUDA_CHECK(cudaMalloc(&dev, n * 2)); std::vector<__nv_bfloat16> t(n);
        const float* f = (const float*)d; for (size_t i = 0; i < n; i++) t[i] = __float2bfloat16(f[i]);
        cudaMemcpy(dev, t.data(), n * 2, cudaMemcpyHostToDevice); return dev;
    }
    float* f32(size_t n) {
        int bf; size_t cnt; const uint8_t* d; next(bf, cnt, d);
        if (!bf) { auto* dev = (float*)(bulk + off); off += al(n * 4);
                   cudaMemcpyAsync(dev, d, n * 4, cudaMemcpyHostToDevice); return dev; }
        float* dev; CUDA_CHECK(cudaMalloc(&dev, n * 4)); std::vector<float> t(n); const uint16_t* s = (const uint16_t*)d;
        for (size_t i = 0; i < n; i++) { uint32_t b = (uint32_t)s[i] << 16; memcpy(&t[i], &b, 4); }
        cudaMemcpy(dev, t.data(), n * 4, cudaMemcpyHostToDevice); return dev;
    }
    std::vector<float> host(size_t n) {
        int bf; size_t cnt; const uint8_t* d; next(bf, cnt, d);
        std::vector<float> v(n);
        if (!bf) memcpy(v.data(), d, n * 4);
        else { const uint16_t* s = (const uint16_t*)d; for (size_t i = 0; i < n; i++) { uint32_t b = (uint32_t)s[i] << 16; memcpy(&v[i], &b, 4); } }
        return v;
    }
};

// mmap a DVL1 file and position a cursor at the first record (after the 8-byte magic+count header).
struct WeightFile {
    uint8_t* base = nullptr; size_t size = 0;
    uint32_t ntensors = 0;
    const uint8_t* records() const { return base + 8; }
    void open(const char* path) {
        platcompat::FileMap m = platcompat::map_readonly(path);   // mmap (posix) / file-mapping (win)
        base = (uint8_t*)m.data; size = m.size;
        if (memcmp(base, "DVL1", 4) != 0) { fprintf(stderr, "weights: bad magic in %s\n", path); exit(1); }
        memcpy(&ntensors, base + 4, 4);
    }
    // total 256-aligned stored bytes across all records (for sizing the single bulk gpu buffer).
    size_t total_bytes() const {
        const uint8_t* q = records(); size_t total = 0;
        for (uint32_t i = 0; i < ntensors; i++) {
            int is = q[0]; size_t cnt; memcpy(&cnt, q + 1, 8); q += 9 + cnt * (is ? 2 : 4);
            total += (cnt * (is ? 2 : 4) + 255) & ~(size_t)255;
        }
        return total;
    }
};

struct DvltConfig {
    int B, S, P, D, H, DH, R, patch_start;
    int enc_blocks, dec_dim, dec_heads, dec_depth;
    int Himg, Wimg, ph, pw, Sp;
    int loop_steps;
    int patch_size = 14;
    bool use_depth_conf_for_pose = false;   // dvlt default: pose weights = ones
};

// load weights in the canonical order produced by tools/convert.cpp.
template<class Cur>
inline void load_dvlt_weights_cur(DvltWeights& w, Cur& cur, const DvltConfig& c) {
    const int D = c.D, DH = c.DH, DD = c.dec_dim;

    // encoder
    w.enc.patch_w = cur.bf16((size_t)D * 3 * 14 * 14);
    w.enc.patch_b = cur.bf16(D);
    w.enc.cls     = cur.bf16(D);
    // keep native pos as f32 (for cli grid interpolation), upload bf16 for the native-grid path.
    w.pos_native  = cur.host((size_t)(1 + c.Sp) * D);
    w.enc.pos     = upload_bf16_from_f32(w.pos_native.data(), w.pos_native.size());
    w.enc.regs    = cur.bf16((size_t)c.R * D);
    for (int i = 0; i < c.enc_blocks; i++) {
        auto& b = w.enc_blocks[i];
        b.norm1_w=cur.bf16(D); b.norm1_b=cur.bf16(D);
        b.qkv_w=cur.bf16((size_t)3*D*D); b.qkv_b=cur.bf16(3*D);
        b.proj_w=cur.bf16((size_t)D*D); b.proj_b=cur.bf16(D);
        b.ls1_gamma=cur.bf16(D);
        b.norm2_w=cur.bf16(D); b.norm2_b=cur.bf16(D);
        b.fc1_w=cur.bf16((size_t)4*D*D); b.fc1_b=cur.bf16(4*D);
        b.fc2_w=cur.bf16((size_t)D*4*D); b.fc2_b=cur.bf16(D);
        b.ls2_gamma=cur.bf16(D);
    }
    w.enc.blocks = w.enc_blocks; w.enc.num_blocks = c.enc_blocks;
    { std::vector<float> ones(D,1.f), zeros(D,0.f);
      __nv_bfloat16* o; CUDA_CHECK(cudaMalloc(&o, D*2)); std::vector<__nv_bfloat16> ob(D);
      for (int i=0;i<D;i++) ob[i]=__float2bfloat16(1.f); cudaMemcpy(o, ob.data(), D*2, cudaMemcpyHostToDevice);
      __nv_bfloat16* z; CUDA_CHECK(cudaMalloc(&z, D*2)); std::vector<__nv_bfloat16> zb(D,__float2bfloat16(0.f));
      cudaMemcpy(z, zb.data(), D*2, cudaMemcpyHostToDevice);
      w.enc.norm_w=o; w.enc.norm_b=z; w.norm_ones=o; w.norm_zeros=z; }

    // tokens
    w.camera_token   = cur.bf16((size_t)2 * D);
    w.register_token = cur.bf16((size_t)c.R * D);

    // loop block: frame then global (each: depth-scaled block weights + gate)
    auto read_ds = [&](DepthScaledBlockWeights& b) {
        b.norm1_w=cur.bf16(D); b.norm1_b=cur.bf16(D);
        b.qkv_w=cur.bf16((size_t)3*D*D); b.qkv_b=cur.bf16(3*D);
        b.q_norm_w=cur.bf16(DH); b.q_norm_b=cur.bf16(DH);
        b.k_norm_w=cur.bf16(DH); b.k_norm_b=cur.bf16(DH);
        b.proj_w=cur.bf16((size_t)D*D); b.proj_b=cur.bf16(D);
        b.ls1=cur.bf16(D);
        b.norm2_w=cur.bf16(D); b.norm2_b=cur.bf16(D);
        b.fc1_w=cur.bf16((size_t)4*D*D); b.fc1_b=cur.bf16(4*D);
        b.fc2_w=cur.bf16((size_t)D*4*D); b.fc2_b=cur.bf16(D);
        b.ls2=cur.bf16(D);
    };
    auto read_gate = [&](IntervalGateWeights& g) {
        g.hidden=64; g.out_dim=3*D;
        g.w1=cur.host((size_t)64*128); g.b1=cur.host(64);
        g.w2=cur.host((size_t)3*D*64); g.b2=cur.host(3*D);
    };
    read_ds(w.frame);  read_gate(w.gframe);
    read_ds(w.global); read_gate(w.gglobal);

    // decoder block reader (dim DD, no layerscale)
    auto read_dec_block = [&](DecoderBlockWeights& b) {
        b.norm1_w=cur.bf16(DD); b.norm1_b=cur.bf16(DD);
        b.qkv_w=cur.bf16((size_t)3*DD*DD); b.qkv_b=cur.bf16(3*DD);
        b.q_norm_w=cur.bf16(DH); b.q_norm_b=cur.bf16(DH);
        b.k_norm_w=cur.bf16(DH); b.k_norm_b=cur.bf16(DH);
        b.proj_w=cur.bf16((size_t)DD*DD); b.proj_b=cur.bf16(DD);
        b.norm2_w=cur.bf16(DD); b.norm2_b=cur.bf16(DD);
        b.fc1_w=cur.bf16((size_t)4*DD*DD); b.fc1_b=cur.bf16(4*DD);
        b.fc2_w=cur.bf16((size_t)DD*4*DD); b.fc2_b=cur.bf16(DD);
    };

    // ray decoder (linear head, out 6)
    w.ray.proj_in_w=cur.bf16((size_t)DD*D); w.ray.proj_in_b=cur.bf16(DD);
    for (int i=0;i<c.dec_depth;i++) read_dec_block(w.ray_blocks[i]);
    w.ray.blocks=w.ray_blocks; w.ray.num_blocks=c.dec_depth;
    w.ray.norm_w=cur.bf16(DD); w.ray.norm_b=cur.bf16(DD);
    w.ray.head_w=cur.f32((size_t)6*14*14*DD); w.ray.head_b=cur.f32(6*14*14);

    // depth decoder (front-end + conv head)
    w.depth.proj_in_w=cur.bf16((size_t)DD*D); w.depth.proj_in_b=cur.bf16(DD);
    for (int i=0;i<c.dec_depth;i++) read_dec_block(w.depth_blocks[i]);
    w.depth.blocks=w.depth_blocks; w.depth.num_blocks=c.dec_depth;
    w.depth.norm_w=cur.bf16(DD); w.depth.norm_b=cur.bf16(DD);
    w.depth.head_w=nullptr; w.depth.head_b=nullptr;
    { int in_ch[3]={DD,256,128}, dims[3]={256,128,64};
      for (int s=0;s<3;s++){ int out=dims[s], hid=2*out; auto& u=w.depth_conv.up[s];
        u.convT_w=cur.f32((size_t)(in_ch[s]+2)*out*4); u.convT_b=cur.f32(out);
        u.conv_w=cur.f32((size_t)out*out*9); u.conv_b=cur.f32(out);
        for (int r=0;r<2;r++){ auto& rb=u.res[r];
          rb.gn1_w=cur.f32(out); rb.gn1_b=cur.f32(out);
          rb.conv1_w=cur.f32((size_t)hid*out*9); rb.conv1_b=cur.f32(hid);
          rb.gn2_w=cur.f32(hid); rb.gn2_b=cur.f32(hid);
          rb.conv2_w=cur.f32((size_t)out*hid*9); rb.conv2_b=cur.f32(out); } }
      w.depth_conv.out_conv1_w=cur.f32((size_t)32*66*9); w.depth_conv.out_conv1_b=cur.f32(32);
      w.depth_conv.out_conv2_w=cur.f32((size_t)2*32);    w.depth_conv.out_conv2_b=cur.f32(2); }

    // camera head
    w.cam.ln_w=cur.bf16(DD); w.cam.ln_b=cur.bf16(DD);
    w.cam.fc0_w=cur.bf16((size_t)DD*DD); w.cam.fc0_b=cur.bf16(DD);
    w.cam.fc1_w=cur.bf16((size_t)DD*DD); w.cam.fc1_b=cur.bf16(DD);
    w.cam.pose_w=cur.f32((size_t)9*DD); w.cam.pose_b=cur.f32(9);
}

// packed mmap DVL1 blob (tools/convert.cpp). bulk-uploads into ONE gpu buffer (1 cudaMalloc);
// each tensor is a 256-aligned slice of it.
inline void load_dvlt_weights(DvltWeights& w, const WeightFile& wf, const DvltConfig& c) {
    uint8_t* bulk = nullptr; CUDA_CHECK(cudaMalloc(&bulk, wf.total_bytes()));
    WCursorBulk cur(wf.records(), bulk);
    load_dvlt_weights_cur(w, cur, c);
    cudaDeviceSynchronize();   // finish the async host->device copies before use
}

// full pipeline. persistent (cross-stage) buffers are cudaMalloc'd; per-stage plans use the arena
// with reset between stages to bound peak memory.
struct DvltPipeline {
    DvltConfig c;
    GpuArena arena;
    CublasCtx* cublas;
    float *cos_d, *sin_d;
    int* pos_d;
    int max_len_;
    // persistent buffers
    __nv_bfloat16 *enc_out, *x, *ray_feat, *cls;
    float *ray_out, *depth_map, *depth_out, *pose;
    // outputs
    float *rays_cl, *depth, *conf, *world;
    // per-step interval gates (depend only on weights + fixed t schedule): built once, one upload
    float* gate_table = nullptr;          // [loop_steps * 6 * D]
    std::vector<float> gate_host;
    // ransac rays_to_pose (fitted extrinsics/intrinsics)
    RansacPose ransac;
    std::vector<int> ransac_rand_idx;     // host-precomputed sample draws (constant)
    float* pose_conf = nullptr;           // [BS*hw] confidence weights for the pose fit

    // arena_bytes == 0 -> size the arena to free vram (after the persistent buffers below are
    // allocated). a nonzero value forces an exact arena size (tests pass a fixed budget).
    void init(CublasCtx& cb, const DvltConfig& cfg, size_t arena_bytes, int max_len) {
        cublas = &cb; c = cfg; max_len_ = max_len;
        int BS = c.B * c.S, P = c.P, D = c.D, DD = c.dec_dim, Sp = c.Sp;
        size_t hw = (size_t)c.Himg * c.Wimg;
        CUDA_CHECK(cudaMalloc(&enc_out,  (size_t)BS*Sp*D*2));
        CUDA_CHECK(cudaMalloc(&x,        (size_t)BS*P*D*2));
        CUDA_CHECK(cudaMalloc(&ray_feat, (size_t)BS*P*DD*2));
        CUDA_CHECK(cudaMalloc(&cls,      (size_t)BS*DD*2));
        CUDA_CHECK(cudaMalloc(&ray_out,  (size_t)BS*6*hw*4));
        CUDA_CHECK(cudaMalloc(&depth_map,(size_t)BS*DD*Sp*4));
        CUDA_CHECK(cudaMalloc(&depth_out,(size_t)BS*2*hw*4));
        CUDA_CHECK(cudaMalloc(&pose,     (size_t)BS*9*4));
        CUDA_CHECK(cudaMalloc(&rays_cl,  (size_t)BS*hw*6*4));
        CUDA_CHECK(cudaMalloc(&depth,    (size_t)BS*hw*4));
        CUDA_CHECK(cudaMalloc(&conf,     (size_t)BS*hw*4));
        CUDA_CHECK(cudaMalloc(&world,    (size_t)BS*hw*3*4));
        CUDA_CHECK(cudaMalloc(&gate_table, (size_t)c.loop_steps*6*D*4));
        gate_host.resize((size_t)c.loop_steps*6*D);

        // ransac pose solver: own buffers; sample draws precomputed once with a host rng
        // (ransac consensus is draw-robust, so production needn't match torch's seed bit-exactly).
        ransac.init(BS, c.Himg, c.Wimg, c.patch_size, 100);
        CUDA_CHECK(cudaMalloc(&pose_conf, (size_t)BS*hw*4));
        {
            std::mt19937 g(42);
            ransac_rand_idx.resize((size_t)ransac.n_iter * 8);
            std::vector<int> perm(ransac.n_sample);
            for (int it = 0; it < ransac.n_iter; it++) {
                std::iota(perm.begin(), perm.end(), 0);
                for (int k = 0; k < 8; k++) {
                    std::uniform_int_distribution<int> d(k, ransac.n_sample - 1);
                    std::swap(perm[k], perm[d(g)]);
                    ransac_rand_idx[it * 8 + k] = perm[k];
                }
            }
        }

        // rope tables
        RopeTable rt; rt.build(100.f, DH_()/4, max_len);
        CUDA_CHECK(cudaMalloc(&cos_d, rt.cos_tab.size()*4)); cudaMemcpy(cos_d, rt.cos_tab.data(), rt.cos_tab.size()*4, cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaMalloc(&sin_d, rt.sin_tab.size()*4)); cudaMemcpy(sin_d, rt.sin_tab.data(), rt.sin_tab.size()*4, cudaMemcpyHostToDevice);

        // rope positions [BS, P, 2]
        std::vector<int> hp((size_t)BS*P*2, 0);
        for (int b = 0; b < BS; b++)
            for (int sp = 0; sp < Sp; sp++) {
                int t = c.patch_start + sp;
                hp[((size_t)b*P + t)*2 + 0] = sp / c.pw + 1;
                hp[((size_t)b*P + t)*2 + 1] = sp % c.pw + 1;
            }
        CUDA_CHECK(cudaMalloc(&pos_d, (size_t)BS*P*2*4));
        cudaMemcpy(pos_d, hp.data(), (size_t)BS*P*2*4, cudaMemcpyHostToDevice);

        // arena last (arena_bytes==0 -> size it to the estimated peak need, capped at free vram).
        // the conv depth head dominates; mirror its cap/pad_cap so the estimate is exact.
        if (arena_bytes == 0) {
            int ph = c.ph, pw = c.pw, H = c.Himg, W = c.Wimg;
            size_t cap_f = (size_t)BS * std::max({(size_t)512*(2*ph)*(2*pw), (size_t)256*(4*ph)*(4*pw),
                                                  (size_t)128*(8*ph)*(8*pw), (size_t)68*H*W});
            size_t pad_f = (size_t)BS * std::max({(size_t)(2*ph+2)*(2*pw+2)*512,
                                                  (size_t)(4*ph+2)*(4*pw+2)*256,
                                                  (size_t)(8*ph+2)*(8*pw+2)*128, (size_t)(H+2)*(W+2)*72});
            size_t convhead = 3*cap_f*2 + pad_f*2 + (size_t)512*512*9*2 + (16ull<<20);  // a_/b_/c_ are bf16
            arena.init_capped(convhead + convhead/5);     // +20% slack (encoder/loop, alignment)
        } else {
            arena.init(arena_bytes);
        }
    }
    int DH_() const { return c.DH; }

    // forward, split into stages so a caller can time/stream each (see dvlt.cu)

    // encoder (per-frame dinov2) + token assembly -> x.
    void run_encoder(const DvltWeights& w, const float* img) {
        const int BS = c.B*c.S, P = c.P, D = c.D, H = c.H, DH = c.DH, threads = 256;
        arena.reset();
        DinoV2EncoderPlan enc; enc.init(*cublas, arena, BS, c.Himg, c.Wimg, 14, D, H, DH, c.R);
        enc.fwd(*cublas, w.enc, img, c.Himg, c.Wimg, enc_out);
        build_tokens_kernel<<<((size_t)BS*P*D+threads-1)/threads, threads>>>(
            enc_out, w.camera_token, w.register_token, x, c.B, c.S, c.Sp, c.R, D, c.patch_start);
    }

    // 12-step looping core (shared frame+global block, per-step interval gate). updates x in place.
    void run_loop(const DvltWeights& w) {
        const int BS = c.B*c.S, P = c.P, D = c.D, H = c.H, DH = c.DH;
        arena.reset();
        DepthScaledBlockPlan frame, global;
        frame.init(*cublas, arena, BS, P, H, D, DH, cos_d, sin_d, max_len_, 1e-5f);
        global.init(*cublas, arena, c.B, c.S*P, H, D, DH, cos_d, sin_d, max_len_, 1e-5f);
        // gates are input-independent: build all 12 steps on host, upload once, then index.
        for (int i = 0; i < c.loop_steps; i++) {
            float tn = (float)i/(float)(c.loop_steps-1);
            float tx = (i+1 < c.loop_steps) ? (float)(i+1)/(float)(c.loop_steps-1) : 1.0f;
            float* dst = gate_host.data() + (size_t)i*6*D;
            interval_gate(w.gframe,  tn, tx, dst);        // writes [3*D]
            interval_gate(w.gglobal, tn, tx, dst + 3*D);  // writes [3*D]
        }
        cudaMemcpy(gate_table, gate_host.data(), (size_t)c.loop_steps*6*D*4, cudaMemcpyHostToDevice);
        for (int i = 0; i < c.loop_steps; i++) {
            float* g = gate_table + (size_t)i*6*D;
            frame.fwd(*cublas, w.frame, x, pos_d, g, g+D, g+2*D, true);
            global.fwd(*cublas, w.global, x, nullptr, g+3*D, g+4*D, g+5*D, false);
        }
    }

    // ray head (pixel-shuffle) + camera head (from ray features[:,0]) -> ray_out, pose.
    void run_ray_cam(const DvltWeights& w) {
        const int BS = c.B*c.S, P = c.P, D = c.D, DH = c.DH, DD = c.dec_dim;
        arena.reset();
        DecoderHeadPlan ray_plan;
        ray_plan.init(*cublas, arena, BS, P, D, DD, c.dec_heads, DH, c.ph, c.pw, 14, 6, cos_d, sin_d, max_len_);
        ray_plan.fwd(*cublas, w.ray, x, pos_d, ray_out, ray_feat, c.patch_start);
        cudaMemcpy2D(cls, DD*2, ray_feat, (size_t)P*DD*2, DD*2, BS, cudaMemcpyDeviceToDevice);
        SimpleCameraHeadPlan cam; cam.init(*cublas, arena, BS, DD, DD, 9);
        cam.fwd(*cublas, w.cam, cls, pose);
    }

    // depth head: decoder front-end (all frames) -> conv stage (chunked over frames) -> depth_out.
    void run_depth(const DvltWeights& w) {
        const int BS = c.B*c.S, P = c.P, D = c.D, DH = c.DH, DD = c.dec_dim;
        arena.reset();
        DecoderHeadPlan depth_plan;
        depth_plan.init(*cublas, arena, BS, P, D, DD, c.dec_heads, DH, c.ph, c.pw, 14, 2, cos_d, sin_d, max_len_);
        depth_plan.run_frontend(*cublas, w.depth, x, pos_d, nullptr);
        depth_plan.patch_map_f32(depth_map, c.patch_start);     // depth_map [BS, DD, Sp] persistent
        // the conv head materialises full-res maps (~140 MB/frame), so chunk it over frames:
        // depth_map/depth_out are persistent, so reset + re-init the head per chunk (per-frame independent).
        int ph = c.ph, pw = c.pw, Hc = c.Himg, Wc = c.Wimg;
        size_t mx = std::max({(size_t)512*(2*ph)*(2*pw), (size_t)256*(4*ph)*(4*pw),
                              (size_t)128*(8*ph)*(8*pw), (size_t)68*Hc*Wc});
        size_t pf = std::max({(size_t)(2*ph+2)*(2*pw+2)*512, (size_t)(4*ph+2)*(4*pw+2)*256,
                              (size_t)(8*ph+2)*(8*pw+2)*128, (size_t)(Hc+2)*(Wc+2)*72});
        size_t per_frame = (3*mx + pf) * 2, fixed = (size_t)512*512*9*2 + (16ull<<20);
        int chunk = (int)((arena.cap > fixed ? arena.cap - fixed : 0) / per_frame);
        chunk = std::max(1, std::min(chunk, BS));
        if (const char* e = getenv("DVLT_CONV_CHUNK")) chunk = std::max(1, std::min(atoi(e), BS));  // test override
        size_t dm_pf = (size_t)DD*c.Sp, do_pf = (size_t)2*c.Himg*c.Wimg;
        for (int f0 = 0; f0 < BS; f0 += chunk) {
            int nf = std::min(chunk, BS - f0);
            arena.reset();
            ConvDepthHeadNHWC conv; conv.init(*cublas, arena, nf, ph, pw, Hc, Wc, 2);
            conv.fwd(w.depth_conv, depth_map + (size_t)f0*dm_pf, DD, depth_out + (size_t)f0*do_pf);
        }
    }

    // head activations + world points (ray_origin + ray_dir*depth) -> world/depth/conf/rays_cl.
    void run_finalize() {
        const int BS = c.B*c.S, threads = 256;
        finalize_kernel<<<((size_t)BS*c.Himg*c.Wimg+threads-1)/threads, threads>>>(
            ray_out, depth_out, rays_cl, depth, conf, world, BS, c.Himg, c.Wimg);
    }

    // img [BS, 3, Himg, Wimg] f32 (already normalized). fills world/depth/conf/rays_cl/pose.
    void forward(const DvltWeights& w, const float* img) {
        static const bool prof = getenv("DVLT_PROF") != nullptr;
        cudaEvent_t ev[6]; if (prof) { for (auto& e : ev) cudaEventCreate(&e); }
        auto stamp = [&](int k){ if (prof) cudaEventRecord(ev[k]); };
        stamp(0); run_encoder(w, img);
        stamp(1); run_loop(w);
        stamp(2); run_ray_cam(w);
        stamp(3); run_depth(w);
        stamp(4); run_finalize();
        stamp(5);
        if (prof) {
            cudaEventSynchronize(ev[5]);
            const char* nm[5] = {"encoder", "loop(12x)", "ray+cam", "depth", "finalize"};
            float tot = 0;
            for (int k = 0; k < 5; k++) { float ms; cudaEventElapsedTime(&ms, ev[k], ev[k+1]); printf("  %-10s %7.1f ms\n", nm[k], ms); tot += ms; }
            printf("  %-10s %7.1f ms\n", "TOTAL", tot);
            printf("  arena hwm %5zu MB / cap %zu MB\n", arena.hwm >> 20, arena.cap >> 20);
            for (auto& e : ev) cudaEventDestroy(e);
        }
    }

    // ransac rays_to_pose on the predicted rays -> extrinsics (ransac.extr) + intrinsics (ransac.intr).
    // call after forward(). pose weights default to ones (dvlt use_depth_conf_for_pose=False).
    void solve_pose() {
        const int BS = c.B*c.S; const size_t hw = (size_t)c.Himg*c.Wimg;
        if (c.use_depth_conf_for_pose) cudaMemcpy(pose_conf, conf, BS*hw*4, cudaMemcpyDeviceToDevice);
        else fill_f32_kernel<<<(BS*hw+255)/256, 256>>>(pose_conf, 1.f, BS*hw);
        ransac.run(rays_cl, pose_conf, ransac_rand_idx.data());
    }
    float* extrinsics() { return ransac.extr; }   // [BS, 4, 4]
    float* intrinsics() { return ransac.intr; }   // [BS, 3, 3]
};

} // namespace dvlt
