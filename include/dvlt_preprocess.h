#pragma once

// dvlt image preprocessing in c++ (mirrors dvlt.util.preprocess.preprocess_images):
// rgb /255 -> antialias bilinear resize (longest side -> img_size) -> center-crop to a
// multiple of patch_size -> resnet normalize.

#include "stb_image.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <atomic>
#include <thread>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace dvltpp {

static const float RESNET_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float RESNET_STD[3]  = {0.229f, 0.224f, 0.225f};

inline int make_divisible(int v, int d) { return (v / d) * d; }

// one separable axis of a pil-style antialias bilinear resample: builds, per output index,
// the source window [start,start+n) and triangle weights (normalized).
struct Resampler {
    std::vector<int> start;       // per out pixel: first src index
    std::vector<int> nweights;    // per out pixel: window length
    std::vector<float> weights;   // flattened windows
    int ksize = 0;

    void build(int in, int out) {
        float scale = (float)in / out;                 // src pixels per dst pixel
        float fscale = scale < 1.f ? 1.f : scale;       // antialias: widen when downsampling
        float support = fscale;                         // triangle radius 1 * fscale
        ksize = (int)ceilf(support) * 2 + 1;
        start.resize(out); nweights.resize(out); weights.assign((size_t)out * ksize, 0.f);
        for (int i = 0; i < out; i++) {
            float center = (i + 0.5f) * scale;
            int s = (int)floorf(center - support + 0.5f); if (s < 0) s = 0;
            int e = (int)floorf(center + support + 0.5f); if (e > in) e = in;
            float wsum = 0.f; float* wp = &weights[(size_t)i * ksize];
            for (int j = s; j < e; j++) {
                float t = (j + 0.5f - center) / fscale;
                float wv = 1.f - fabsf(t); if (wv < 0.f) wv = 0.f;
                wp[j - s] = wv; wsum += wv;
            }
            if (wsum > 0.f) for (int j = 0; j < e - s; j++) wp[j] /= wsum;
            start[i] = s; nweights[i] = e - s;
        }
    }
};

// antialias resize an interleaved-rgb float image [src_h,src_w,3] (0..1) -> [dst_h,dst_w,3].
inline std::vector<float> resize_rgb(const std::vector<float>& src, int src_h, int src_w,
                                     int dst_h, int dst_w) {
    Resampler rx, ry; rx.build(src_w, dst_w); ry.build(src_h, dst_h);
    std::vector<float> tmp((size_t)src_h * dst_w * 3, 0.f);   // horizontal pass
    for (int y = 0; y < src_h; y++)
        for (int x = 0; x < dst_w; x++) {
            const float* wp = &rx.weights[(size_t)x * rx.ksize]; int s = rx.start[x], n = rx.nweights[x];
            float acc[3] = {0, 0, 0};
            for (int k = 0; k < n; k++) {
                const float* p = &src[((size_t)y * src_w + (s + k)) * 3];
                acc[0] += p[0] * wp[k]; acc[1] += p[1] * wp[k]; acc[2] += p[2] * wp[k];
            }
            float* o = &tmp[((size_t)y * dst_w + x) * 3]; o[0] = acc[0]; o[1] = acc[1]; o[2] = acc[2];
        }
    // vertical pass. the weights are uniform across x for a given output row y, so it's a row-wise
    // axpy (dst_row += w[k] * tmp_row[s+k]) - vectorized with avx2/fma (scalar fallback).
    std::vector<float> dst((size_t)dst_h * dst_w * 3, 0.f);
    const int W3 = dst_w * 3;
    for (int y = 0; y < dst_h; y++) {
        const float* wp = &ry.weights[(size_t)y * ry.ksize]; int s = ry.start[y], n = ry.nweights[y];
        float* o = &dst[(size_t)y * W3];
        for (int k = 0; k < n; k++) {
            const float* row = &tmp[(size_t)(s + k) * W3]; float wk = wp[k];
            int i = 0;
#if defined(__AVX2__)
            __m256 vw = _mm256_set1_ps(wk);
            for (; i + 8 <= W3; i += 8)
                _mm256_storeu_ps(o + i, _mm256_fmadd_ps(_mm256_loadu_ps(row + i), vw, _mm256_loadu_ps(o + i)));
#endif
            for (; i < W3; i++) o[i] += row[i] * wk;
        }
    }
    return dst;
}

// preprocessed frame: rgb [0,1] image (for point colors) + its post-crop size.
struct Frame { std::vector<float> rgb; int H, W; };   // rgb is [H,W,3] interleaved, 0..1

// ---- EXIF orientation (ported from vggttt-win, 2026-08-15) --------------------------
// stb_image ignores EXIF, so phone photos arrive as the sensor stored them -- often
// sideways. A batch mixing gravity directions shatters the reconstruction (measured on
// the Sputnik set: 88x rot90CW + 9x rot90CCW + 9x rot180 + 33x upright -> fragment soup).
// Minimal JPEG APP1/TIFF walk for tag 0x0112; returns 1 (upright) on anything unexpected.
inline int exif_orientation(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 1;
    auto rd = [&](uint8_t* b, size_t n) { return fread(b, 1, n, f) == n; };
    uint8_t h[2];
    if (!rd(h, 2) || h[0] != 0xFF || h[1] != 0xD8) { fclose(f); return 1; }
    while (true) {
        uint8_t m[2];
        if (!rd(m, 2) || m[0] != 0xFF) break;
        if (m[1] == 0xDA || m[1] == 0xD9) break;
        uint8_t lb[2];
        if (!rd(lb, 2)) break;
        int len = (lb[0] << 8 | lb[1]) - 2;
        if (len < 0) break;
        if (m[1] != 0xE1) { fseek(f, len, SEEK_CUR); continue; }
        std::vector<uint8_t> seg(len);
        if (!rd(seg.data(), len)) break;
        if (len < 14 || memcmp(seg.data(), "Exif\0\0", 6) != 0) continue;
        const uint8_t* t = seg.data() + 6;
        size_t tn = len - 6;
        bool le;
        if (t[0] == 'I' && t[1] == 'I') le = true;
        else if (t[0] == 'M' && t[1] == 'M') le = false;
        else break;
        auto u16 = [&](size_t o) { return le ? (t[o] | t[o + 1] << 8) : (t[o] << 8 | t[o + 1]); };
        auto u32 = [&](size_t o) {
            return le ? ((uint32_t)t[o] | t[o+1] << 8 | t[o+2] << 16 | (uint32_t)t[o+3] << 24)
                      : ((uint32_t)t[o] << 24 | t[o+1] << 16 | t[o+2] << 8 | (uint32_t)t[o+3]);
        };
        if (tn < 8) break;
        uint32_t ifd = u32(4);
        if (ifd + 2 > tn) break;
        int cnt = u16(ifd);
        for (int i = 0; i < cnt; i++) {
            size_t e = ifd + 2 + (size_t)i * 12;
            if (e + 12 > tn) break;
            if (u16(e) == 0x0112) {
                int v = u16(e + 8);
                fclose(f);
                return (v >= 1 && v <= 8) ? v : 1;
            }
        }
        break;
    }
    fclose(f);
    return 1;
}

// Undo the stored orientation: dst is the upright image. Orientations 5-8 swap dimensions.
inline void exif_reorient(std::vector<uint8_t>& px, int& w, int& h, int o) {
    if (o <= 1) return;
    int W = w, H = h;
    int nw = (o >= 5) ? H : W, nh = (o >= 5) ? W : H;
    std::vector<uint8_t> out((size_t)nw * nh * 3);
    for (int dy = 0; dy < nh; dy++)
        for (int dx = 0; dx < nw; dx++) {
            int sx, sy;
            switch (o) {
                case 2: sx = W - 1 - dx; sy = dy;         break;
                case 3: sx = W - 1 - dx; sy = H - 1 - dy; break;
                case 4: sx = dx;         sy = H - 1 - dy; break;
                case 5: sx = dy;         sy = dx;         break;
                case 6: sx = dy;         sy = H - 1 - dx; break;   // rot 90 CW to display
                case 7: sx = W - 1 - dy; sy = H - 1 - dx; break;
                default:sx = W - 1 - dy; sy = dx;         break;   // 8: rot 90 CCW
            }
            memcpy(&out[((size_t)dy * nw + dx) * 3], &px[((size_t)sy * W + sx) * 3], 3);
        }
    px.swap(out); w = nw; h = nh;
}

// load one image -> EXIF-upright -> resize-longest -> center-crop /patch. rgb [H,W,3] 0..1.
inline Frame load_frame(const char* path, int img_size, int patch) {
    int sw, sh, nc; uint8_t* raw = stbi_load(path, &sw, &sh, &nc, 3);
    if (!raw) { fprintf(stderr, "preprocess: cannot load %s: %s\n", path, stbi_failure_reason()); exit(1); }
    std::vector<uint8_t> px(raw, raw + (size_t)sh * sw * 3);
    stbi_image_free(raw);
    exif_reorient(px, sw, sh, exif_orientation(path));
    std::vector<float> src((size_t)sh * sw * 3);
    for (size_t i = 0; i < src.size(); i++) src[i] = px[i] / 255.f;
    px.clear(); px.shrink_to_fit();

    float scale = (float)img_size / std::max(sh, sw);
    int nh = (int)lroundf(sh * scale), nw = (int)lroundf(sw * scale);
    auto rs = resize_rgb(src, sh, sw, nh, nw);

    int ch = make_divisible(nh, patch), cw = make_divisible(nw, patch);
    int top = (nh - ch) / 2, left = (nw - cw) / 2;
    Frame f; f.H = ch; f.W = cw; f.rgb.resize((size_t)ch * cw * 3);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            const float* p = &rs[((size_t)(y + top) * nw + (x + left)) * 3];
            float* o = &f.rgb[((size_t)y * cw + x) * 3]; o[0] = p[0]; o[1] = p[1]; o[2] = p[2];
        }
    return f;
}

// load paths[off, off+n) into frames[0, n) on min(n, cores) workers pulling indices from a
// counter. a full-res decode is transient but large (~0.3 GB for 24 MP as float), so one
// thread per frame held every frame's decode at once: 10-45 GB of host RAM for 150 frames.
inline void load_frames(std::vector<Frame>& frames, const std::vector<std::string>& paths,
                        size_t off, int n, int img_size, int patch) {
    int nt = std::min(n, (int)std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++)
        th.emplace_back([&]{
            for (int i = next++; i < n; i = next++) frames[i] = load_frame(paths[off + i].c_str(), img_size, patch);
        });
    for (auto& t : th) t.join();
}

// batch of frames -> normalized [S,3,H,W] (center-padded to common H,W, pad 0) + size.
// point colors come from `norm` (the ply writer denormalizes), so no separate rgb buffer is kept.
struct Batch { std::vector<float> norm; int S, H, W; };

inline Batch preprocess(const std::vector<std::string>& paths, int img_size, int patch) {
    int S = (int)paths.size();
    std::vector<Frame> frames(S);
    load_frames(frames, paths, 0, S, img_size, patch);

    int H = 0, W = 0;
    for (auto& f : frames) { H = std::max(H, f.H); W = std::max(W, f.W); }
    Batch b; b.S = S; b.H = H; b.W = W;
    b.norm.assign((size_t)S * 3 * H * W, 0.f);
    for (int i = 0; i < S; i++) {
        Frame& f = frames[i];
        int pt = (H - f.H) / 2, pl = (W - f.W) / 2;
        for (int y = 0; y < f.H; y++)
            for (int x = 0; x < f.W; x++) {
                const float* p = &f.rgb[((size_t)y * f.W + x) * 3];
                for (int c = 0; c < 3; c++) {
                    size_t o = ((size_t)i * 3 + c) * H * W + (size_t)(y + pt) * W + (x + pl);
                    b.norm[o] = (p[c] - RESNET_MEAN[c]) / RESNET_STD[c];
                }
            }
    }
    return b;
}

} // namespace dvltpp
