// dvlt.cu
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS   // localtime/snprintf on MSVC
#endif
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <system_error>

#define STB_IMAGE_IMPLEMENTATION
#include "include/dvlt_preprocess.h"
#include "include/arena.h"
#include "include/pos_interp.h"
#include "include/ply_writer.h"
#include "kernels/dvlt_model.cuh"
#include "include/cuda_util.h"

static bool is_directory(const char* p) { std::error_code ec; return std::filesystem::is_directory(p, ec); }

// print the hand-made logo (assets/logo.txt) in cuda-green; terminal only, silently
// skipped if the file isn't found. the saved report.txt uses a plain "dvlt.cu" title.
static void print_banner() {
    FILE* f = fopen("assets/logo.txt", "r");
    if (!f) return;
    fputs("\033[32m", stdout);
    char buf[256];
    while (fgets(buf, sizeof buf, f)) fputs(buf, stdout);
    fclose(f);
    fputs("github.com/yassa9\033[0m\n", stdout);
}

// expand a .txt LIST FILE: one image path per line, '#' starts a comment, blank lines
// skipped. Relative paths resolve against the list file's own directory (lists stay
// portable); ORDER IS PRESERVED AS WRITTEN — a list is an explicit ordering (frame 0 is
// DVLT's reference camera), unlike a directory, which is sorted. Missing files fail fast
// with the offending line number. Also sidesteps the Windows command-line length cap for
// 1000+ inputs across any number of directories.
static bool is_txt(const std::string& p) {
    if (p.size() < 4) return false;
    std::string e = p.substr(p.size() - 4);
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".txt";
}
static std::vector<std::string> expand_list(const char* path, bool* ok) {
    std::vector<std::string> out;
    *ok = false;
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open list file %s\n", path); return out; }
    std::string dir = path;
    size_t cut = dir.find_last_of("/\\");
    dir = (cut == std::string::npos) ? std::string() : dir.substr(0, cut + 1);
    char buf[4096];
    int lineno = 0;
    while (fgets(buf, sizeof buf, f)) {
        lineno++;
        std::string s = buf;
        if (lineno == 1 && s.size() >= 3 && (unsigned char)s[0] == 0xEF
            && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
            s = s.substr(3);                                  // Notepad UTF-8 BOM
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r'
                              || s.back() == ' ' || s.back() == '\t')) s.pop_back();
        size_t b = s.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        s = s.substr(b);
        if (s[0] == '#') continue;
        bool abs = s[0] == '/' || s[0] == '\\' || (s.size() > 1 && s[1] == ':');
        std::string full = abs ? s : dir + s;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) {
            fprintf(stderr, "%s line %d: file not found: %s\n", path, lineno, full.c_str());
            fclose(f);
            return {};
        }
        out.push_back(full);
    }
    fclose(f);
    if (out.empty()) fprintf(stderr, "%s: no image paths listed\n", path);
    else *ok = true;
    return out;
}

// collect jpg/jpeg/png from a directory, sorted (so frame order is deterministic).
static std::vector<std::string> expand_dir(const char* dir) {
    std::vector<std::string> out;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    if (ec) { fprintf(stderr, "cannot open directory %s\n", dir); return out; }
    for (; it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        std::string l = it->path().extension().string();   // includes the dot, e.g. ".JPG"
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l == ".jpg" || l == ".jpeg" || l == ".png")
            out.push_back(it->path().generic_string());     // forward slashes on every platform
    }
    std::sort(out.begin(), out.end());
    return out;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [options] <directory | img1 img2 [img3 ...]>\n\n"
        "options:\n"
        "  -o, --output <dir>     output directory (default: output/<timestamp>)\n"
        "  -w, --weights <path>   weights file     (default: model/weights.dvlt)\n"
        "  -c, --conf <0-1>       drop the lowest-confidence fraction of points (default: 0)\n"
        "  -s, --img-size <N>     working resolution (default: 504)\n"
        "  -k, --loop-steps <N>   looping transformer iterations (default: 12)\n"
        "                         higher K -> better pose accuracy, linear compute cost\n"
        "                         e.g. -k 24 doubles loop time; -k 4 for fast preview\n"
        "  -i, --input <img>      add an input image (repeatable; input is also positional)\n"
        "  -b, --no-banner        hide the ascii logo banner\n"
        "  -x, --export-aux       also write conf.bin (f32 [S,H,W]) and desc.bin\n"
        "                         (f32 [S,768] mean DINOv2 patch tokens per frame)\n"
        "  -e, --encode-only      DINOv2 encoder only: write desc.bin for ALL inputs, then exit\n"
        "                         (streamed sub-batches, near-constant VRAM; no scene.ply)\n"
        "  -h, --help             show this help\n\n"
        "outputs:\n"
        "  <output>/scene.ply     merged point cloud (all views, world space)\n"
        "  <output>/poses.json    camera poses (extrinsics c2w + intrinsics per view)\n\n"
        "examples:\n"
        "  %s photos/                      all jpg/png in directory\n"
        "  %s img1.jpg img2.jpg            explicit image list\n"
        "  %s -o results/ -c 0.5 photos/   custom output + confidence drop\n"
        "  %s views.txt                    paths listed one per line ('#' comments;\n"
        "                                  relative lines resolve against the list's\n"
        "                                  folder; order kept as written - line 1 is\n"
        "                                  the reference camera; any directories)\n\n"
        "video: extract frames first, then run on the directory:\n"
        "  ./tools/v2f.sh -i clip.mp4 -s   sample sharp frames -> inputs/clip/\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    // weights default to the converter's output path; override with -w.
    std::string weights_path = "model/weights.dvlt";
    const char* output_dir = nullptr;   // default below: output/<timestamp>
    int img_size = 504;                 // released working resolution (config.json img_size)
    float conf_frac = 0.f;              // drop this bottom fraction of points by confidence
    int loop_steps = 12;                // inference-time compute knob K (DéjàView §3.4)
    bool show_banner = true;            // -b/--no-banner suppresses the terminal logo
    bool export_aux = false;            // -x: dump conf.bin + desc.bin next to scene.ply
    bool encode_only = false;           // -e: corpus descriptors only, no reconstruction
    std::vector<std::string> positional;
    auto opt = [&](int i, const char* s, const char* l){ return !strcmp(argv[i], s) || !strcmp(argv[i], l); };
    for (int i = 1; i < argc; i++) {
        if (opt(i, "-h", "--help")) { usage(argv[0]); return 0; }
        else if (opt(i, "-b", "--no-banner")) show_banner = false;
        else if (opt(i, "-x", "--export-aux")) export_aux = true;
        else if (opt(i, "-e", "--encode-only")) encode_only = true;
        else if (opt(i, "-o", "--output")   && i + 1 < argc) output_dir   = argv[++i];
        else if (opt(i, "-w", "--weights")  && i + 1 < argc) weights_path = argv[++i];
        else if (opt(i, "-c", "--conf")     && i + 1 < argc) conf_frac    = atof(argv[++i]);
        else if (opt(i, "-s", "--img-size") && i + 1 < argc) img_size     = atoi(argv[++i]);
        else if (opt(i, "-k", "--loop-steps") && i + 1 < argc) {
            loop_steps = atoi(argv[++i]);
            if (loop_steps < 1) { fprintf(stderr, "--loop-steps must be >= 1\n"); return 1; }
        }
        else if (opt(i, "-i", "--input")    && i + 1 < argc) positional.push_back(argv[++i]);
        else if (argv[i][0] == '-') { fprintf(stderr, "unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
        else positional.push_back(argv[i]);
    }
    // default output dir: output/YYYY_MM_DD_HH_MM_SS.
    char ts_buf[64];
    if (!output_dir) {
        time_t now = time(nullptr); struct tm* lt = localtime(&now);
        snprintf(ts_buf, sizeof ts_buf, "output/%04d_%02d_%02d_%02d_%02d_%02d",
                 lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
        output_dir = ts_buf;
    }
    // a single directory arg expands to its images; a .txt positional expands to the
    // paths it lists (order preserved, mixable with direct image args); otherwise the
    // positionals are the image list.
    std::vector<std::string> imgs;
    if (positional.size() == 1 && is_directory(positional[0].c_str())) imgs = expand_dir(positional[0].c_str());
    else {
        for (auto& p : positional) {
            if (is_txt(p)) {
                bool ok = false;
                auto lst = expand_list(p.c_str(), &ok);
                if (!ok) return 1;
                imgs.insert(imgs.end(), lst.begin(), lst.end());
            } else imgs.push_back(p);
        }
    }
    if (imgs.empty()) { usage(argv[0]); return 1; }
    dvt::require_sm80();

    // -e/--encode-only (v1.5 addendum §4.0): descriptors for the WHOLE corpus must exist
    // before any chunking decision does, so stream frames through the per-frame DINOv2
    // encoder in fixed sub-batches and write desc.bin (DVLD, f32 [N,768] raw patch-token
    // means) — no looped block, no decode, no scene.ply. Cheap because frame attention is
    // per-view: only the looped global attention couples views, and that is skipped here,
    // so there is no O(N^2) term and VRAM is near-constant in N.
    //
    // NOTE ON PADDING PARITY: a full run pads to the max crop over ITS OWN frames, while
    // -e pads to the max over the whole corpus. For a mixed-aspect corpus the two grids
    // differ, so -e descriptors are NOT interchangeable with a chunk's -x descriptors.
    // That is fine as used — every corpus-level decision (dedup, FPS, assignment, bridge
    // ranking, hub-mode retrieval) reads -e output only, and chain-mode loop closure reads
    // -x output only — but do not start mixing the two sources without re-checking this.
    //
    // Host-side orchestration only; no device kernel changes.
    if (encode_only) {
        auto t0 = std::chrono::high_resolution_clock::now();
        constexpr int De = 768, PATCH = 14, Mn = 37, SUB = 16;
        const int N = (int)imgs.size();
        // header-only pre-scan (no pixel decode): per-image post-crop dims -> corpus max grid.
        int Hmax = 0, Wmax = 0;
        for (auto& p : imgs) {
            int sw, sh, nc;
            if (!stbi_info(p.c_str(), &sw, &sh, &nc)) {
                fprintf(stderr, "encode-only: cannot read %s\n", p.c_str()); return 1;
            }
            // header dims ignore EXIF; load_frame rotates 5-8, so the grid must too
            // (else a portrait frame centre-pads negatively and writes out of bounds)
            if (dvltpp::exif_orientation(p.c_str()) >= 5) std::swap(sw, sh);
            float sc = (float)img_size / std::max(sh, sw);
            Hmax = std::max(Hmax, dvltpp::make_divisible((int)lroundf(sh * sc), PATCH));
            Wmax = std::max(Wmax, dvltpp::make_divisible((int)lroundf(sw * sc), PATCH));
        }
        const int ph = Hmax / PATCH, pw = Wmax / PATCH, Sp = ph * pw;

        dvlt::DvltConfig ecfg{};
        ecfg.B = 1; ecfg.D = De; ecfg.H = 12; ecfg.DH = 64; ecfg.R = 4; ecfg.patch_start = 5;
        ecfg.enc_blocks = 12; ecfg.dec_dim = 384; ecfg.dec_heads = 6; ecfg.dec_depth = 2;
        ecfg.loop_steps = 1; ecfg.patch_size = PATCH;
        dvlt::DvltConfig ecfg_native = ecfg;
        ecfg_native.Himg = Mn * PATCH; ecfg_native.Wimg = Mn * PATCH; ecfg_native.ph = Mn;
        ecfg_native.pw = Mn; ecfg_native.Sp = Mn * Mn; ecfg_native.P = 5 + Mn * Mn;

        CublasCtx cublas; cublas.init();
        dvlt::WeightFile wf; wf.open(weights_path.c_str());
        dvlt::DvltWeights w{};
        dvlt::load_dvlt_weights(w, wf, ecfg_native);
        if (ph != Mn || pw != Mn) w.enc.pos = dvlt::interpolate_pos_embed(w.pos_native, Mn, De, ph, pw);

        ecfg.S = SUB; ecfg.Himg = Hmax; ecfg.Wimg = Wmax; ecfg.ph = ph; ecfg.pw = pw;
        ecfg.Sp = Sp; ecfg.P = ecfg.patch_start + Sp;
        dvlt::DvltPipeline pipe;
        pipe.init(cublas, ecfg, /*arena_bytes=*/0, /*max_len=*/std::max(ph, pw) + 4);

        const size_t img_elems = (size_t)SUB * 3 * Hmax * Wmax;
        float* img_dev = nullptr;
        CUDA_CHECK(cudaMalloc(&img_dev, img_elems * 4));
        std::vector<float> staging(img_elems);
        std::vector<__nv_bfloat16> henc((size_t)SUB * Sp * De);
        std::vector<float> desc((size_t)N * De, 0.f);

        for (int off = 0; off < N; off += SUB) {
            const int n = std::min(SUB, N - off);
            std::fill(staging.begin(), staging.end(), 0.f);   // zero-pad short final batch
            std::vector<dvltpp::Frame> fr(n);
            dvltpp::load_frames(fr, imgs, off, n, img_size, PATCH);
            for (int i = 0; i < n; i++) {                     // center-pad + resnet-normalize
                const dvltpp::Frame& f = fr[i];
                int pt = (Hmax - f.H) / 2, pl = (Wmax - f.W) / 2;
                for (int y = 0; y < f.H; y++)
                    for (int x = 0; x < f.W; x++) {
                        const float* p = &f.rgb[((size_t)y * f.W + x) * 3];
                        for (int c = 0; c < 3; c++)
                            staging[((size_t)i * 3 + c) * Hmax * Wmax + (size_t)(y + pt) * Wmax + (x + pl)]
                                = (p[c] - dvltpp::RESNET_MEAN[c]) / dvltpp::RESNET_STD[c];
                    }
            }
            CUDA_CHECK(cudaMemcpy(img_dev, staging.data(), img_elems * 4, cudaMemcpyHostToDevice));
            pipe.run_encoder(w, img_dev);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(henc.data(), pipe.enc_out, henc.size() * 2, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) {
                float* dp = desc.data() + (size_t)(off + i) * De;
                for (int ti = 0; ti < Sp; ti++) {
                    const __nv_bfloat16* row = henc.data() + ((size_t)i * Sp + ti) * De;
                    for (int k = 0; k < De; k++) dp[k] += __bfloat162float(row[k]);
                }
                for (int k = 0; k < De; k++) dp[k] /= (float)Sp;
            }
            fprintf(stderr, "\r  encode %d/%d", std::min(off + SUB, N), N); fflush(stderr);
        }
        fprintf(stderr, "\n");

        { std::error_code ec; std::filesystem::create_directories(output_dir, ec); }
        std::string dpath = std::string(output_dir) + "/desc.bin";
        FILE* df = fopen(dpath.c_str(), "wb");
        if (!df) { fprintf(stderr, "encode-only: cannot write %s\n", dpath.c_str()); return 1; }
        const char dmagic[4] = {'D','V','L','D'};
        int32_t ddims[2] = {N, De};
        fwrite(dmagic, 1, 4, df); fwrite(ddims, 4, 2, df);
        fwrite(desc.data(), 4, desc.size(), df);
        fclose(df);
        double ems = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        printf("encode-only: %d frames @ %dx%d -> %s (%.0f ms, %.1f ms/frame)\n",
               N, Wmax, Hmax, dpath.c_str(), ems, ems / std::max(1, N));
        return 0;
    }

    auto t_wall = std::chrono::high_resolution_clock::now();
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    // print to stdout AND tee into `report` (ansi escapes stripped) -> written to report.txt later.
    // the report drops the leading 2-space shift on each line (kept on the terminal) so the file
    // is flush-left; deeper sub-item indent is preserved (only the base shift is removed).
    std::string report;
    auto emit = [&](const char* s){
        fputs(s, stdout); fflush(stdout);
        int indent = (report.empty() || report.back() == '\n') ? 2 : 0;
        for (const char* p = s; *p; ) {
            if (*p == '\033' && p[1] == '[') {                       // skip a CSI escape (e.g. color)
                p += 2; while (*p && !((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z'))) p++; if (*p) p++;
                continue;
            }
            char c = *p++;
            if (c == '\n') { report += c; indent = 2; continue; }
            if (indent > 0 && c == ' ') { indent--; continue; }      // strip the base shift
            indent = 0;
            report += c;
        }
    };
    auto line = [&](const char* fmt, auto... a){ char b[128]; snprintf(b, sizeof b, fmt, a...); emit(b); };
    auto row  = [&](const char* k, double ms){ line("  %-22s %10.1f\n", k, ms); };
    auto rows = [&](const char* k, const char* v){ line("  %-22s %10s\n", k, v); };
    const char* rule = "──────────────────────";

    constexpr int B = 1, D = 768, H = 12, DH = 64, R = 4, PATCH_START = 5;
    constexpr int ENC_BLOCKS = 12, DEC_DIM = 384, DEC_HEADS = 6, DEC_DEPTH = 2;
    constexpr int PATCH = 14, M = 37;   // M = native dinov2 grid the blob was dumped at
    dvlt::DvltConfig cfg{};
    cfg.B = B; cfg.D = D; cfg.H = H; cfg.DH = DH; cfg.R = R; cfg.patch_start = PATCH_START;
    cfg.enc_blocks = ENC_BLOCKS; cfg.dec_dim = DEC_DIM; cfg.dec_heads = DEC_HEADS; cfg.dec_depth = DEC_DEPTH;
    cfg.loop_steps = loop_steps; cfg.patch_size = PATCH;
    dvlt::DvltConfig cfg_native = cfg;
    cfg_native.Himg = M * PATCH; cfg_native.Wimg = M * PATCH; cfg_native.ph = M; cfg_native.pw = M;
    cfg_native.Sp = M * M; cfg_native.P = PATCH_START + M * M;

    // overlap the cpu preprocess (worker thread, no cuda) with the weight load (main thread):
    // independent since the weights parse at the native 37x37 grid. each records its own duration.
    double prep_ms = 0, weight_ms = 0;
    dvltpp::Batch batch;
    auto t_pp = std::chrono::high_resolution_clock::now();
    std::thread pp([&]{ batch = dvltpp::preprocess(imgs, img_size, 14); prep_ms = ms_since(t_pp); });

    auto t_wt = std::chrono::high_resolution_clock::now();
    CublasCtx cublas; cublas.init();
    dvlt::WeightFile wf; wf.open(weights_path.c_str());
    dvlt::DvltWeights w{};
    dvlt::load_dvlt_weights(w, wf, cfg_native);   // correct cursor walk at native Sp
    weight_ms = ms_since(t_wt);
    pp.join();

    const int S = batch.S, Himg = batch.H, Wimg = batch.W;
    const int ph = Himg / 14, pw = Wimg / 14, Sp = ph * pw;
    // interpolate the encoder pos embed from the native MxM grid to the cropped ph x pw grid.
    if (ph != M || pw != M) w.enc.pos = dvlt::interpolate_pos_embed(w.pos_native, M, D, ph, pw);

    // streamed report: title, an info block (config/value), then a timing block (stage/time).
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); char v[64];
    if (show_banner) print_banner();                    // green logo, terminal only
    report += "\ndvlt.cu\ngithub.com/yassa9\n";          // plain title in the saved report
    line("\n  %-22s %10s\n", "info", "value");
    rows(rule, "──────────");
    snprintf(v, sizeof v, "sm_%d", prop.major*10 + prop.minor);     rows("arch", v);
    snprintf(v, sizeof v, "%zu MB", prop.totalGlobalMem >> 20);     rows("vram total", v);
    line("  %-22s %10d\n", "frames", S);
    snprintf(v, sizeof v, "%dx%d", Wimg, Himg);                     rows("resolution", v);
    snprintf(v, sizeof v, "%dx%d", pw, ph);                         rows("patch grid", v);
    snprintf(v, sizeof v, "%.2f", conf_frac);                       rows("conf drop", v);
    line("  %-22s %10d\n", "loop steps (-k)", loop_steps);
    rows(rule, "──────────");
    line("  %-22s %10s\n", "stage", "time (ms)");
    rows(rule, "──────────");
    row("weights loading", weight_ms);
    row("preprocess", prep_ms);   // runs concurrently with the weight load (overlapped in wall time)

    // pipeline init (sizes the arena to free vram).
    auto t = std::chrono::high_resolution_clock::now();
    cfg.S = S; cfg.Himg = Himg; cfg.Wimg = Wimg; cfg.ph = ph; cfg.pw = pw; cfg.Sp = Sp; cfg.P = cfg.patch_start + Sp;
    dvlt::DvltPipeline pipe;
    pipe.init(cublas, cfg, /*arena_bytes=*/0, /*max_len=*/std::max(ph, pw) + 4);
    double init_ms = ms_since(t); row("pipeline init", init_ms);
    float* img_dev = dvt::to_dev_f32(batch.norm.data(), (size_t)S * 3 * Himg * Wimg);

    // gpu stages: sync after each so the row streams (and the time is real). the sync reports
    // faults during execution; a launch that never ran (bad config, no kernel image for this
    // gpu) only shows in cudaGetLastError - unchecked, the stage "succeeds" on stale buffers.
    double gpu_total = 0;
    auto stage = [&](const char* name, auto&& fn) {
        auto ts = std::chrono::high_resolution_clock::now();
        fn(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        double dt = ms_since(ts); gpu_total += dt; row(name, dt);
    };
    char loop_label[32]; snprintf(loop_label, sizeof loop_label, "loop (%d steps)", loop_steps);
    stage("encoder (12 blk)",   [&]{ pipe.run_encoder(w, img_dev); });
    stage(loop_label,           [&]{ pipe.run_loop(w); });
    stage("ray + camera head",  [&]{ pipe.run_ray_cam(w); });
    stage("depth head",         [&]{ pipe.run_depth(w); });
    stage("world points",       [&]{ pipe.run_finalize(); });
    stage("ransac pose",        [&]{ pipe.solve_pose(); });

    // vram now at its peak (arena + weights + buffers all resident, nothing freed yet).
    size_t fb, tb; cudaMemGetInfo(&fb, &tb);
    size_t vram_peak = (tb - fb) >> 20, vram_total = tb >> 20;

    // read back, filter, write outputs (one timed "output" row).
    auto t_out = std::chrono::high_resolution_clock::now();
    const size_t hw = (size_t)Himg * Wimg;
    auto world = dvt::from_dev_f32(pipe.world, (size_t)S * hw * 3);
    auto depth = dvt::from_dev_f32(pipe.depth, (size_t)S * hw);
    auto conf  = dvt::from_dev_f32(pipe.conf,  (size_t)S * hw);
    auto extr  = dvt::from_dev_f32(pipe.extrinsics(), (size_t)S * 16);
    auto intr  = dvt::from_dev_f32(pipe.intrinsics(), (size_t)S * 9);

    // keep finite, positive-depth points; with --conf P>0 also drop the lowest-P fraction by
    // confidence (percentile threshold over the finite/positive points).
    float conf_thresh = -1e30f;
    if (conf_frac > 0.f) {
        std::vector<float> cv;
        for (size_t i = 0; i < (size_t)S * hw; i++)
            if (std::isfinite(conf[i]) && depth[i] > 0.f && depth[i] < 1e4f) cv.push_back(conf[i]);
        if (!cv.empty()) {
            size_t k = (size_t)(conf_frac * cv.size()); if (k >= cv.size()) k = cv.size() - 1;
            std::nth_element(cv.begin(), cv.begin() + k, cv.end());
            conf_thresh = cv[k];
        }
    }
    std::vector<uint8_t> mask((size_t)S * hw, 0);
    size_t kept = 0;
    for (size_t i = 0; i < (size_t)S * hw; i++) {
        float d = depth[i];
        bool ok = std::isfinite(world[i*3]) && std::isfinite(world[i*3+1]) && std::isfinite(world[i*3+2])
                  && d > 0.f && d < 1e4f && conf[i] >= conf_thresh;
        mask[i] = ok ? 1 : 0; kept += ok;
    }
    { std::error_code ec; std::filesystem::create_directories(output_dir, ec); }  // makes parent "output/" too
    std::string ply_stem = std::string(output_dir) + "/scene";
    write_pointcloud_ply_filtered(ply_stem.c_str(), world.data(), mask.data(), batch.norm.data(), S, Himg, Wimg);

    // poses.json: per-frame extrinsics (c2w 4x4) + intrinsics (3x3).
    std::string jpath = std::string(output_dir) + "/poses.json";
    FILE* jf = fopen(jpath.c_str(), "w");
    fprintf(jf, "{\n  \"frames\": [\n");
    for (int s = 0; s < S; s++) {
        const float* E = extr.data() + (size_t)s * 16;
        const float* K = intr.data() + (size_t)s * 9;
        fprintf(jf, "    {\n      \"extrinsics_c2w\": [");
        for (int k = 0; k < 16; k++) fprintf(jf, "%s%.6f", k ? ", " : "", E[k]);
        fprintf(jf, "],\n      \"intrinsics\": [");
        for (int k = 0; k < 9; k++) fprintf(jf, "%s%.6f", k ? ", " : "", K[k]);
        fprintf(jf, "]\n    }%s\n", s + 1 < S ? "," : "");
    }
    fprintf(jf, "  ]\n}\n");
    fclose(jf);

    // -x aux export: per-pixel confidence + per-frame mean DINOv2 patch token.
    // conf is already on the host; enc_out persists untouched after run_encoder.
    if (export_aux) {
        std::string cpath = std::string(output_dir) + "/conf.bin";
        if (FILE* cf = fopen(cpath.c_str(), "wb")) {
            const char cmagic[4] = {'D','V','L','C'};
            int32_t cdims[3] = {S, Himg, Wimg};
            fwrite(cmagic, 1, 4, cf); fwrite(cdims, 4, 3, cf);
            fwrite(conf.data(), 4, (size_t)S * hw, cf);
            fclose(cf);
        }
        std::vector<__nv_bfloat16> henc((size_t)S * Sp * D);
        cudaMemcpy(henc.data(), pipe.enc_out, henc.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> desc((size_t)S * D, 0.f);
        for (int si = 0; si < S; si++) {
            float* dp = desc.data() + (size_t)si * D;
            for (int ti = 0; ti < Sp; ti++) {
                const __nv_bfloat16* row = henc.data() + ((size_t)si * Sp + ti) * D;
                for (int ki = 0; ki < D; ki++) dp[ki] += __bfloat162float(row[ki]);
            }
            for (int ki = 0; ki < D; ki++) dp[ki] /= (float)Sp;
        }
        std::string dpath = std::string(output_dir) + "/desc.bin";
        if (FILE* df = fopen(dpath.c_str(), "wb")) {
            const char dmagic[4] = {'D','V','L','D'};
            int32_t ddims[2] = {S, D};
            fwrite(dmagic, 1, 4, df); fwrite(ddims, 4, 2, df);
            fwrite(desc.data(), 4, desc.size(), df);
            fclose(df);
        }
    }
    double post_ms = ms_since(t_out);

    // footer: aggregate timing + vram + summary
    // compute = sum of the gpu stage rows; wall clock = whole run (disk load -> ply written).
    rows(rule, "──────────");
    row("compute",     gpu_total);
    row("postprocess", post_ms);
    row("wall clock",  ms_since(t_wall));
    rows(rule, "──────────");
    snprintf(v, sizeof v, "%zu MB", vram_peak);  rows("vram peak", v);
    snprintf(v, sizeof v, "%zu MB", vram_total); rows("vram total", v);

    const size_t total_pts = (size_t)S * hw;
    line("\n  %s/\n", output_dir);
    line("    %-16s %zu pts  (%d%% kept)\n", "scene.ply", kept, (int)(100.0 * kept / total_pts));
    line("    %-16s %d poses\n", "poses.json", S);
    if (export_aux) {
        line("    %-16s f32 [S,H,W] confidence\n", "conf.bin");
        line("    %-16s f32 [S,768] frame descriptors\n", "desc.bin");
    }
    line("    %-16s the report above\n\n", "report.txt");

    // dump the full report (as printed, minus colors) next to the ply/json.
    std::string rpath = std::string(output_dir) + "/report.txt";
    if (FILE* rf = fopen(rpath.c_str(), "w")) { fputs(report.c_str(), rf); fclose(rf); }
    return 0;
}
