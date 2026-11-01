// safetensors -> DVL1 weight blob. bf16 for the transformer path, f32 for the
// precision-sensitive bits (pos_embed, gates, output heads), in load_dvlt_weights order.
//   usage: ./build/convert [model/model.safetensors] [model/weights.dvlt]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include "../include/platform_compat.h"   // read-only file mapping: mmap (posix) / MapViewOfFile (win32)

static uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    b += 0x7FFF + ((b >> 16) & 1);                   // round to nearest even
    return (uint16_t)(b >> 16);
}

// minimal json parser for the safetensors header.
struct TensorInfo { std::string dtype; size_t begin = 0, end = 0; };

struct JsonParser {
    const char* p; const char* end;
    void ws() { while (p < end && (*p==' '||*p=='\n'||*p=='\r'||*p=='\t')) p++; }
    bool eat(char c) { ws(); if (p < end && *p == c) { p++; return true; } return false; }
    std::string str() {
        ws(); if (p >= end || *p != '"') { fprintf(stderr, "json: expected '\"'\n"); exit(1); }
        p++; std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\' && p+1 < end) { p++; s += (*p=='n'?'\n':*p=='t'?'\t':*p); }
            else s += *p;
            p++;
        }
        if (p < end) p++; return s;
    }
    int64_t num() {
        ws(); bool neg = false; if (p < end && *p=='-') { neg = true; p++; }
        int64_t v = 0; while (p < end && *p>='0' && *p<='9') v = v*10 + (*p++ - '0');
        return neg ? -v : v;
    }
    std::vector<int64_t> arr() {
        std::vector<int64_t> a; eat('['); ws();
        if (p < end && *p==']') { p++; return a; }
        while (true) { a.push_back(num()); ws();
            if (p < end && *p==',') { p++; continue; }
            if (p < end && *p==']') { p++; break; } exit(1); }
        return a;
    }
    void skip() {
        ws(); if (p >= end) return;
        if (*p=='"') str();
        else if (*p=='{') { p++; ws(); if (p<end && *p=='}') { p++; return; }
            while (true) { str(); eat(':'); skip(); ws();
                if (p<end && *p==',') { p++; continue; } if (p<end && *p=='}') { p++; break; } } }
        else if (*p=='[') { p++; ws(); if (p<end && *p==']') { p++; return; }
            while (true) { skip(); ws();
                if (p<end && *p==',') { p++; continue; } if (p<end && *p==']') { p++; break; } } }
        else while (p < end && *p!=',' && *p!='}' && *p!=']') p++;
    }
    TensorInfo descriptor() {
        TensorInfo ti; eat('{'); ws(); if (p<end && *p=='}') { p++; return ti; }
        while (true) {
            std::string k = str(); eat(':');
            if (k=="dtype") ti.dtype = str();
            else if (k=="data_offsets") { auto o = arr(); if (o.size()>=2) { ti.begin=o[0]; ti.end=o[1]; } }
            else skip();
            ws(); if (p<end && *p==',') { p++; continue; } if (p<end && *p=='}') { p++; break; }
        }
        return ti;
    }
    std::unordered_map<std::string, TensorInfo> header() {
        std::unordered_map<std::string, TensorInfo> m; eat('{'); ws();
        if (p<end && *p=='}') return m;
        while (true) {
            std::string k = str(); eat(':');
            if (k=="__metadata__") skip(); else m[k] = descriptor();
            ws(); if (p<end && *p==',') { p++; continue; } if (p<end && *p=='}') { p++; break; }
        }
        return m;
    }
};

static const int NB_ENC = 12;

// precision-sensitive tensors stored as f32; everything else bf16.
static bool is_f32(const std::string& n) {
    auto ew = [&](const char* s){ size_t k = strlen(s); return n.size()>=k && n.compare(n.size()-k,k,s)==0; };
    auto sw = [&](const char* s){ return n.rfind(s, 0) == 0; };
    return ew("pos_embed")
        || n.find("depth_scale.") != std::string::npos
        || sw("ray_decoder.head.")
        || sw("camera_head.fc_pose.")
        || sw("depth_decoder.upsample_blocks.")
        || sw("depth_decoder.output_block.");
}

int main(int argc, char** argv) {
    const char* src = argc > 1 ? argv[1] : "model/model.safetensors";
    const char* dst = argc > 2 ? argv[2] : "model/weights.dvlt";
    { std::error_code ec; std::filesystem::create_directories(std::filesystem::path(dst).parent_path(), ec); }

    platcompat::FileMap fm = platcompat::map_readonly(src);   // fail-fast with a message, like the loader
    const uint8_t* base = fm.data;

    uint64_t hlen; memcpy(&hlen, base, 8);
    JsonParser jp{ (const char*)base + 8, (const char*)base + 8 + hlen };
    auto hdr = jp.header();
    const uint8_t* data = base + 8 + hlen;   // tensor data section follows the json header

    // canonical tensor order (must match load_dvlt_weights' cursor walk).
    std::vector<std::string> order;
    auto add = [&](std::initializer_list<std::string> ns){ for (auto& n : ns) order.push_back(n); };

    const std::string e = "patch_embed_encoder.";
    add({e+"patch_embed.proj.weight", e+"patch_embed.proj.bias", e+"cls_token", e+"pos_embed", e+"register_tokens"});
    for (int i = 0; i < NB_ENC; i++) {
        std::string p = e + "blocks." + std::to_string(i) + ".";
        add({p+"norm1.weight", p+"norm1.bias", p+"attn.qkv.weight", p+"attn.qkv.bias",
             p+"attn.proj.weight", p+"attn.proj.bias", p+"ls1.gamma",
             p+"norm2.weight", p+"norm2.bias", p+"mlp.fc1.weight", p+"mlp.fc1.bias",
             p+"mlp.fc2.weight", p+"mlp.fc2.bias", p+"ls2.gamma"});
    }
    add({"camera_token", "register_token"});

    auto ds = [&](const std::string& p){
        add({p+"norm1.weight", p+"norm1.bias", p+"attn.qkv.weight", p+"attn.qkv.bias",
             p+"attn.q_norm.weight", p+"attn.q_norm.bias", p+"attn.k_norm.weight", p+"attn.k_norm.bias",
             p+"attn.proj.weight", p+"attn.proj.bias", p+"ls1.gamma", p+"norm2.weight", p+"norm2.bias",
             p+"mlp.fc1.weight", p+"mlp.fc1.bias", p+"mlp.fc2.weight", p+"mlp.fc2.bias", p+"ls2.gamma",
             p+"depth_scale.proj.0.weight", p+"depth_scale.proj.0.bias",
             p+"depth_scale.proj.2.weight", p+"depth_scale.proj.2.bias"});
    };
    ds("recurrent_blocks.0.frame_attn.");
    ds("recurrent_blocks.0.global_attn.");

    auto dec_block = [&](const std::string& p){
        add({p+"norm1.weight", p+"norm1.bias", p+"attn.qkv.weight", p+"attn.qkv.bias",
             p+"attn.q_norm.weight", p+"attn.q_norm.bias", p+"attn.k_norm.weight", p+"attn.k_norm.bias",
             p+"attn.proj.weight", p+"attn.proj.bias", p+"norm2.weight", p+"norm2.bias",
             p+"mlp.fc1.weight", p+"mlp.fc1.bias", p+"mlp.fc2.weight", p+"mlp.fc2.bias"});
    };

    add({"ray_decoder.proj_in.weight", "ray_decoder.proj_in.bias"});
    for (int i = 0; i < 2; i++) dec_block("ray_decoder.blocks." + std::to_string(i) + ".");
    add({"ray_decoder.norm.weight", "ray_decoder.norm.bias", "ray_decoder.head.weight", "ray_decoder.head.bias"});

    add({"depth_decoder.proj_in.weight", "depth_decoder.proj_in.bias"});
    for (int i = 0; i < 2; i++) dec_block("depth_decoder.blocks." + std::to_string(i) + ".");
    add({"depth_decoder.norm.weight", "depth_decoder.norm.bias"});
    for (int s = 0; s < 3; s++) {
        std::string up = "depth_decoder.upsample_blocks." + std::to_string(s) + ".";
        add({up+"0.0.weight", up+"0.0.bias", up+"0.1.weight", up+"0.1.bias"});
        for (int r : {1, 2}) {
            std::string rp = up + std::to_string(r) + ".layers.";
            add({rp+"0.weight", rp+"0.bias", rp+"2.weight", rp+"2.bias",
                 rp+"3.weight", rp+"3.bias", rp+"5.weight", rp+"5.bias"});
        }
    }
    add({"depth_decoder.output_block.0.weight", "depth_decoder.output_block.0.bias",
         "depth_decoder.output_block.2.weight", "depth_decoder.output_block.2.bias"});

    add({"camera_head.mlp.0.weight", "camera_head.mlp.0.bias", "camera_head.mlp.1.weight",
         "camera_head.mlp.1.bias", "camera_head.mlp.3.weight", "camera_head.mlp.3.bias",
         "camera_head.fc_pose.weight", "camera_head.fc_pose.bias"});

    int missing = 0;
    for (auto& n : order) if (!hdr.count(n)) { if (missing < 3) fprintf(stderr, "missing: %s\n", n.c_str()); missing++; }
    if (missing) { fprintf(stderr, "missing %d tensors\n", missing); return 1; }

    FILE* f = fopen(dst, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", dst); return 1; }
    fwrite("DVL1", 1, 4, f);
    uint32_t count = (uint32_t)order.size(); fwrite(&count, 4, 1, f);

    int n_bf16 = 0;
    std::vector<uint16_t> bf;
    for (auto& name : order) {
        const TensorInfo& ti = hdr[name];                // checkpoint is all f32
        uint64_t n = (ti.end - ti.begin) / 4;            // element count
        const float* a = (const float*)(data + ti.begin);
        uint8_t dt = is_f32(name) ? 0 : 1;
        fwrite(&dt, 1, 1, f); fwrite(&n, 8, 1, f);
        if (dt == 0) {
            fwrite(a, 4, n, f);
        } else {
            bf.resize(n);
            for (uint64_t i = 0; i < n; i++) bf[i] = f32_to_bf16(a[i]);
            fwrite(bf.data(), 2, n, f);
            n_bf16++;
        }
    }
    long bytes = ftell(f);
    fclose(f);
    platcompat::unmap(fm);
    printf("wrote %s: %zu tensors (%d bf16), %.1f MB\n", dst, order.size(), n_bf16, bytes / 1e6);
    return 0;
}
