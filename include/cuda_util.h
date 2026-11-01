#pragma once
// minimal cuda helpers used by the cli: error check, gpu guard, host<->device float transfers.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

namespace dvt {

// refuse to start on a GPU the kernels cannot run on: they are bf16 tensor-core code built for
// sm_80+, and on an older part every launch fails without a word, leaving garbage outputs.
inline void require_sm80() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        fprintf(stderr, "error: no usable CUDA GPU (%s) - needs an NVIDIA GPU and driver\n",
                e != cudaSuccess ? cudaGetErrorString(e) : "none found");
        exit(1);
    }
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    if (p.major < 8) {
        fprintf(stderr, "error: %s is sm_%d%d; dvlt needs compute capability 8.0+ "
                "(Ampere or newer, e.g. RTX 30xx) for its bf16 tensor-core kernels\n",
                p.name, p.major, p.minor);
        exit(1);
    }
}

inline float* to_dev_f32(const float* h, size_t n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n*4));
    CUDA_CHECK(cudaMemcpy(d, h, n*4, cudaMemcpyHostToDevice));
    return d;
}

inline std::vector<float> from_dev_f32(const float* d, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n*4, cudaMemcpyDeviceToHost));
    return v;
}

} // namespace dvt
