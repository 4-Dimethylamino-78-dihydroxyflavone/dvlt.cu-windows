#pragma once
// gemm interface. bf16 -> cuBLASLt, fp32 -> cutlass (output heads only).
// convention: y[M,N] = x[M,K] @ W^T, W stored [N,K].

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdlib>
#include <cstdio>
#include "cuda_util.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// a failed cuBLASLt call leaves its output untouched and the run carries on with garbage
// (e.g. a gpu newer than the bundled cublasLt64_12.dll), so every status is fatal.
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cublasLt error %s:%d: %s - %s\n", __FILE__, __LINE__, \
                cublasLtGetStatusName(_s), cublasLtGetStatusString(_s)); \
        exit(1); \
    } \
} while(0)

struct CublasCtx {
    cublasLtHandle_t lt = nullptr;
    void* workspace = nullptr;
    static constexpr size_t WS_BYTES = 4 * 1024 * 1024;
    void init() {
        CUBLAS_CHECK(cublasLtCreate(&lt));
        CUDA_CHECK(cudaMalloc(&workspace, WS_BYTES));
    }
    void destroy() {
        if (workspace) { cudaFree(workspace); workspace = nullptr; }
        if (lt) { cublasLtDestroy(lt); lt = nullptr; }
    }
};

// fp32 gemm (cutlass, used only by output heads)
namespace dvlt_detail {

using F32GemmTF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4, 4, 4
>;

using F32GemmSimt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 1, 1
>;

template<typename Gemm>
inline void launch_f32_gemm(const float* A, const float* W, float* C,
                             int M, int K, int N, float alpha, float beta,
                             cudaStream_t stream) {
    Gemm op;
    typename Gemm::Arguments args(
        {M, N, K},
        {A, K}, {W, K}, {C, N}, {C, N},
        {alpha, beta}
    );
    auto s = op.can_implement(args);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass f32 gemm: can_implement failed (%d) M=%d K=%d N=%d\n",
                (int)s, M, K, N);
        exit(1);
    }
    s = op(args, nullptr, stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "cutlass f32 gemm: launch failed (%d)\n", (int)s);
        exit(1);
    }
}

} // namespace dvlt_detail

inline void gemm_f32(
    CublasCtx&,
    const float* A, const float* W, float* C,
    int M, int K, int N,
    float alpha = 1.f, float beta = 0.f,
    cudaStream_t stream = nullptr
) {
    if (N % 4 == 0 && K % 4 == 0 && M % 4 == 0)
        dvlt_detail::launch_f32_gemm<dvlt_detail::F32GemmTF32>(A, W, C, M, K, N, alpha, beta, stream);
    else
        dvlt_detail::launch_f32_gemm<dvlt_detail::F32GemmSimt>(A, W, C, M, K, N, alpha, beta, stream);
}
