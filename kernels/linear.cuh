#pragma once
// linear layer: y = x @ W^T + b
// cublas backend - cuBLASLt for bf16 gemms with auto-tuned algorithms.
// W stored [N, K] (row-major), x [M, K], y [M, N], all row-major bf16.
// fused variant: y = gelu(x @ W^T + bias) via cuBLASLt epilogue.

#include "../include/gemm.h"
#include <cstdio>

namespace dvlt {

// bias-add after gemm: out[m,n] += bias[n]
// out [M,N] bf16, bias [N] fp32.
__global__ void linear_bias_add(
        __nv_bfloat16* __restrict__ out,
        const float*   __restrict__ bias,
        int total, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int n = idx % N;
        out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + bias[n]);
    }
}

// cached linear plan: y = x @ w^T + b, w [out,in]. cuBLASLt, init once per (M,K,N).
struct LinearPlan {
    int M_ = 0, K_ = 0, N_ = 0;

    // cached cuBLASLt descriptors for plain gemm
    cublasLtMatmulDesc_t    desc_     = nullptr;
    cublasLtMatrixLayout_t  lW_       = nullptr;
    cublasLtMatrixLayout_t  lA_       = nullptr;
    cublasLtMatrixLayout_t  lC_       = nullptr;
    cublasLtMatmulAlgo_t    algo_;

    // cached cuBLASLt descriptors for gelu+bias gemm
    cublasLtMatmulDesc_t    gelu_desc_ = nullptr;
    cublasLtMatmulAlgo_t    gelu_algo_;
    cublasStatus_t          gelu_st_   = CUBLAS_STATUS_NOT_INITIALIZED;

    // best heuristic algo for desc over the cached layouts. NOT_SUPPORTED when the heuristic
    // succeeds but offers none; any other failure keeps its own status for the message.
    cublasStatus_t pick_algo(CublasCtx& ctx, cublasLtMatmulDesc_t desc, cublasLtMatmulPreference_t pref,
                             cublasLtMatmulAlgo_t& algo) {
        cublasLtMatmulHeuristicResult_t result;
        int nr = 0;
        cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(ctx.lt, desc, lW_, lA_, lC_, lC_, pref, 1, &result, &nr);
        if (s == CUBLAS_STATUS_SUCCESS && nr == 0) s = CUBLAS_STATUS_NOT_SUPPORTED;
        if (s == CUBLAS_STATUS_SUCCESS) algo = result.algo;
        return s;
    }

    // no algo: matmul on an unset algo would only write garbage. Only "none offered" points
    // at the dll; other statuses are reported as they are.
    void no_kernel(const char* what, cublasStatus_t s) const {
        fprintf(stderr, "cublasLt: no bf16 %s kernel (M=%d K=%d N=%d): %s - %s%s\n", what, M_, K_, N_,
                cublasLtGetStatusName(s), cublasLtGetStatusString(s),
                s == CUBLAS_STATUS_NOT_SUPPORTED ? " (the bundled cublasLt64_12.dll may predate this gpu)" : "");
        exit(1);
    }

    void init(CublasCtx& ctx, int M, int K, int N) {
        M_ = M; K_ = K; N_ = N;

        // row-major C[M,N] = A[M,K] @ W[N,K]^T, expressed col-major for cuBLASLt (transa=T, transb=N).
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;

        // plain gemm
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lW_, CUDA_R_16BF, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA_, CUDA_R_16BF, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lC_, CUDA_R_16BF, N, M, N));

        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws = CublasCtx::WS_BYTES;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &ws, sizeof(ws)));
        if (cublasStatus_t s = pick_algo(ctx, desc_, pref, algo_)) no_kernel("gemm", s);

        // gelu+bias gemm
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&gelu_desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        cudaDataType_t biasType = CUDA_R_16BF;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &biasType, sizeof(biasType)));
        gelu_st_ = pick_algo(ctx, gelu_desc_, pref, gelu_algo_);   // checked at use: few plans run it

        cublasLtMatmulPreferenceDestroy(pref);
    }

    // input [M,K], weight [N,K], output [M,N], bias [N] fp32 or nullptr.
    void exec(CublasCtx& ctx,
              const __nv_bfloat16* input,
              const __nv_bfloat16* weight,
              __nv_bfloat16*       output,
              const float*         bias   = nullptr,
              float                alpha  = 1.f,
              float                beta   = 0.f,
              cudaStream_t         stream = nullptr) {
        CUBLAS_CHECK(cublasLtMatmul(ctx.lt, desc_, &alpha, weight, lW_, input, lA_,
                                    &beta, output, lC_, output, lC_,
                                    &algo_, ctx.workspace, CublasCtx::WS_BYTES, stream));
        if (bias) {
            int total = M_ * N_;
            linear_bias_add<<<(total + 255) / 256, 256, 0, stream>>>(output, bias, total, N_);
        }
    }

    // fused: output[M,N] = gelu(input[M,K] @ weight[N,K]^T + bias[N])
    // bias is bf16 (same as fc1_b in weight structs).
    void exec_gelu_bias(CublasCtx& ctx,
                        const __nv_bfloat16* input,
                        const __nv_bfloat16* weight,
                        __nv_bfloat16*       output,
                        const __nv_bfloat16* bias,
                        cudaStream_t         stream = nullptr) {
        if (gelu_st_ != CUBLAS_STATUS_SUCCESS) no_kernel("gelu+bias gemm", gelu_st_);
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(gelu_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias, sizeof(bias)));
        float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasLtMatmul(ctx.lt, gelu_desc_, &alpha, weight, lW_, input, lA_,
                                    &beta, output, lC_, output, lC_,
                                    &gelu_algo_, ctx.workspace, CublasCtx::WS_BYTES, stream));
    }

    void destroy() {
        if (lW_)        { cublasLtMatrixLayoutDestroy(lW_);      lW_ = nullptr; }
        if (lA_)        { cublasLtMatrixLayoutDestroy(lA_);      lA_ = nullptr; }
        if (lC_)        { cublasLtMatrixLayoutDestroy(lC_);      lC_ = nullptr; }
        if (desc_)      { cublasLtMatmulDescDestroy(desc_);      desc_ = nullptr; }
        if (gelu_desc_) { cublasLtMatmulDescDestroy(gelu_desc_); gelu_desc_ = nullptr; }
    }
};

} // namespace dvlt
