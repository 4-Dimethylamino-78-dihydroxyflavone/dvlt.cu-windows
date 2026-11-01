#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// gpu vram arena: one cudaMalloc, bump-allocated, 256-byte aligned. no free.
struct GpuArena {
    uint8_t* base   = nullptr;
    size_t   cap    = 0;
    size_t   offset = 0;
    size_t   hwm    = 0; // high-water mark: max bytes ever live (across reset cycles)

    void init(size_t bytes) {
        assert(base == nullptr && "arena already initialized");
        cudaError_t e = cudaMalloc(&base, bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "arena: cudaMalloc(%zu) failed: %s\n",
                    bytes, cudaGetErrorString(e));
            exit(1);
        }
        cap    = bytes;
        offset = 0;
    }

    // size the arena to `want` bytes (the pipeline's estimated peak need) but never more than the
    // vram free right now minus a headroom reserve (cuda context, cublasLt/cutlass internal
    // workspaces, fragmentation). call AFTER the pipeline's persistent buffers are allocated so
    // cudaMemGetInfo reflects the true remainder. shrinks and retries if the malloc fails (free mem
    // is never fully allocatable in one block). if `want` exceeds the cap, we still allocate the cap
    // and let alloc()'s overflow check fire loudly during forward.
    void init_capped(size_t want, size_t headroom = (size_t)768 * 1024 * 1024) {
        assert(base == nullptr && "arena already initialized");
        size_t freeb = 0, totalb = 0;
        cudaMemGetInfo(&freeb, &totalb);
        size_t lim = (freeb > headroom) ? freeb - headroom : freeb / 2;
        size_t bytes = want < lim ? want : lim;
        // always attempt the wanted size first; the 128 MB floor only bounds the backoff, so a
        // genuinely small need (low --img-size) still allocates instead of being skipped.
        constexpr size_t FLOOR = (size_t)128 * 1024 * 1024;
        cudaError_t e = cudaErrorMemoryAllocation;
        for (;;) {
            e = cudaMalloc(&base, bytes);
            if (e == cudaSuccess) break;
            cudaGetLastError();              // clear only this attempt's oom, or the next stage's
                                             // launch check reports it as its own
            if (bytes <= FLOOR) break;       // genuine oom: give up once down to the floor
            bytes = bytes * 9 / 10;          // back off 10% and retry
        }
        if (e != cudaSuccess) {
            fprintf(stderr, "arena: could not allocate %zu MB (%zu MB free) - lower --img-size "
                    "or use a gpu with more vram\n", bytes >> 20, freeb >> 20);
            exit(1);
        }
        cap = bytes; offset = 0;

        if (bytes < want)
            fprintf(stderr, "  note: arena capped to %zu MB (ideal %zu, %zu free) -> conv head will chunk\n",
                    bytes >> 20, want >> 20, freeb >> 20);
    }

    // alloc n elements of type T, aligned to 256 bytes. the overflow check runs even under NDEBUG
    // (a silent over-alloc returns an out-of-bounds pointer -> "illegal memory access" later).
    template<typename T>
    T* alloc(size_t n) {
        constexpr size_t ALIGN = 256;
        size_t bytes  = n * sizeof(T);
        size_t padded = (bytes + ALIGN - 1) & ~(ALIGN - 1);
        if (offset + padded > cap) {
            fprintf(stderr,
                    "arena overflow: this alloc needs %zu MB but only %zu MB left (cap %zu MB). "
                    "fewer frames / lower --img-size, or more vram.\n",
                    padded >> 20, (cap - offset) >> 20, cap >> 20);
            std::abort();
        }
        T* ptr  = reinterpret_cast<T*>(base + offset);
        offset += padded;
        if (offset > hwm) hwm = offset;
        return ptr;
    }

    // reset to zero (does not free, just rewinds cursor)
    void reset() { offset = 0; }

    size_t used()  const { return offset; }
    size_t avail() const { return cap - offset; }

    void destroy() {
        if (base) { cudaFree(base); base = nullptr; }
    }
};
