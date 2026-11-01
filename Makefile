## dvlt
##   make dvlt              build -> build/dvlt
##   make convert           build the weight converter -> build/convert
##   make dvlt ARCH=sm_86   build for a specific gpu arch

NVCC := nvcc
CXX  := g++

# cutlass headers
CUTLASS_ROOT := $(abspath ext/cutlass)

# gpu arch: auto-detect, or pass ARCH=sm_XX
ifndef ARCH
  _CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  ifneq ($(_CC),)
    ARCH := sm_$(_CC)
  endif
endif
ifndef ARCH
  $(error could not auto-detect gpu arch; pass ARCH=sm_XX, e.g. make dvlt ARCH=sm_86)
endif
ARCH_FLAGS := -arch=$(ARCH) --generate-code arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH)

NVCCFLAGS := \
    -std=c++20 -O3 --use_fast_math \
    $(ARCH_FLAGS) \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -ftemplate-backtrace-limit=0 \
    -Xcompiler=-fno-strict-aliasing,-mavx2,-mfma \
    -lineinfo \
    -DNDEBUG \
    --diag-suppress=550,177 \
    -I. \
    -I$(CUTLASS_ROOT)/include \
    -I$(CUTLASS_ROOT)/examples/41_fused_multi_head_attention

LIBS := -lpthread -lcublasLt

DVLT_DEPS := $(wildcard kernels/*.cuh) $(wildcard include/*.h)

dvlt: dvlt.cu $(DVLT_DEPS)
	@mkdir -p build
	$(NVCC) dvlt.cu $(NVCCFLAGS) $(LIBS) -o build/$@

# weight converter: safetensors -> DVL1 blob
convert: tools/convert.cpp platform_compat.cpp include/platform_compat.h
	@mkdir -p build
	$(CXX) tools/convert.cpp platform_compat.cpp -std=c++20 -O2 -o build/$@

clean:
	rm -f build/dvlt build/convert

.PHONY: dvlt convert clean
