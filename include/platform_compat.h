#pragma once
// cross-platform read-only whole-file mapping (declarations only).
// the implementation lives in platform_compat.cpp so that <windows.h> never
// enters the cuda/cutlass translation unit (dvlt.cu). POSIX uses mmap+MAP_POPULATE,
// Windows uses CreateFileMapping/MapViewOfFile. handles are kept as opaque void*
// so this header pulls in no OS headers.
#include <cstddef>
#include <cstdint>

namespace platcompat {

struct FileMap {
    const uint8_t* data = nullptr;  // mapped, read-only, whole file
    size_t         size = 0;
    void*          h0   = nullptr;  // win: file handle      / posix: unused
    void*          h1   = nullptr;  // win: mapping handle   / posix: unused
    long long      fd   = -1;       // posix: file descriptor/ win: unused
};

// map a file read-only; prints to stderr and exit(1) on any failure (matches the
// original loader's fail-fast behavior).
FileMap map_readonly(const char* path);

// release the mapping (the process normally just exits, but this keeps it clean).
void unmap(FileMap& m);

} // namespace platcompat
