// platform_compat.cpp — OS-specific read-only file mapping, isolated from the
// cuda/cutlass translation unit. compiled as a plain host C++ object and linked in.
#include "include/platform_compat.h"
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace platcompat {

FileMap map_readonly(const char* path) {
    FileMap m;
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "weights: cannot open %s\n", path); exit(1);
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) {
        fprintf(stderr, "weights: size query failed on %s\n", path); exit(1);
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        fprintf(stderr, "weights: CreateFileMapping failed on %s\n", path); exit(1);
    }
    void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
        fprintf(stderr, "weights: MapViewOfFile failed on %s\n", path); exit(1);
    }
    m.data = (const uint8_t*)p;
    m.size = (size_t)sz.QuadPart;
    m.h0   = (void*)fh;
    m.h1   = (void*)mh;
#else
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "weights: cannot open %s\n", path); exit(1); }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "weights: fstat failed on %s\n", path); exit(1); }
    int flags = MAP_PRIVATE;
  #ifdef MAP_POPULATE
    flags |= MAP_POPULATE;  // prefault the whole file (faster than scattered faults)
  #endif
    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, flags, fd, 0);
    if (p == MAP_FAILED) { fprintf(stderr, "weights: mmap failed on %s\n", path); exit(1); }
    m.data = (const uint8_t*)p;
    m.size = (size_t)st.st_size;
    m.fd   = fd;
#endif
    return m;
}

void unmap(FileMap& m) {
#if defined(_WIN32)
    if (m.data) UnmapViewOfFile((LPCVOID)m.data);
    if (m.h1)   CloseHandle((HANDLE)m.h1);
    if (m.h0 && m.h0 != (void*)INVALID_HANDLE_VALUE) CloseHandle((HANDLE)m.h0);
#else
    if (m.data) munmap((void*)m.data, m.size);
    if (m.fd >= 0) ::close((int)m.fd);
#endif
    m.data = nullptr; m.size = 0; m.h0 = m.h1 = nullptr; m.fd = -1;
}

} // namespace platcompat
