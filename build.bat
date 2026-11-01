@echo off
setlocal
cd /d "%~dp0"

REM ======================================================================
REM dvlt.cu-windows - build dvlt.exe and convert.exe from this tree.
REM   needs   CUDA 12.x (nvcc + cuBLASLt), Visual Studio 2022 C++ build tools,
REM           and the CUTLASS headers in ext\cutlass (setup.bat fetches them)
REM   writes  build\dvlt.exe (copied to .\dvlt.exe, where the wrappers and the
REM           release layout expect it) and build\convert.exe
REM CUDA_PATH is set by the CUDA installer (and by the CI action); point it at
REM a different 12.x install to build against that one.
REM ======================================================================

if not defined CUDA_PATH (echo ERROR: CUDA_PATH is not set - install the CUDA 12.x toolkit & exit /b 1)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (echo ERROR: nvcc.exe not found under CUDA_PATH\bin & exit /b 1)
if not exist "ext\cutlass\include\cutlass\cutlass.h" (echo ERROR: CUTLASS headers missing - run setup.bat first, it fetches ext\cutlass & exit /b 1)

REM --- MSVC: ask vswhere for a VS 2022 (17.x) C++ toolset and load its x64
REM environment - always, even in a developer prompt: a cl.exe that is merely
REM on PATH without INCLUDE/LIB (a bare PATH entry, seen in the wild) compiles
REM nothing. Not newer than 17.x: MSVC 14.4x is what this port is built and
REM tested with, and it is already past CUDA 12.1's supported range (hence
REM -allow-unsupported-compiler).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo ERROR: vswhere.exe not found - install Visual Studio 2022 Build Tools with the C++ workload & exit /b 1)
"%VSWHERE%" -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath > "%TEMP%\dvlt_vsroot.txt"
set "VSROOT="
set /p VSROOT=<"%TEMP%\dvlt_vsroot.txt"
del "%TEMP%\dvlt_vsroot.txt"
if not defined VSROOT (echo ERROR: no Visual Studio 2022 with the C++ x64 toolset found & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo ERROR: vcvars64.bat failed & exit /b 1)

if not exist build mkdir build

echo === building build\dvlt.exe  (sm_80/86/89 SASS + compute_80 PTX) ===
"%CUDA_PATH%\bin\nvcc.exe" dvlt.cu platform_compat.cpp ^
  -std=c++20 -O3 --use_fast_math ^
  -gencode arch=compute_80,code=sm_80 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_80,code=compute_80 ^
  -allow-unsupported-compiler ^
  --expt-extended-lambda --expt-relaxed-constexpr ^
  -cudart static ^
  -DNDEBUG ^
  --diag-suppress=550,177 ^
  -I. -Iext\cutlass\include -Iext\cutlass\examples\41_fused_multi_head_attention ^
  -Xcompiler "/O2 /EHsc /std:c++20 /arch:AVX2 /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH /wd4244 /wd4267 /wd4305" ^
  -lcublasLt ^
  -o build\dvlt.exe
if errorlevel 1 (echo === dvlt.exe build FAILED === & exit /b 1)

echo === building build\convert.exe ===
cl /nologo /std:c++20 /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS /Fobuild\ tools\convert.cpp platform_compat.cpp /Febuild\convert.exe
if errorlevel 1 (echo === convert.exe build FAILED === & exit /b 1)

copy /Y build\dvlt.exe dvlt.exe >nul
echo === OK: dvlt.exe, build\convert.exe ===
