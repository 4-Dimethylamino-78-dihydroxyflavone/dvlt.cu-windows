@echo off
setlocal
cd /d "%~dp0"

REM ======================================================================
REM dvlt.cu-windows - one-time setup, the Windows twin of upstream's setup.sh
REM   1. CUTLASS headers       -> ext\cutlass             sparse git clone, ~30 MB
REM   2. cuBLASLt runtime DLL  -> cublasLt64_12.dll       copied from CUDA_PATH\bin
REM   3. build                 -> dvlt.exe, build\convert.exe        (build.bat)
REM   4. python runtime        -> python\                 embeddable CPython (setup_python.ps1)
REM   5. NVIDIA checkpoint     -> model\model.safetensors  468 MB from Hugging Face
REM   6. bf16 weight blob      -> model\weights.dvlt       convert.exe
REM Each phase is skipped when its output already exists, so re-running is
REM cheap.  "setup.bat nomodel" stops after 4 (what CI does on plain pushes).
REM Needs: git, curl (ships with Windows 10+), CUDA 12.x, VS 2022 C++ tools.
REM ======================================================================

set "CUTLASS_TAG=v4.6.0"
set "HF=https://huggingface.co/nvidia/dvlt/resolve/main"

REM --- 1. CUTLASS: include\ plus the one fused-attention example, nothing else
if exist "ext\cutlass\include\cutlass\cutlass.h" (echo == ext\cutlass present, skipping & goto :cutlass_done)
where git >nul 2>nul || (echo ERROR: git not found & exit /b 1)
echo == fetching CUTLASS %CUTLASS_TAG% headers (sparse clone)
if exist "ext\cutlass.tmp" rmdir /s /q "ext\cutlass.tmp"
git clone --quiet --depth 1 --branch %CUTLASS_TAG% --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git "ext\cutlass.tmp" || exit /b 1
git -C "ext\cutlass.tmp" sparse-checkout set include examples/41_fused_multi_head_attention || exit /b 1
git -C "ext\cutlass.tmp" checkout --quiet || exit /b 1
if not exist "ext\cutlass\examples" mkdir "ext\cutlass\examples"
move "ext\cutlass.tmp\include" "ext\cutlass\include" >nul || exit /b 1
move "ext\cutlass.tmp\examples\41_fused_multi_head_attention" "ext\cutlass\examples\41_fused_multi_head_attention" >nul || exit /b 1
rmdir /s /q "ext\cutlass.tmp"
:cutlass_done

REM --- 2. cuBLASLt: the single CUDA DLL dvlt.exe imports (cudart is linked
REM static). Windows loads it from the exe's own folder, so it lives next to
REM dvlt.exe; the CUDA EULA lists it as redistributable, so the release zip
REM carries it too.
if exist "cublasLt64_12.dll" (echo == cublasLt64_12.dll present, skipping & goto :dll_done)
if not defined CUDA_PATH (echo ERROR: CUDA_PATH is not set - install the CUDA 12.x toolkit & exit /b 1)
if not exist "%CUDA_PATH%\bin\cublasLt64_12.dll" (echo ERROR: cublasLt64_12.dll not found under CUDA_PATH\bin & exit /b 1)
echo == copying cublasLt64_12.dll from "%CUDA_PATH%\bin"
copy /Y "%CUDA_PATH%\bin\cublasLt64_12.dll" . >nul || exit /b 1
:dll_done

REM --- 3. build dvlt.exe + convert.exe
call "%~dp0build.bat" || exit /b 1

REM --- 4. python\: the program folder's own interpreter, so reconstruct.bat
REM never falls back to whatever python.exe is first on PATH. Skipped only when
REM it imports its packages: a half-built python\ (failed download or pip run)
REM is rebuilt, not trusted.
if exist "python\python.exe" "python\python.exe" -c "import numpy, scipy, PIL, trimesh" 2>nul && (echo == python\ present, skipping & goto :python_done)
echo == building python\ (embeddable CPython + requirements.txt)
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_python.ps1" || exit /b 1
:python_done

REM --- 5. checkpoint. NVIDIA License (non-commercial: research and evaluation
REM only), redistributable with its license text - see model\LICENSE.txt. The
REM license is checked on its own, not behind the weights: package.ps1 refuses
REM to ship weights without it.
if /i "%~1"=="nomodel" goto :done
if not exist model mkdir model
if not exist "model\LICENSE.txt" (curl -sSL --fail -o "model\LICENSE.txt" "%HF%/LICENSE.txt" || exit /b 1)
if exist "model\weights.dvlt" (echo == model\weights.dvlt present, skipping & goto :done)
if exist "model\model.safetensors" goto :convert
echo == downloading model.safetensors (468 MB) from %HF%
curl -L --fail --retry 3 -C - -o "model\model.safetensors.part" "%HF%/model.safetensors"
if errorlevel 1 (
    echo ERROR: download failed. Fetch model.safetensors by hand from
    echo        https://huggingface.co/nvidia/dvlt  into  model\  and re-run setup.bat
    exit /b 1
)
move /Y "model\model.safetensors.part" "model\model.safetensors" >nul

REM --- 6. safetensors (f32) -> DVL1 blob: bf16 transformer path, f32 heads
:convert
echo == converting to model\weights.dvlt
"%~dp0build\convert.exe" "model\model.safetensors" "model\weights.dvlt" || exit /b 1

:done
echo.
echo == setup done.  Try:  reconstruct.bat inputs\shoe output\shoe
