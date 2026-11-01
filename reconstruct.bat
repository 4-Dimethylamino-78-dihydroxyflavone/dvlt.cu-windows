@echo off
setlocal EnableDelayedExpansion

REM ======================================================================
REM dvlt.cu Windows - Smart Reconstruction Script
REM Usage: reconstruct.bat <image_dir | list.txt> [output_dir] [conf_drop] [loop_steps] [formats]
REM   list.txt    one image path per line ('#' comments), e.g. a vfold
REM               selection; order is kept as written (line 1 = ref camera)
REM   output_dir  default: a sibling of the input named <input>_out_dvlt
REM   conf_drop   0-1: drop that fraction of lowest-confidence points
REM               (default 0.5 = NVIDIA's demo default; affects density, not
REM                reconstruction quality, and suppresses sky/background
REM                hallucination. Lower toward 0.1 to keep low-texture ground
REM                like water and shadow.)
REM   loop_steps  K, trained range 8-16 (default 16 = paper optimum;
REM               K>16 degrades poses, K<8 is out-of-distribution)
REM   formats     dvlt_postprocess outputs (default ply,glb,las). Comma list from
REM               ply,glb,las,xyz,pcd,npz,colmap,transforms - or 'all'. xyz is slow.
REM Datasets over 150 frames (on 16 GB; every frame limit scales with the
REM GPU's VRAM) are routed to the long-sequence Sim3 pipeline.
REM Optional preprocessing (single-pass runs), intermediates in <output>\prep:
REM   --ppisp     vppisp: a first dvlt pass supplies correspondences, PPISP's
REM               camera model (exposure, colour, vignetting, response curve)
REM               is fitted to them, and every frame is re-rendered at one
REM               exposure. The COLMAP export gets these corrected frames.
REM   --sky       vsky: sky segmented and painted black before dvlt sees it,
REM               then every exactly-black point is stripped from scene.ply.
REM   --keep-prep keep <output>\prep (deleted after a successful run by default;
REM               the COLMAP export already holds the corrected frames).
REM ======================================================================

REM ROOT is captured FIRST. After the cd, a quoted relative invocation
REM ("sub dir\reconstruct.bat") re-resolves %~dp0 against the new cwd and
REM doubles the folder; after a plain `shift`, %~dp0 means "directory of the
REM last argument". The loop below uses `shift /1` to leave %0 alone, and
REM everything script-relative goes through ROOT anyway. The cd stays because
REM dvlt.exe reads assets\logo.txt from the cwd.
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "DVLT=%ROOT%dvlt.exe"
REM Python: the local python\ (setup.bat builds it, release zips ship it), else
REM the first python.exe on PATH - named in the banner, since a PATH python may
REM lack numpy/scipy/trimesh.
set "PY=%ROOT%python\python.exe"
if not exist "%PY%" for %%P in (python.exe) do set "PY=%%~$PATH:P"
REM Optional preprocessing tools: beside this script (vsky.exe, or vsky\vsky.exe;
REM likewise vppisp), else the sibling source checkouts' release builds. The
REM VSKY / VPPISP environment variables override all of these. vsky.exe needs
REM onnxruntime_providers_*.dll beside it for CUDA, or it says so and uses CPU.
for %%P in ("%ROOT%vsky.exe" "%ROOT%vsky\vsky.exe" "%ROOT%..\vsky\target\release\vsky.exe") do if not defined VSKY if exist "%%~fP" set "VSKY=%%~fP"
for %%P in ("%ROOT%vppisp.exe" "%ROOT%vppisp\vppisp.exe" "%ROOT%..\vppisp\target\release\vppisp.exe") do if not defined VPPISP if exist "%%~fP" set "VPPISP=%%~fP"

REM -d / --default: the lazy preset. It fills in every option NOT given on the
REM command line, so anything typed still wins (-d -c 0.3 is conf 0.3), and
REM --no-sky / --no-ppisp switch a preset step off for one run. Edit freely.
set "D_CONF=0.1"
set "D_K=16"
set "D_FORMATS=all"
set "D_SKY=1"
set "D_PPISP=1"

if "%~1"=="" (
    echo Usage: reconstruct.bat ^<image_folder ^| list.txt^> [output_dir] [conf_drop 0-1] [loop_steps 8-16] [formats]
    echo.
    echo Flags accepted anywhere: -f/--formats  -c/--conf  -k/--loop-steps  -o/--out
    echo                          --sky ^(vsky^)  --ppisp ^(vppisp^)   single-pass runs only
    echo                          -d/--default: the preset at the top of this file fills
    echo                          every option not typed ^(now: -c !D_CONF! -k !D_K! -f !D_FORMATS!,
    echo                          sky !D_SKY!, ppisp !D_PPISP!^); --no-sky/--no-ppisp opt out
    echo                          --keep-prep: keep ^<output^>\prep ^(deleted by default^)
    echo output_dir defaults to a sibling of the input: ^<input^>_out_dvlt
    echo A bare number or format name is never the output: 0.3 is conf_drop, 12 is
    echo loop_steps, glb or colmap,glb are formats. Such a folder needs -o.
    echo.
    echo Examples:
    echo   reconstruct.bat C:\Photos\scene                    -^> C:\Photos\scene_out_dvlt
    echo   reconstruct.bat C:\Photos\scene 0.3 12             -^> same folder, conf 0.3, K 12
    echo   reconstruct.bat C:\Photos\scene\vfold-60.txt 0.3   -^> C:\Photos\scene_vfold-60_out_dvlt
    echo   reconstruct.bat inputs\shoe output\shoe_result 0.2
    echo   reconstruct.bat C:\Photos\scene output\scene 0.1 12
    echo   reconstruct.bat C:\Photos\scene\vfold-60.txt output\scene-60
    echo   reconstruct.bat C:\Photos\scene\vfold-60.txt output\scene-60 -f all
    echo   reconstruct.bat --formats colmap,glb C:\Photos\scene output\scene
    echo   reconstruct.bat C:\Photos\scene 0.1 -f all --sky --ppisp
    echo   reconstruct.bat C:\Photos\scene -d                 -^> the preset, same as above
    exit /b 1
)

REM Arguments: positional <input> [output] [conf] [k] [formats], with flags
REM accepted anywhere: -f/--formats, -c/--conf, -k/--loop-steps, -o/--out.
REM Flags exist because nobody remembers positional slot 5; positionals stay
REM because every existing invocation uses them. A flag claims its slot, and
REM remaining positionals fill the still-empty slots left to right - except
REM that a bare NUMBER or FORMAT NAME is never the output folder (see below).
set "INPUT="
set "OUTPUT="
set "CONF="
set "K="
set "FORMATS="
set "SKY="
set "PPISP="
set "PRESET="
set "KEEP_PREP="

:parse
if "%~1"=="" goto parsed
set "ARG=%~1"
if /i "%ARG%"=="-f"           ( set "FORMATS=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="--formats"    ( set "FORMATS=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="-c"           ( set "CONF=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="--conf"       ( set "CONF=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="-k"           ( set "K=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="--loop-steps" ( set "K=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="-o"           ( set "OUTPUT=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="--out"        ( set "OUTPUT=%~2" & shift /1 & shift /1 & goto parse )
if /i "%ARG%"=="--sky"        ( set "SKY=1" & shift /1 & goto parse )
if /i "%ARG%"=="--ppisp"      ( set "PPISP=1" & shift /1 & goto parse )
if /i "%ARG%"=="--no-sky"     ( set "SKY=0" & shift /1 & goto parse )
if /i "%ARG%"=="--no-ppisp"   ( set "PPISP=0" & shift /1 & goto parse )
if /i "%ARG%"=="--keep-prep"  ( set "KEEP_PREP=1" & shift /1 & goto parse )
if /i "%ARG%"=="-d"           ( set "PRESET=1" & shift /1 & goto parse )
if /i "%ARG%"=="--default"    ( set "PRESET=1" & shift /1 & goto parse )
REM !ARG! not %ARG% inside the block: percent expansion happens when the block
REM is parsed, so a ')' in the argument (e.g. "scene (2)") would close the
REM block early and run the exit unconditionally. Delayed expansion is safe.
if "%ARG:~0,1%"=="-" (
    echo Unknown flag: !ARG!
    echo Flags: -f/--formats  -c/--conf  -k/--loop-steps  -o/--out  --sky  --ppisp  --keep-prep  -d/--default
    exit /b 1
)
REM cmd splits unquoted commas and PowerShell strips the quotes that would
REM protect them, so "-f colmap,glb" arrives as -f colmap glb. An argument made
REM only of format names (glb, colmap,glb) therefore joins the format list
REM wherever it stands after the input or -f, and is never the output: a folder
REM named like a format needs -o. Doubled commas give every name its own
REM ",name," to delete; the '.' keeps REST defined once the names are gone.
set "REST=.,!ARG:,=,,!,"
for %%n in (ply glb las xyz pcd npz colmap transforms all) do set "REST=!REST:,%%n,=!"
if "!REST!"=="." if defined INPUT   ( set "FORMATS=!FORMATS!,!ARG!" & shift /1 & goto parse )
if "!REST!"=="." if defined FORMATS ( set "FORMATS=!FORMATS!,!ARG!" & shift /1 & goto parse )
if not defined INPUT   ( set "INPUT=%ARG%" & shift /1 & goto parse )
REM After the input, an argument made only of digits and dots (at least one
REM digit) is a number, never a folder: conf if it has a '.' or is 0/1,
REM loop_steps if it is any other integer. So "X 0.3" is conf 0.3 with the
REM default output, not a folder named 0.3. A numeric folder needs -o.
REM Pure cmd, no `echo | findstr`: a pipe runs in a child cmd where !ARG! does
REM not expand. First loop: any non-[0-9.] character yields a token -> not a
REM number. Second: splitting on '.' yields a token only if a digit exists.
set "NUM="
for /f "delims=0123456789." %%x in ("!ARG!") do set "NUM=no"
if not defined NUM for /f "delims=." %%x in ("!ARG!") do set "NUM=yes"
if "!NUM!"=="yes" (
    set "ISCONF="
    if not "!ARG:.=!"=="!ARG!" set "ISCONF=1"
    if "!ARG!"=="0" set "ISCONF=1"
    if "!ARG!"=="1" set "ISCONF=1"
    if defined ISCONF if not defined CONF ( set "CONF=!ARG!" & shift /1 & goto parse )
    if not defined ISCONF if not defined K ( set "K=!ARG!" & shift /1 & goto parse )
    echo Unexpected number: !ARG! - conf/loop_steps already set. A folder named
    echo like a number must be given with -o.
    exit /b 1
)
if not defined OUTPUT  ( set "OUTPUT=%ARG%" & shift /1 & goto parse )
if not defined CONF    ( set "CONF=%ARG%" & shift /1 & goto parse )
if not defined K       ( set "K=%ARG%" & shift /1 & goto parse )
echo Too many arguments at: !ARG!
exit /b 1
:parsed

if not defined INPUT (
    echo No input given. Run without arguments for usage.
    exit /b 1
)
if defined FORMATS if "!FORMATS:~0,1!"=="," set "FORMATS=!FORMATS:~1!"
REM Paths go full first (relative = from the program folder, the cwd), which
REM collapses doubled separators and turns / into \. Then a trailing \ goes:
REM "scene\" would put the default output INSIDE the input (scene\_out_dvlt),
REM and a \ before a closing quote reaches dvlt.exe/python as \" - an escaped
REM quote that merges the following arguments. A drive root stays a root:
REM "E:\" becomes "E:\.", never the drive-relative "E:".
for %%V in (INPUT OUTPUT) do if defined %%V (
    for %%A in ("!%%V!") do set "%%V=%%~fA"
    if "!%%V:~-1!"=="\" set "%%V=!%%V:~0,-1!"
    if "!%%V:~1!"==":" set "%%V=!%%V!\."
)
if not exist "!INPUT!" (
    echo Input not found: !INPUT!
    exit /b 1
)
REM The preset fills only what the command line left empty.
if defined PRESET (
    if not defined CONF set "CONF=!D_CONF!"
    if not defined K set "K=!D_K!"
    if not defined FORMATS set "FORMATS=!D_FORMATS!"
    if not defined SKY set "SKY=!D_SKY!"
    if not defined PPISP set "PPISP=!D_PPISP!"
)
if "!SKY!"=="0" set "SKY="
if "!PPISP!"=="0" set "PPISP="
if not defined CONF set "CONF=0.5"
if not defined K set "K=16"

REM Validate numerics HERE: a mistyped positional otherwise reaches dvlt.exe
REM as a cryptic "--loop-steps must be >= 1". K is 1-64 with no leading zero:
REM cmd's IF reads 017 as octal 15 while dvlt.exe reads 17, and anything past
REM 64 is wasted compute far outside the trained 8-16.
echo %K%| findstr /r /x "[1-9] [1-5][0-9] 6[0-4]" >nul || (
    echo loop_steps/-k must be a whole number 1-64 without leading zeros, got: "!K!"
    exit /b 1
)
echo %CONF%| findstr /r /x "0 1 0\.[0-9]* \.[0-9]* 1\.0*" >nul || (
    echo conf_drop/-c must be a number in 0-1, got: "!CONF!"
    exit /b 1
)
REM A fresh clone has no dvlt.exe, DLL or weights, so say so before anything is
REM created. cuBLASLt cannot be left for the run to discover: without it dvlt.exe
REM dies in the Windows loader before it can print a word (exit 0xC0000135) and
REM the run just stops after the banner. A CUDA toolkit on PATH also satisfies it,
REM which is how a missing DLL goes unnoticed on a build machine.
set "CUBLAS="
if exist "%ROOT%cublasLt64_12.dll" set "CUBLAS=1"
if not defined CUBLAS for %%P in (cublasLt64_12.dll) do set "CUBLAS=%%~$PATH:P"
set "MISSING="
if not exist "%DVLT%" set "MISSING=!MISSING! dvlt.exe"
if not defined CUBLAS set "MISSING=!MISSING! cublasLt64_12.dll"
if not exist "%ROOT%model\weights.dvlt" set "MISSING=!MISSING! model\weights.dvlt"
if defined MISSING (
    echo Not found in !ROOT!:!MISSING!
    echo Download the ready-to-run zip and unzip it:
    echo   https://github.com/4-Dimethylamino-78-dihydroxyflavone/dvlt.cu-windows/releases/latest
    echo or run setup.bat to build everything here ^(needs the CUDA toolkit and Visual Studio 2022^).
    exit /b 1
)
if not defined PY (
    echo No Python found: run setup.bat to build the local python\ folder.
    exit /b 1
)
if defined SKY if not defined VSKY (
    echo --sky needs vsky.exe: put it in !ROOT!vsky\ or set VSKY to its path.
    exit /b 1
)
if defined PPISP if not defined VPPISP (
    echo --ppisp needs vppisp.exe: put it in !ROOT!vppisp\ or set VPPISP to its path.
    exit /b 1
)

set "FMTARG="
if defined FORMATS set "FMTARG=--formats %FORMATS%"

REM A .txt input is a LIST FILE (vfold output: one path per line, '#'
REM comments, blank lines skipped) - dvlt.exe expands it natively. A directory
REM whose NAME ends in .txt is still a directory (exist "...\" is true), so it
REM takes the dir branch - not the list branch.
set "IS_LIST="
for %%A in ("%INPUT%") do set "INEXT=%%~xA"
if /i "%INEXT%"==".txt" if not exist "%INPUT%\" set "IS_LIST=1"
REM No output given: a SIBLING of the input named <input>_out_dvlt - never
REM inside an image folder, where COLMAP / 3DGS trainers would ingest it. A
REM folder keeps its whole name, dots included (scene.v2 -> scene.v2_out_dvlt).
REM A list drops its .txt (lists\vfold-60.txt -> lists\vfold-60_out_dvlt), and
REM a list that sits IN an image folder (vfold's default) steps out beside it,
REM folder name first: scene\vfold-60.txt -> scene_vfold-60_out_dvlt.
if not defined OUTPUT if defined IS_LIST for %%A in ("!INPUT!") do (
    set "OUTPUT=%%~dpnA_out_dvlt"
    dir /b /a-d "%%~dpA" 2>nul | findstr /i /l /e ".jpg .jpeg .png" >nul && for %%D in ("%%~dpA.") do set "OUTPUT=%%~fD_%%~nA_out_dvlt"
)
if not defined OUTPUT for %%A in ("!INPUT!") do set "OUTPUT=%%~fA_out_dvlt"

REM Create the output now and prove it writable: dvlt.exe ignores a failed
REM mkdir and crashes on a NULL FILE* after the whole forward pass (read-only
REM media). The probe file doubles as the list count's scratch file - one per
REM output folder, unlike a fixed %TEMP% name shared by concurrent runs.
if not exist "!OUTPUT!\" mkdir "!OUTPUT!" 2>nul
set "PROBE=!OUTPUT!\dvlt_count.tmp"
type nul > "!PROBE!" || (
    echo Cannot create or write the output folder: !OUTPUT!
    exit /b 1
)

REM Count images: the count drives dvlt_long routing + the resolution ladder.
REM A list is counted via python with the SAME rules dvlt_long/postprocess load
REM (utf-8-sig BOM, strip, '#') so a BOM header, an indented comment, or a
REM whitespace-only line can't inflate the count and cross a threshold. Written
REM through the probe file, not for /f: a quoted interpreter path inside
REM for /f's set gets cmd /c's outer-quote stripping and breaks on paths with
REM spaces. It fails CLOSED: no count (python missing or broken, a UTF-16
REM list) stops here instead of single-passing a 300-frame list into an OOM.
REM Directory case: single pass; NTFS globs are case-insensitive, so listing
REM *.jpg and *.JPG as separate patterns double-counts every file.
REM Counted inside the loop, NOT with `find /c`: Git for Windows ships a GNU
REM find.exe, and wherever Git\usr\bin precedes System32 on PATH, `find /c /v ""`
REM walks the whole C: drive into for /f's in-memory buffer (2026-10-03: that
REM ran until the machine bugchecked).
set /a COUNT=0
if not defined IS_LIST for /f "eol=| delims=" %%f in ('dir /b /a-d "%INPUT%" 2^>nul ^| findstr /i /l /e ".jpg .jpeg .png"') do set /a COUNT+=1
if defined IS_LIST (
    set "COUNT="
    "!PY!" -c "import sys,pathlib;print(sum(1 for l in pathlib.Path(sys.argv[1]).read_text('utf-8-sig').splitlines() if l.strip() and not l.strip().startswith('#')))" "!INPUT!" > "!PROBE!"
    set /p COUNT=<"!PROBE!"
)
del "!PROBE!"
for /f "delims=0123456789" %%x in ("!COUNT!") do set "COUNT="
if not defined COUNT (
    echo Could not count the images listed in !INPUT! with !PY! - see the error above.
    exit /b 1
)

REM Every frame limit below was tuned on 16 GB and scales with the VRAM left
REM after a fixed ~2 GB (measured: 36 frames @ 504x336 = 6.0 GB, 80 = 11.6 GB).
REM The total snaps to whole GiB first, as drivers report a little less than
REM the card's size, so every 16 GB card keeps exactly 80/110/140/150 frames
REM and 60/30 chunks. Multi-GPU: the first GPU listed.
REM Inside for /f's command, = and , must be escaped or cmd turns them into spaces.
set "VRAM="
for /f %%v in ('%SystemRoot%\System32\nvidia-smi.exe --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul') do if not defined VRAM set "VRAM=%%v"
for /f "delims=0123456789" %%x in ("!VRAM!") do set "VRAM="
set "VRAMSRC="
if not defined VRAM set "VRAM=16384" & set "VRAMSRC= (assumed: no nvidia-smi)"
set /a "V=(VRAM+512)/1024*1024-2000, N504=(80*V+7192)/14384, N448=(110*V+7192)/14384, N392=(140*V+7192)/14384, NMAX=(150*V+7192)/14384, CHUNK=(60*V+7192)/14384, OVERLAP=(30*V+7192)/14384"

echo === DVLT Reconstruction ===
echo Input:   !INPUT!
echo Output:  !OUTPUT!
echo Images:  %COUNT%
if defined PRESET echo Preset:  -d ^(the D_ settings at the top of reconstruct.bat^)
echo Conf:    %CONF%  (drops lowest-confidence fraction of points)
echo K steps: %K%
echo Python:  !PY!
if not exist "%ROOT%python\python.exe" echo          ^(not the local python\ - setup.bat builds it^)
if defined PPISP echo Prep:    PPISP photometric compensation - !VPPISP!
if defined SKY echo Prep:    sky keyed out and stripped - !VSKY!
echo VRAM:    %VRAM% MiB%VRAMSRC%: 504px up to %N504% frames, 448 to %N448%, 392 to %N392%, 336 to %NMAX%
echo          beyond %NMAX%: dvlt_long, chunks of %CHUNK% overlapping %OVERLAP%
echo.

if %K% GTR 16 (
    echo Warning: K=%K% exceeds the trained range [8,16] - pose accuracy
    echo          degrades beyond 16 and collapses past ~30 ^(paper App. D^).
    echo.
)

REM Beyond the single-pass limit a global-attention pass cannot fit / is far
REM outside the model's trained 2-18 view range: use the long-sequence
REM pipeline (dense IRLS seams + loop closure + Sim3 pose graph). !errorlevel!
REM not %errorlevel%: inside the block the latter is the value at parse time,
REM so a failed dvlt_long used to exit 0. Below 4 GB the scaled chunks overlap
REM by fewer than the 3 frames a seam's camera-centre fallback needs (0 or less
REM at 2 GB, where dvlt_long dies with "invalid chunking"): stop with why.
REM The guard is its own block, not nested in the next: under delayed expansion
REM an exit /b inside a nested block that more commands follow returns 0.
if %COUNT% GTR %NMAX% if %OVERLAP% LSS 3 (
    echo %COUNT% frames exceeds the single-pass limit ^(%NMAX% on %VRAM% MiB^),
    echo and dvlt_long needs a GPU with at least 4 GB: here its chunks would be
    echo %CHUNK% frames overlapping %OVERLAP%, too few to align.
    exit /b 1
)
if %COUNT% GTR %NMAX% (
    echo %COUNT% frames exceeds the single-pass limit ^(%NMAX% on %VRAM% MiB^).
    echo Switching to long-sequence pipeline: dvlt_long.py chunk=%CHUNK% overlap=%OVERLAP%
    echo conf_drop/-c does not apply there: dvlt_long keeps points above 0.75x
    echo each chunk's mean confidence ^(its own merge gate^).
    if defined SKY echo --sky prepares single-pass runs only: ignored here.
    if defined PPISP echo --ppisp prepares single-pass runs only: ignored here.
    echo.
    "!PY!" "!ROOT!dvlt_long.py" "!INPUT!" "!OUTPUT!" --loop-steps %K% --chunk-size %CHUNK% --overlap %OVERLAP% !FMTARG!
    exit /b !errorlevel!
)

REM Auto-select resolution based on frame count
set IMG_SIZE=504
if %COUNT% GTR %N504% set IMG_SIZE=448
if %COUNT% GTR %N448% set IMG_SIZE=392
if %COUNT% GTR %N392% set IMG_SIZE=336

if not %IMG_SIZE%==504 (
    echo Note: %COUNT% frames exceeds %N504% - reducing resolution to %IMG_SIZE%px to fit in %VRAM% MiB VRAM
    echo       ^(lower resolution = more frames fit^)
    echo.
)

REM Preprocessing, all of it under OUTPUT\prep. IMGS is what dvlt reconstructs
REM from; COLOR is what the COLMAP export copies - corrected but never keyed,
REM since a trainer must not learn a black sky. Each step hands the next an
REM images.txt in the input's order, so a list keeps its reference camera.
REM Plain lines, not blocks: these paths may contain ')' (see the parser).
set "IMGS=!INPUT!"
set "COLOR=!INPUT!"
set "PREP=!OUTPUT!\prep"
if not defined PPISP goto prep_sky
echo === Prep: PPISP photometric compensation ^(vppisp^) ===
set "PASS1=!INPUT!"
if defined SKY "!VSKY!" "!INPUT!" --masks "!PREP!\masks" --key "!PREP!\keyed_raw" || exit /b
if defined SKY set "PASS1=!PREP!\keyed_raw\images.txt"
REM The first pass only supplies correspondences, and vppisp's visibility test
REM z-buffers this cloud: -c 0.1 keeps occluders whole. -c 0.5 punched holes
REM that let hidden points pass as seen (measured: residual 18.3 vs 7.6 /255,
REM and exposures off by up to 1.3 EV).
"%DVLT%" -w "%ROOT%model\weights.dvlt" -o "!PREP!\pass1" -c 0.1 -s %IMG_SIZE% -k %K% "!PASS1!" || exit /b
"!VPPISP!" "!INPUT!" --recon "!PREP!\pass1" --size %IMG_SIZE% --out "!PREP!\ppisp" || exit /b
set "IMGS=!PREP!\ppisp\images.txt"
set "COLOR=!IMGS!"
echo.
:prep_sky
if not defined SKY goto reconstruct
echo === Prep: sky keying ^(vsky^) ===
"!VSKY!" "!IMGS!" --masks "!PREP!\masks" --key "!PREP!\keyed" || exit /b
set "IMGS=!PREP!\keyed\images.txt"
echo.
:reconstruct
"%DVLT%" -w "%ROOT%model\weights.dvlt" -o "%OUTPUT%" -c %CONF% -s %IMG_SIZE% -k %K% "!IMGS!"
REM A crash or a loader failure prints nothing of its own: always say it stopped.
if %errorlevel% neq 0 (
    echo dvlt.exe failed with exit code %errorlevel%.
    exit /b %errorlevel%
)
if defined SKY "!VSKY!" --strip "!OUTPUT!" || exit /b

echo.
echo === Post-processing: COLMAP + GLB + orientation fix ===
"%PY%" "%ROOT%dvlt_postprocess.py" "%OUTPUT%" --images "!COLOR!" %FMTARG%
if %errorlevel% neq 0 exit /b %errorlevel%
REM prep is scaffolding: the COLMAP export already holds the corrected frames
REM (only with colmap in the formats; --keep-prep keeps them otherwise).
REM ppisp.json (every fitted parameter, a few KB) is kept as the run's record.
if exist "!PREP!\ppisp\ppisp.json" copy /y "!PREP!\ppisp\ppisp.json" "!OUTPUT!\ppisp.json" >nul
if not defined KEEP_PREP if exist "!PREP!\" rmdir /s /q "!PREP!"
exit /b 0
