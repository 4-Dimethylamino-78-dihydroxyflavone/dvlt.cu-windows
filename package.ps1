<#
dvlt.cu-windows - assemble the self-contained release: dist\dvlt-win64-<version>\ + .zip

The layout mirrors the repo root, so reconstruct.bat runs unchanged from either:
  dvlt.exe  cublasLt64_12.dll  convert.exe       built and fetched by setup.bat
  model\weights.dvlt  model\LICENSE.txt          NVIDIA checkpoint (bf16 blob) + its license
  python\                                        embeddable CPython + numpy/scipy/pillow/trimesh
  reconstruct.bat  run.bat  dvlt_long.py  dvlt_postprocess.py  requirements.txt
  assets\logo.txt  tools\viewer.html  tools\vendor\  inputs\shoe\
  README.md  LICENSE  NOTICE  THIRD_PARTY.md

Run setup.bat first.
  powershell -ExecutionPolicy Bypass -File package.ps1 [-Version v0.1.0] [-NoPython]
#>
param(
  [string]$Version = "",
  [switch]$NoPython
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if (-not $Version) { $Version = (& git describe --tags --always).Trim() }
$name = "dvlt-win64-$Version"
$out  = Join-Path $root "dist\$name"

$need = 'dvlt.exe', 'cublasLt64_12.dll', 'build\convert.exe', 'model\weights.dvlt', 'model\LICENSE.txt'
if (-not $NoPython) { $need += 'python\python.exe' }
foreach ($f in $need) {
  if (-not (Test-Path $f)) { throw "missing $f - run setup.bat first" }
}

if (Test-Path $out) { Remove-Item -Recurse -Force $out }
foreach ($d in 'model', 'assets', 'tools\vendor', 'inputs\shoe') { New-Item -ItemType Directory -Force "$out\$d" | Out-Null }

Copy-Item ('dvlt.exe', 'cublasLt64_12.dll', 'reconstruct.bat', 'run.bat', 'dvlt_long.py', 'dvlt_postprocess.py',
           'requirements.txt', 'README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY.md') $out
Copy-Item 'build\convert.exe' $out
Copy-Item ('model\weights.dvlt', 'model\LICENSE.txt') "$out\model"
Copy-Item 'assets\logo.txt' "$out\assets"
Copy-Item 'tools\viewer.html' "$out\tools"
Copy-Item 'tools\vendor\*' "$out\tools\vendor"
Copy-Item 'inputs\shoe\*' "$out\inputs\shoe"

if (-not $NoPython) { Copy-Item -Recurse 'python' "$out\python" }

# Windows' own tar.exe (bsdtar, Windows 10+ and CI runners) writes a standard,
# deflated zip with forward-slash entries; Compress-Archive on Windows PowerShell
# 5.1 does not, and a Git-for-Windows GNU tar earlier on PATH would write a tar.
$zipOut = "$root\dist\$name.zip"
if (Test-Path $zipOut) { Remove-Item $zipOut }
Write-Host "== zipping $zipOut"
Push-Location "$root\dist"
& "$env:SystemRoot\System32\tar.exe" -a -c -f "$name.zip" $name
Pop-Location
if ($LASTEXITCODE) { throw "zip failed" }
$mb = [math]::Round((Get-Item $zipOut).Length / 1MB)
Write-Host "== done: $zipOut ($mb MB)"
