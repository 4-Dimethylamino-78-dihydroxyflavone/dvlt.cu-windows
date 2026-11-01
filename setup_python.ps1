<#
dvlt.cu-windows - build python\, the program folder's own interpreter:
embeddable CPython + the pinned requirements.txt. No installer, no registry,
nothing on PATH; reconstruct.bat prefers python\python.exe whenever it exists,
so a copied program folder needs no Python on the target machine.

setup.bat runs this unless python\ already imports its packages, and
package.ps1 copies the result. Run it directly to rebuild, e.g. after editing
requirements.txt:
  powershell -NoProfile -ExecutionPolicy Bypass -File setup_python.ps1
#>
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'     # Invoke-WebRequest is 10x slower with the progress bar
$PyVer = '3.11.9'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py   = "$root\python"
if (Test-Path $py) { Remove-Item -Recurse -Force $py }

$zip = "$env:TEMP\python-$PyVer-embed-amd64.zip"
Invoke-WebRequest -UseBasicParsing "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $zip
Expand-Archive $zip -DestinationPath $py

# Pin sys.path to the folder before anything runs. The stock ._pth's "import
# site" would add the HOST USER's %APPDATA%\Python\Python311\site-packages (a
# ._pth sets isolated mode without -s) ahead of ours: a stray numpy there would
# shadow the bundled one at run time, and if its version matched, pip would not
# install ours at all. An explicit path list runs no site.py, and pip still
# installs into Lib\site-packages: nothing outside python\ is ever imported.
$pth   = Get-ChildItem "$py\python*._pth" | Select-Object -First 1
$pyzip = (Get-ChildItem "$py\python*.zip" | Select-Object -First 1).Name
Set-Content $pth.FullName -Encoding ascii -Value @($pyzip, '.', 'Lib\site-packages')

Invoke-WebRequest -UseBasicParsing 'https://bootstrap.pypa.io/get-pip.py' -OutFile "$env:TEMP\get-pip.py"
& "$py\python.exe" "$env:TEMP\get-pip.py" -q --no-warn-script-location --no-setuptools --no-wheel
if ($LASTEXITCODE) { throw "get-pip failed" }
& "$py\python.exe" -m pip install -q --no-warn-script-location -r "$root\requirements.txt"
if ($LASTEXITCODE) { throw "pip install failed" }
# Scripts\*.exe (pip, f2py) hard-code this build's absolute path: dead in a copy.
Remove-Item -Recurse -Force "$py\Scripts"
& "$py\python.exe" -c "import sys, os, numpy, scipy, PIL, trimesh; base=os.path.dirname(sys.executable); bad=[p for p in sys.path if p and not p.startswith(base)]; assert not bad, bad; print('python ok: numpy', numpy.__version__, 'scipy', scipy.__version__, 'pillow', PIL.__version__, 'trimesh', trimesh.__version__, '-', len(sys.path), 'sys.path entries, all under the python folder')"
if ($LASTEXITCODE) { throw "python self-check failed" }
