@echo off
REM Raw dvlt.exe with the local weights; a -w later on the line still wins
REM (dvlt.exe parses left to right). No setlocal: a trailing endlocal would
REM report exit code 0 to PowerShell for a failed run.
"%~dp0dvlt.exe" -w "%~dp0model\weights.dvlt" %*
exit /b %errorlevel%
