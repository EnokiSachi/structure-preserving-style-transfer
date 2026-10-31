$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -B (Join-Path $projectRoot 'scripts\demo_adain_release.py') --root $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Demo exited with an error; see the message above.' }
