# Builds dist\KickTheFly.exe: one file with the game, Python, numpy/scipy/pygame and the brain pack inside.
# Needs data\kick_brain.npz first:  python brainpack.py build   (after the connectome/layout builds in README.md)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path data\kick_brain.npz)) { throw "data\kick_brain.npz missing: run 'python brainpack.py build' first" }
if (-not (Test-Path .venv-build)) {
    py -3.11 -m venv .venv-build
    .\.venv-build\Scripts\python.exe -m pip install numpy scipy pygame pyinstaller pillow
}
.\.venv-build\Scripts\python.exe tools\make_icon.py build\icon.png
.\.venv-build\Scripts\pyinstaller.exe --noconfirm --clean --onefile --windowed --name KickTheFly `
    --icon build\icon.png --add-data "data\kick_brain.npz;." `
    --exclude-module connectome.loader --exclude-module connectome.layout --exclude-module pyarrow `
    --exclude-module tkinter --exclude-module matplotlib --exclude-module PIL `
    kick_the_fly.py
.\dist\KickTheFly.exe --smoke 3 | Out-Null
Get-Item dist\KickTheFly.exe | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
