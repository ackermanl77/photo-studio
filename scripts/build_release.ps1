[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$Sign,
    [string]$CertificateSubject = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $Python)) { $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
$Dist = Join-Path $ProjectRoot "dist-release"
$Build = Join-Path $ProjectRoot "build-release"
$Package = Join-Path $ProjectRoot "PhotoStudio-$Version-Windows"
$Archive = Join-Path $ProjectRoot "PhotoStudio-$Version-Windows.7z"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found. Install Python globally or create .venv."
}

Push-Location $ProjectRoot
try {
    & $Python -m pip install -r requirements-desktop.txt
    & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build PhotoStudio.spec

    if (Test-Path $Package) { Remove-Item $Package -Recurse -Force }
    $oneFolder = Join-Path $Dist "PhotoStudio"
    $oneFile = Join-Path $Dist "PhotoStudio.exe"
    if (Test-Path $oneFolder) {
        Copy-Item $oneFolder $Package -Recurse
    } elseif (Test-Path $oneFile) {
        New-Item -ItemType Directory -Path $Package | Out-Null
        Copy-Item $oneFile (Join-Path $Package "PhotoStudio.exe")
    } else {
        throw "PyInstaller output not found in $Dist"
    }

    if ($Sign) {
        if ([string]::IsNullOrWhiteSpace($CertificateSubject)) {
            throw "-CertificateSubject is required when -Sign is used."
        }
        $exe = Join-Path $Package "PhotoStudio.exe"
        & signtool sign /sha1 $CertificateSubject /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $exe
    }

    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if ($sevenZip) {
        if (Test-Path $Archive) { Remove-Item $Archive -Force }
        & $sevenZip.Source a -t7z -mx=9 -m0=lzma2 -ms=on $Archive (Join-Path $Package "*")
    } else {
        Write-Warning "7z was not found. The unpacked package is ready at $Package."
    }

    $hashPath = "$Archive.sha256"
    if (Test-Path $Archive) {
        (Get-FileHash $Archive -Algorithm SHA256).Hash | Set-Content $hashPath
        Write-Host "Release archive: $Archive"
        Write-Host "SHA-256 file: $hashPath"
    }
    Write-Host "Release folder: $Package"
} finally {
    Pop-Location
}
