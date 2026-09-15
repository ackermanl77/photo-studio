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

# Microsoft Edge WebView2 Runtime is required to open the app's window at all (pywebview's
# edgechromium backend needs it) -- confirmed missing on a second, freshly-provisioned PC. Rather
# than making every user install it themselves, a pinned "Fixed Version" copy (Microsoft's own
# distribution mode for exactly this case -- see the WebView2 downloads page's "Fixed Version"
# section) is bundled as a sibling "WebView2Runtime" folder next to PhotoStudio.exe, and
# app/desktop.py points pywebview at it via webview.settings['WEBVIEW2_RUNTIME_PATH']. Unlike the
# normal Evergreen runtime, a bundled Fixed Version copy does NOT auto-update with Chromium
# security patches, so $WebView2Version should be bumped deliberately every so often when cutting
# a release -- there's no stable "latest" URL to construct by hand; get a new one from
# https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section (pick the
# version + x64 under "Fixed Version", accept the EULA, then read the .cab URL out of the page's
# own HTML -- search it for "filestreamingservice" -- since each version gets its own URL).
$WebView2Version = "153.0.4234.32"
$WebView2CabName = "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64.cab"
$WebView2Url = "https://msedge.sf.dl.delivery.mp.microsoft.com/filestreamingservice/files/c3d95bc1-a0a7-4ca6-aaa1-fa0ac3dd1a37/$WebView2CabName"
$WebView2CacheDir = Join-Path $ProjectRoot "vendor\webview2-cache"
$WebView2Cab = Join-Path $WebView2CacheDir $WebView2CabName
$WebView2ExtractedRoot = Join-Path $WebView2CacheDir "extracted"
$WebView2RuntimeSource = Join-Path $WebView2ExtractedRoot "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64"

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

    # Download once and cache under vendor/ (gitignored) -- a >100MB sanity check guards against a
    # previous run being interrupted mid-download and leaving a truncated .cab behind.
    New-Item -ItemType Directory -Force -Path $WebView2CacheDir | Out-Null
    if (-not (Test-Path $WebView2Cab) -or (Get-Item $WebView2Cab).Length -lt 100MB) {
        Write-Host "Downloading WebView2 Runtime $WebView2Version (x64)..."
        Invoke-WebRequest -Uri $WebView2Url -OutFile $WebView2Cab
    }
    if (-not (Test-Path $WebView2RuntimeSource)) {
        Write-Host "Extracting WebView2 Runtime..."
        New-Item -ItemType Directory -Force -Path $WebView2ExtractedRoot | Out-Null
        & expand.exe -F:* $WebView2Cab $WebView2ExtractedRoot | Out-Null
    }
    if (-not (Test-Path (Join-Path $WebView2RuntimeSource "msedgewebview2.exe"))) {
        throw "WebView2 Runtime extraction did not produce msedgewebview2.exe at $WebView2RuntimeSource"
    }
    Copy-Item $WebView2RuntimeSource (Join-Path $Package "WebView2Runtime") -Recurse

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
