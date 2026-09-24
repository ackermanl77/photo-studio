[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$Sign,
    [string]$CertificateSubject = "",
    # Build with a specific interpreter instead of the default lookup below -- e.g. the project's
    # own .venv when that's where requirements-gpu.txt (CUDA torch + Real-ESRGAN, for AI Upscale)
    # is installed; PyInstaller only bundles what the interpreter running it can import.
    [string]$Python = "",
    # Skip the 7z archive. A GPU build bundles several GB of CUDA libraries, and -mx=9 LZMA over
    # that takes a long time -- skip it when only the unpacked release\ folder is needed.
    [switch]$SkipArchive,
    # Also build the Inno Setup installer (installer\PhotoStudio.iss) from the finished release\
    # folder, passing -Version through so the installer's version matches the build. Needs ISCC.exe
    # (Inno Setup 6) on PATH or in its default install location.
    [switch]$Installer,
    # Keep PyInstaller's build-release\ and dist-release\ folders after a successful build instead of
    # deleting them. build-release\ is PyInstaller's analysis cache, so keeping it makes the next
    # rebuild faster at the cost of a few hundred MB; dist-release\ is empty either way, since its
    # output is moved into release\.
    [switch]$KeepBuildFiles
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = "C:\Users\MELEJ\AppData\Local\Python\pythoncore-3.14-64\python.exe" }
if (-not (Test-Path $Python)) { $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    # Neither of the two dev-machine-specific paths above exist on a CI runner (e.g. the
    # release-build GitHub Actions workflow, which sets up Python via actions/setup-python and
    # puts it on PATH instead) -- fall back to whatever "python" resolves to there.
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $Python = $onPath.Source }
}
$Dist = Join-Path $ProjectRoot "dist-release"
$Build = Join-Path $ProjectRoot "build-release"
# The only unpacked copy of the finished app: a fixed name, so the built .exe always has one stable,
# predictable place to find it (release\PhotoStudio.exe) without knowing the current version number.
# Only the distributable archive carries the version in its name.
$ReleaseFolder = Join-Path $ProjectRoot "release"
$Archive = Join-Path $ProjectRoot "PhotoStudio-$Version-Windows.7z"
# Older versions of this script also kept a version-named copy of release\ here -- a full duplicate
# of a multi-GB GPU build. It's no longer created; a leftover one is removed after a successful build.
$LegacyPackage = Join-Path $ProjectRoot "PhotoStudio-$Version-Windows"

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
    # $ErrorActionPreference doesn't cover native commands -- check their exit codes explicitly, so a
    # failed install or build stops here instead of packaging (and cleaning up around) a stale output.
    & $Python -m pip install -r requirements-desktop.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit code $LASTEXITCODE)." }
    & $Python -m PyInstaller --noconfirm --distpath $Dist --workpath $Build PhotoStudio.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)." }

    # Move (not copy) PyInstaller's output into release\: on the same drive that's a rename, so the
    # multi-GB build is never duplicated on disk, not even briefly.
    if (Test-Path $ReleaseFolder) { Remove-Item $ReleaseFolder -Recurse -Force }
    $oneFolder = Join-Path $Dist "PhotoStudio"
    $oneFile = Join-Path $Dist "PhotoStudio.exe"
    if (Test-Path $oneFolder) {
        Move-Item $oneFolder $ReleaseFolder
    } elseif (Test-Path $oneFile) {
        New-Item -ItemType Directory -Path $ReleaseFolder | Out-Null
        Move-Item $oneFile (Join-Path $ReleaseFolder "PhotoStudio.exe")
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
    Copy-Item $WebView2RuntimeSource (Join-Path $ReleaseFolder "WebView2Runtime") -Recurse

    if ($Sign) {
        if ([string]::IsNullOrWhiteSpace($CertificateSubject)) {
            throw "-CertificateSubject is required when -Sign is used."
        }
        $exe = Join-Path $ReleaseFolder "PhotoStudio.exe"
        & signtool sign /sha1 $CertificateSubject /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $exe
    }

    $hashPath = "$Archive.sha256"
    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if ($SkipArchive) {
        # An archive left over from an earlier build no longer matches release\ -- remove it rather
        # than leave a stale copy that looks like this build's.
        foreach ($stale in @($Archive, $hashPath)) {
            if (Test-Path $stale) { Remove-Item $stale -Force; Write-Host "Removed stale $stale" }
        }
        Write-Host "Skipping 7z archive (-SkipArchive)."
    } elseif ($sevenZip) {
        if (Test-Path $Archive) { Remove-Item $Archive -Force }
        & $sevenZip.Source a -t7z -mx=9 -m0=lzma2 -ms=on $Archive (Join-Path $ReleaseFolder "*")
        if ($LASTEXITCODE -ne 0) { throw "7z failed (exit code $LASTEXITCODE)." }
        (Get-FileHash $Archive -Algorithm SHA256).Hash | Set-Content $hashPath
        Write-Host "Release archive: $Archive"
        Write-Host "SHA-256 file: $hashPath"
    } else {
        Write-Warning "7z was not found. The unpacked build is ready at $ReleaseFolder."
    }
    Write-Host "Release folder: $ReleaseFolder"

    if ($Installer) {
        $iscc = Get-Command iscc -ErrorAction SilentlyContinue
        $isccPath = if ($iscc) { $iscc.Source } else { Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe" }
        if (-not (Test-Path $isccPath)) { throw "ISCC.exe (Inno Setup 6) not found -- needed for -Installer." }
        & $isccPath "/DMyAppVersion=$Version" (Join-Path $ProjectRoot "installer\PhotoStudio.iss")
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit code $LASTEXITCODE)." }
    }

    # Only reached when everything above succeeded -- a failed build keeps its intermediates for
    # debugging.
    $leftovers = @($LegacyPackage)
    if (-not $KeepBuildFiles) { $leftovers += @($Build, $Dist) }
    foreach ($leftover in $leftovers) {
        if (Test-Path $leftover) { Remove-Item $leftover -Recurse -Force; Write-Host "Cleaned up $leftover" }
    }
} finally {
    Pop-Location
}
