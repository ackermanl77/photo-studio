[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [switch]$Sign,
    # SHA-1 thumbprint of the code-signing certificate (signtool /sha1). Was misleadingly named
    # -CertificateSubject although it was always passed to /sha1; the old name still works.
    [Alias("CertificateSubject")]
    [string]$CertificateThumbprint = "",
    # Build profile. Every build runs from a fresh virtual environment the script creates itself
    # (build-venv\<profile>\), so what ships depends only on the requirement files -- never on
    # whatever happens to be installed in the developer's own interpreter (the old default build
    # bundled rawpy/CUDA torch only because the dev interpreter had them, and a CI build shipped
    # without RAW support).
    #   cpu: requirements-desktop.txt + requirements-raw.txt (RAW import is part of every build)
    #   gpu: cpu + requirements-gpu.txt (CUDA torch + Real-ESRGAN for AI Upscale). torch's CUDA
    #        wheels come from PyTorch's own index -- see requirements-gpu.txt's header; pass it via
    #        -ExtraIndexUrl, e.g. https://download.pytorch.org/whl/cu130
    # A constraints-<profile>.txt next to this repo's requirement files, when present, pins every
    # transitive version (pip -c) for a reproducible build.
    [ValidateSet("cpu", "gpu")]
    [string]$BuildProfile = "cpu",
    [string]$ExtraIndexUrl = "",
    # The *base* interpreter the build venv is created from (not the one PyInstaller runs in).
    # Default: `py -3.14` when the Python launcher is present, else whatever `python` is on PATH.
    [string]$Python = "",
    # Reuse an existing build-venv\<profile>\ instead of recreating it (faster local rebuilds;
    # never used by CI).
    [switch]$ReuseVenv,
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

# OneDrive (this repo lives under it) briefly locks files it is syncing, so deleting a large tree
# such as build-venv\ or release\ can fail with "access denied" on one file. Retry before giving up.
function Remove-TreeWithRetry([string]$Path) {
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        try {
            Remove-Item $Path -Recurse -Force
            return
        } catch {
            if ($attempt -eq 6) { throw }
            Write-Warning "Removing $Path failed (attempt $attempt, retrying): $($_.Exception.Message)"
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $Python = (& $launcher.Source -3.14 -c "import sys; print(sys.executable)" 2>$null)
    }
    if (-not $Python) {
        $onPath = Get-Command python -ErrorAction SilentlyContinue
        if ($onPath) { $Python = $onPath.Source }
    }
}
$BuildVenv = Join-Path $ProjectRoot "build-venv\$BuildProfile"
$VenvPython = Join-Path $BuildVenv "Scripts\python.exe"
$SelfTestReport = Join-Path $ProjectRoot "build-release-self-test.json"
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
# SHA-256 of that version's .cab, checked after every download (and against a cached copy) so a
# corrupted or substituted download can't end up inside a release. Update together with
# $WebView2Version. $WebView2PinnedOn is read by .github/workflows/webview2-age.yml, which opens an
# issue once the pin is more than 90 days old.
$WebView2Sha256 = "2CB653A74426F0AA802C2396775C6BC674FD662D5396BD677F47BFA6E12EBA9C"
$WebView2PinnedOn = "2026-09-15"
$WebView2CabName = "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64.cab"
$WebView2Url = "https://msedge.sf.dl.delivery.mp.microsoft.com/filestreamingservice/files/c3d95bc1-a0a7-4ca6-aaa1-fa0ac3dd1a37/$WebView2CabName"
$WebView2CacheDir = Join-Path $ProjectRoot "vendor\webview2-cache"
$WebView2Cab = Join-Path $WebView2CacheDir $WebView2CabName
$WebView2ExtractedRoot = Join-Path $WebView2CacheDir "extracted"
$WebView2RuntimeSource = Join-Path $WebView2ExtractedRoot "Microsoft.WebView2.FixedVersionRuntime.$WebView2Version.x64"

if (-not $Python -or -not (Test-Path $Python)) {
    throw "Base Python interpreter not found. Install Python 3.14 (or pass -Python <path>)."
}

Push-Location $ProjectRoot
try {
    # $ErrorActionPreference doesn't cover native commands -- check their exit codes explicitly, so a
    # failed install or build stops here instead of packaging (and cleaning up around) a stale output.
    if ((Test-Path $BuildVenv) -and -not $ReuseVenv) { Remove-TreeWithRetry $BuildVenv }
    if (-not (Test-Path $VenvPython)) {
        & $Python -m venv $BuildVenv
        if ($LASTEXITCODE -ne 0) { throw "Creating the build venv failed (exit code $LASTEXITCODE)." }
    }
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit code $LASTEXITCODE)." }
    $requirements = @("-r", "requirements-desktop.txt", "-r", "requirements-raw.txt")
    if ($BuildProfile -eq "gpu") { $requirements += @("-r", "requirements-gpu.txt") }
    $constraints = Join-Path $ProjectRoot "constraints-$BuildProfile.txt"
    if (Test-Path $constraints) { $requirements += @("-c", $constraints) } else { Write-Warning "No constraints-$BuildProfile.txt -- transitive versions are not pinned for this build." }
    if ($ExtraIndexUrl) { $requirements += @("--extra-index-url", $ExtraIndexUrl) }
    if ($BuildProfile -eq "gpu") {
        # basicsr 1.4.2 is sdist-only and its setup.py fails on Python 3.13+ ("KeyError:
        # '__version__'", PEP 667), and it imports the removed torchvision functional_tensor module
        # -- see requirements-gpu.txt. Patch a hash-verified copy of the sdist and install it first;
        # the main install below then finds basicsr already satisfied instead of building it.
        $basicsrCache = Join-Path $ProjectRoot "vendor\basicsr-cache"
        $basicsrTarball = Join-Path $basicsrCache "basicsr-1.4.2.tar.gz"
        $basicsrSource = Join-Path $basicsrCache "basicsr-1.4.2"
        New-Item -ItemType Directory -Force -Path $basicsrCache | Out-Null
        if (-not (Test-Path $basicsrTarball)) {
            Invoke-WebRequest -UseBasicParsing -OutFile $basicsrTarball -Uri "https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz"
        }
        $basicsrHash = (Get-FileHash $basicsrTarball -Algorithm SHA256).Hash
        if ($basicsrHash -ne "B89B595A87EF964CDA9913B4D99380DDB6554C965577C0C10CB7B78E31301E87") {
            Remove-Item $basicsrTarball -Force
            throw "basicsr-1.4.2.tar.gz failed its SHA-256 check (got $basicsrHash) -- deleted it; rerun to download again."
        }
        if (Test-Path $basicsrSource) { Remove-TreeWithRetry $basicsrSource }
        # Windows' own bsdtar by full path: a Git-for-Windows tar earlier on PATH reads "C:" as a host.
        & (Join-Path $env:SystemRoot "System32\tar.exe") -xzf $basicsrTarball -C $basicsrCache
        if ($LASTEXITCODE -ne 0) { throw "Extracting basicsr failed (exit code $LASTEXITCODE)." }
        $patches = @(
            @{ File = "setup.py"; Old = "        exec(compile(f.read(), version_file, 'exec'))`n    return locals()['__version__']"; New = "        ns = {}`n        exec(compile(f.read(), version_file, 'exec'), ns)`n    return ns['__version__']" },
            # torch/numpy/cython are only needed to compile the optional CUDA extensions, which this
            # build doesn't use; dropping them lets setup.py run without build isolation.
            @{ File = "setup.py"; Old = "        setup_requires=['cython', 'numpy', 'torch'],`n"; New = "" },
            @{ File = "basicsr\data\degradations.py"; Old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"; New = "from torchvision.transforms.functional import rgb_to_grayscale" }
        )
        foreach ($patch in $patches) {
            $path = Join-Path $basicsrSource $patch.File
            $text = [IO.File]::ReadAllText($path).Replace("`r`n", "`n")
            if (-not $text.Contains($patch.Old)) { throw "basicsr patch target not found in $($patch.File)." }
            [IO.File]::WriteAllText($path, $text.Replace($patch.Old, $patch.New))
        }
        $buildTools = @("setuptools")
        if (Test-Path $constraints) { $buildTools += @("-c", $constraints) }
        & $VenvPython -m pip install @buildTools
        if ($LASTEXITCODE -ne 0) { throw "Installing setuptools failed (exit code $LASTEXITCODE)." }
        $basicsrInstall = @("--no-build-isolation", $basicsrSource)
        if (Test-Path $constraints) { $basicsrInstall += @("-c", $constraints) }
        if ($ExtraIndexUrl) { $basicsrInstall += @("--extra-index-url", $ExtraIndexUrl) }
        & $VenvPython -m pip install @basicsrInstall
        if ($LASTEXITCODE -ne 0) { throw "Installing the patched basicsr failed (exit code $LASTEXITCODE)." }
    }
    & $VenvPython -m pip install @requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit code $LASTEXITCODE)." }
    if ($BuildProfile -eq "gpu") {
        # basicsr/realesrgan/facexlib/gfpgan pull in opencv-python (with GUI), the app uses
        # opencv-python-headless, and both install the same cv2 package -- whichever went in last
        # owns the files, and removing either later breaks cv2. Keep only the headless build.
        & $VenvPython -m pip uninstall -y opencv-python
        if ($LASTEXITCODE -ne 0) { throw "pip uninstall opencv-python failed (exit code $LASTEXITCODE)." }
        $headless = @("--force-reinstall", "--no-deps", "opencv-python-headless")
        if (Test-Path $constraints) { $headless += @("-c", $constraints) }
        & $VenvPython -m pip install @headless
        if ($LASTEXITCODE -ne 0) { throw "Reinstalling opencv-python-headless failed (exit code $LASTEXITCODE)." }
    }
    & $VenvPython -m PyInstaller --noconfirm --distpath $Dist --workpath $Build PhotoStudio.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit code $LASTEXITCODE)." }

    # Move (not copy) PyInstaller's output into release\: on the same drive that's a rename, so the
    # multi-GB build is never duplicated on disk, not even briefly.
    if (Test-Path $ReleaseFolder) { Remove-TreeWithRetry $ReleaseFolder }
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
    $cabHash = (Get-FileHash $WebView2Cab -Algorithm SHA256).Hash
    if ($cabHash -ne $WebView2Sha256) {
        Remove-Item $WebView2Cab -Force
        throw "WebView2 runtime download failed its SHA-256 check (got $cabHash, expected $WebView2Sha256) -- deleted it; rerun to download again."
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

    # Post-build smoke test (app/desktop.py --self-test): every bundled module imports, rawpy and a
    # face detector are present. The windowed exe has no console, so the report is read from a file.
    if (Test-Path $SelfTestReport) { Remove-Item $SelfTestReport -Force }
    $selfTest = Start-Process -FilePath (Join-Path $ReleaseFolder "PhotoStudio.exe") -ArgumentList @("--self-test", "--self-test-report", "`"$SelfTestReport`"") -Wait -PassThru
    if (Test-Path $SelfTestReport) { Get-Content $SelfTestReport | Write-Host }
    if ($selfTest.ExitCode -ne 0) { throw "PhotoStudio.exe --self-test failed (exit code $($selfTest.ExitCode)) -- see $SelfTestReport." }
    if ($BuildProfile -eq "gpu") {
        $report = Get-Content $SelfTestReport -Raw | ConvertFrom-Json
        if (-not $report.capabilities.ai_upscale_installed) { throw "gpu profile build is missing the AI Upscale packages." }
    }

    if ($Sign) {
        if ([string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
            throw "-CertificateThumbprint is required when -Sign is used."
        }
        $exe = Join-Path $ReleaseFolder "PhotoStudio.exe"
        & signtool sign /sha1 $CertificateThumbprint /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $exe
        if ($LASTEXITCODE -ne 0) { throw "signtool failed (exit code $LASTEXITCODE) -- the release is unsigned." }
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
        if (Test-Path $leftover) { Remove-TreeWithRetry $leftover; Write-Host "Cleaned up $leftover" }
    }
} finally {
    Pop-Location
}
