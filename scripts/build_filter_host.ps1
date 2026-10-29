<#
.SYNOPSIS
  Builds the Photoshop filter plug-in host (PS20): PSFilterHost at a pinned commit, then
  tools\filter_host (PhotoStudio.FilterHost.exe), into build\filter_host\.

.DESCRIPTION
  PSFilterHost (Ms-PL, https://github.com/0xC0000054/PSFilterHost) implements the Photoshop plug-in
  API; the helper is the small console program app\filter_plugins.py runs. The source is fetched at
  $PSFilterHostCommit and the build stops if the checked-out commit differs. Needs git and the .NET
  SDK (it builds for .NET Framework 4.5.2 / 4.8, which Windows 10/11 include).

  PhotoStudio.spec bundles build\filter_host\ as filterhost\ when it exists, so run this before
  scripts\build_release.ps1 (which calls it with -FilterHost).
#>
param(
    [string]$PSFilterHostCommit = "c2cfc7c28d41aacd929781fc99d3dafd3fdcd871",
    [string]$PSFilterHostRepo = "https://github.com/0xC0000054/PSFilterHost.git"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Work = Join-Path $ProjectRoot "build\psfilterhost-src"
$Output = Join-Path $ProjectRoot "build\filter_host"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is needed to fetch PSFilterHost." }
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw "The .NET SDK (dotnet) is needed to build the filter host." }

if (-not (Test-Path (Join-Path $Work ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Work) | Out-Null
    git clone --quiet $PSFilterHostRepo $Work
    if ($LASTEXITCODE -ne 0) { throw "Cloning PSFilterHost failed." }
}
git -C $Work fetch --quiet origin
git -C $Work -c advice.detachedHead=false checkout --quiet $PSFilterHostCommit
if ($LASTEXITCODE -ne 0) { throw "PSFilterHost commit $PSFilterHostCommit isn't available." }
$head = (git -C $Work rev-parse HEAD).Trim()
if ($head -ne $PSFilterHostCommit) { throw "PSFilterHost is at $head, expected $PSFilterHostCommit." }

dotnet build (Join-Path $Work "src\PSFilterHost.csproj") -c Release -f net452 -p:Platform=x64 -nologo -v quiet
if ($LASTEXITCODE -ne 0) { throw "Building PSFilterHost failed (exit code $LASTEXITCODE)." }
$libraryDir = Join-Path $Work "src\bin\x64\Release\net452"

if (Test-Path $Output) { Remove-Item -Recurse -Force $Output }
dotnet build (Join-Path $ProjectRoot "tools\filter_host\PhotoStudio.FilterHost.csproj") -c Release -nologo -v quiet `
    "-p:PSFilterHostDir=$libraryDir" -o $Output
if ($LASTEXITCODE -ne 0) { throw "Building PhotoStudio.FilterHost failed (exit code $LASTEXITCODE)." }

# The plug-in host ships PSFilterHost's licences (Ms-PL, and the Paint.NET licence for code derived from it).
Copy-Item (Join-Path $Work "License.txt") (Join-Path $Output "PSFilterHost-License.txt")
Copy-Item (Join-Path $Work "License-pdn.txt") (Join-Path $Output "PSFilterHost-License-pdn.txt")
Get-ChildItem $Output -Filter *.pdb | Remove-Item
Get-ChildItem $Output -Filter *.xml | Remove-Item

$exe = Join-Path $Output "PhotoStudio.FilterHost.exe"
& $exe list $Output | Out-Null
if ($LASTEXITCODE -ne 0) { throw "The built filter host doesn't run (exit code $LASTEXITCODE)." }
Write-Host "Filter host: $exe (PSFilterHost $PSFilterHostCommit)"
