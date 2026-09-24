#define MyAppName "Photo Studio"
; Passed by scripts\build_release.ps1 -Installer as /DMyAppVersion=<its -Version>, so the installer
; can't drift from the build it packages; the fallback only applies to a manual ISCC run.
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#define MyAppPublisher "Photo Studio"
#define MyAppExeName "PhotoStudio.exe"

[Setup]
AppId={{A8D8F2C7-0F0A-4EB4-9A4F-5E4A42BC0C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Photo Studio
DefaultGroupName=Photo Studio
OutputDir=..\installer-output
OutputBaseFilename=PhotoStudio-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; ..\release is build_release.ps1's finished package folder (PhotoStudio.exe + _internal + the
; bundled WebView2Runtime). dist-release\PhotoStudio is only PyInstaller's raw output, without
; WebView2Runtime -- packaging that left the installed app unable to open its window on any PC
; without the Evergreen WebView2 Runtime, the exact case the bundled copy exists for.
Source: "..\release\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"
Name: "{autodesktop}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"

[Run]
Filename: "{app}\PhotoStudio.exe"; Description: "Launch Photo Studio"; Flags: nowait postinstall skipifsilent
