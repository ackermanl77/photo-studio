#define MyAppName "Photo Studio"
; Passed by scripts\build_release.ps1 -Installer as /DMyAppVersion=<its -Version>, so the installer
; can't drift from the build it packages; the fallback only applies to a manual ISCC run.
#ifndef MyAppVersion
  #define MyAppVersion "0.2.0"
#endif
#define MyAppPublisher "Photo Studio"
#define MyAppExeName "PhotoStudio.exe"
; The optional GPU pack (AI Upscale's torch/Real-ESRGAN stack, app/gpu_pack.py) becomes an optional
; component when build_release.ps1 -GpuPack has built it next to this repo's release folder.
#define GpuPackManifest AddBackslash(SourcePath) + "..\release-gpu-pack\gpu-pack\gpu-pack.json"
#if FileExists(GpuPackManifest)
  #define WithGpuPack
#endif

[Setup]
AppId={{A8D8F2C7-0F0A-4EB4-9A4F-5E4A42BC0C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Photo Studio
DefaultGroupName=Photo Studio
OutputDir=..\installer-output
OutputBaseFilename=PhotoStudio-Setup-{#MyAppVersion}
; build_release.ps1 -Sign passes /DSign and the "codesign" tool (/Scodesign=...): Inno Setup then
; signs this installer and the uninstaller it writes.
#ifdef Sign
SignTool=codesign
SignedUninstaller=yes
#endif
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\app\assets\photostudio.ico
UninstallDisplayIcon={app}\PhotoStudio.exe
; .photoproject association below writes to HKA (per-machine when installed for all users).
ChangesAssociations=yes
; Lets a non-admin user install "for me only" ({autopf} -> %LOCALAPPDATA%\Programs, HKA -> HKCU),
; from the wizard's dialog or with /CURRENTUSER; without it Setup always demanded elevation.
PrivilegesRequiredOverridesAllowed=commandline dialog

#ifdef WithGpuPack
[Types]
Name: "base"; Description: "Photo Studio (CPU)"
Name: "full"; Description: "Photo Studio with the GPU pack (AI Upscale)"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "main"; Description: "Photo Studio"; Types: base full custom; Flags: fixed
Name: "gpupack"; Description: "GPU pack: AI Upscale with Real-ESRGAN (several GB, an NVIDIA GPU recommended)"; Types: full
#endif

[Tasks]
Name: "assocproject"; Description: "Open .photoproject files with Photo Studio"; GroupDescription: "File associations:"

[Registry]
Root: HKA; Subkey: "Software\Classes\.photoproject"; ValueType: string; ValueName: ""; ValueData: "PhotoStudio.Project"; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project"; ValueType: string; ValueName: ""; ValueData: "Photo Studio Project"; Flags: uninsdeletekey; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\PhotoStudio.exe,0"; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\PhotoStudio.exe"" ""%1"""; Tasks: assocproject

[Files]
; ..\release is build_release.ps1's finished package folder (PhotoStudio.exe + _internal + the
; bundled WebView2Runtime). dist-release\PhotoStudio is only PyInstaller's raw output, without
; WebView2Runtime -- packaging that left the installed app unable to open its window on any PC
; without the Evergreen WebView2 Runtime, the exact case the bundled copy exists for.
Source: "..\release\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
#ifdef WithGpuPack
; Lands as {app}\gpu-pack, where the app looks for it (app/gpu_pack.py).
Source: "..\release-gpu-pack\gpu-pack\*"; DestDir: "{app}\gpu-pack"; Components: gpupack; Flags: recursesubdirs createallsubdirs ignoreversion
#endif

[Icons]
Name: "{group}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"
Name: "{autodesktop}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"

[Code]
// Settings, presets, the library catalog and logs live in %LOCALAPPDATA%\PhotoStudio
// (app.config.user_data_dir()). They are kept on uninstall unless the user agrees here.
// SuppressibleMsgBox, not MsgBox: a /SUPPRESSMSGBOXES silent uninstall answers IDNO (keep the data)
// instead of blocking on a dialog nobody sees.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if SuppressibleMsgBox('Also delete your Photo Studio presets, catalog, downloaded models, logs and any GPU pack kept there (' + ExpandConstant('{localappdata}\PhotoStudio') + ')?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
      DelTree(ExpandConstant('{localappdata}\PhotoStudio'), True, True, True);
end;

[Run]
Filename: "{app}\PhotoStudio.exe"; Description: "Launch Photo Studio"; Flags: nowait postinstall skipifsilent
