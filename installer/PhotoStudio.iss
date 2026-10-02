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
SetupIconFile=..\app\assets\photostudio.ico
UninstallDisplayIcon={app}\PhotoStudio.exe
; .photoproject association below writes to HKA (per-machine when installed for all users).
ChangesAssociations=yes

[Tasks]
Name: "assocproject"; Description: "Open .photoproject files with Photo Studio"; GroupDescription: "File associations:"

[Registry]
Root: HKA; Subkey: "Software\Classes\.photoproject"; ValueType: string; ValueName: ""; ValueData: "PhotoStudio.Project"; Flags: uninsdeletevalue; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project"; ValueType: string; ValueName: ""; ValueData: "Photo Studio Project"; Flags: uninsdeletekey; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\PhotoStudio.exe,0"; Tasks: assocproject
Root: HKA; Subkey: "Software\Classes\PhotoStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\PhotoStudio.exe"" ""%1"""; Tasks: assocproject

[Files]
; ..\release is build_release.ps1's finished package folder (PhotoStudio.exe + _internal + the
; bundled WebView2Runtime). dist-release\PhotoStudio is only PyInstaller's raw output, without
; WebView2Runtime -- packaging that left the installed app unable to open its window on any PC
; without the Evergreen WebView2 Runtime, the exact case the bundled copy exists for.
Source: "..\release\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"
Name: "{autodesktop}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"

[Code]
; Settings, presets, the library catalog and logs live in %LOCALAPPDATA%\PhotoStudio (app.config.
; user_data_dir()). They are kept on uninstall unless the user agrees here.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if MsgBox('Also delete your Photo Studio presets, catalog, downloaded models and logs (' + ExpandConstant('{localappdata}\PhotoStudio') + ')?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(ExpandConstant('{localappdata}\PhotoStudio'), True, True, True);
end;

[Run]
Filename: "{app}\PhotoStudio.exe"; Description: "Launch Photo Studio"; Flags: nowait postinstall skipifsilent
