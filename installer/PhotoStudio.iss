#define MyAppName "Photo Studio"
#define MyAppVersion "0.1.0"
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
Source: "..\dist-release\PhotoStudio\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"
Name: "{autodesktop}\Photo Studio"; Filename: "{app}\PhotoStudio.exe"

[Run]
Filename: "{app}\PhotoStudio.exe"; Description: "Launch Photo Studio"; Flags: nowait postinstall skipifsilent
