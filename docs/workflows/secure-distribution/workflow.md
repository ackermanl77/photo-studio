# Secure Windows Distribution Workflow

## Goal

Produce a transparent Windows release with fewer antivirus false positives and a verifiable download.

## Release steps

1. Start from a clean Windows build environment.
2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
```

3. Build the one-folder application with `PhotoStudio.spec`.
4. Confirm UPX is disabled in the spec.
5. Optionally sign `PhotoStudio.exe` with a trusted Authenticode certificate.
6. Compress the complete folder with 7-Zip using LZMA2.
7. Generate and publish the SHA-256 checksum.
8. Test extraction and launch on a clean Windows account.
9. Build the Inno Setup installer from `installer/PhotoStudio.iss`.
10. Sign the installer before public distribution.

## Automated build

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

Signed release:

```powershell
.\scripts\build_release.ps1 -Version "0.1.0" -Sign -CertificateSubject "YOUR_CERTIFICATE_THUMBPRINT"
```

## Security defaults

- Flask binds to `127.0.0.1`.
- Upload size and image pixel limits are enabled.
- UPX compression is disabled.
- The build does not download or execute external code.
- The distribution includes the complete dependency folder rather than extracting a hidden one-file payload.

Code signing reduces warnings but cannot guarantee that every antivirus vendor will classify the application identically.
