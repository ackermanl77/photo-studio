# Security policy

## Supported versions

Only the latest release of Photo Studio gets security fixes.

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately through GitHub:
the repository's **Security** tab > **Report a vulnerability**. Include the version, what you
did, what happened, and a file or steps that reproduce it if you can.

You'll get an acknowledgement within 7 days, and a fix or a decision within 30 days for
confirmed issues. Credit is given in the release notes unless you'd rather stay anonymous.

## Scope

In scope: the app and its backend (`/process` and the other HTTP endpoints), decoding of untrusted image, RAW, archive and sidecar files, and the installer. Out of scope: vulnerabilities in third-party packages that are already fixed upstream (update instead), and the AI model weights' own behaviour.
