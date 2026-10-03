"""Finishing step for the optional GPU pack (K5), run by scripts/build_release.ps1 -GpuPack with the
build venv's interpreter -- the same Python the base PhotoStudio.exe is frozen with, which is the
version the pack's compiled packages must match.

    python scripts/gpu_pack_tools.py finalize --pack release-gpu-pack/gpu-pack \
        --base-constraints constraints-cpu.txt --app-version 0.1.0

1. Prunes from <pack>/site-packages every distribution the CPU base already bundles (the names in
   constraints-cpu.txt -- numpy, Pillow, OpenCV, ...) plus the GUI opencv-python that basicsr pulls
   in: at runtime the base's copies are earlier on sys.path and always win, so the pack's would
   only waste space (and a second OpenCV is exactly what the gpu profile build had to avoid).
2. Writes <pack>/gpu-pack.json, which app.gpu_pack checks before using the pack.

Standard library only: it runs before anything is known about the venv beyond pip.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

PACK_FORMAT = 1
ALWAYS_PRUNE = {"opencv-python", "opencv-python-headless", "opencv-contrib-python"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def constraint_names(constraints_file: Path) -> set[str]:
    names = set()
    for line in constraints_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if match:
            names.add(normalize(match.group(0)))
    return names


def prune(site_packages: Path, names: set[str]) -> list[str]:
    """Remove the distributions called `names` (normalized) from `site_packages`, file by file
    from each one's RECORD, then the folders that leaves empty. Returns the removed names."""
    removed = []
    site_packages = site_packages.resolve()
    for dist in list(metadata.distributions(path=[str(site_packages)])):
        name = normalize(dist.metadata["Name"] or "")
        if name not in names:
            continue
        touched_dirs = set()
        for file in dist.files or []:
            path = (site_packages / file).resolve()
            if site_packages not in path.parents:
                continue  # never follow a RECORD entry out of the pack (e.g. ../../Scripts)
            if path.is_file():
                path.unlink()
            touched_dirs.add(path.parent)
        dist_info = Path(str(dist._path)).resolve()  # noqa: SLF001 -- the .dist-info folder itself
        if dist_info.is_dir() and site_packages in dist_info.parents:
            shutil.rmtree(dist_info)
        for folder in sorted(touched_dirs, key=lambda p: len(p.parts), reverse=True):
            while folder != site_packages and site_packages in folder.parents and folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
                folder = folder.parent
        removed.append(name)
    return sorted(removed)


def write_manifest(pack_dir: Path, app_version: str) -> dict:
    versions = {normalize(dist.metadata["Name"] or ""): dist.version for dist in metadata.distributions(path=[str(pack_dir / "site-packages")])}
    manifest = {
        "format": PACK_FORMAT,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "app_version": app_version,
        "torch": versions.get("torch"),
        "packages": dict(sorted(versions.items())),
    }
    (pack_dir / "gpu-pack.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def finalize(pack_dir: Path, base_constraints: Path, app_version: str) -> dict:
    site_packages = pack_dir / "site-packages"
    if not site_packages.is_dir():
        raise SystemExit(f"{site_packages} does not exist -- install the GPU packages first")
    removed = prune(site_packages, constraint_names(base_constraints) | ALWAYS_PRUNE)
    # Console-script launchers pip writes next to the packages point at the build venv's python.
    shutil.rmtree(site_packages / "bin", ignore_errors=True)
    manifest = write_manifest(pack_dir, app_version)
    if not manifest["torch"]:
        raise SystemExit("torch is not in the GPU pack -- the install step failed")
    print(f"GPU pack: removed {len(removed)} base distribution(s); torch {manifest['torch']} for Python {manifest['python']}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--pack", required=True, type=Path)
    final.add_argument("--base-constraints", required=True, type=Path)
    final.add_argument("--app-version", required=True)
    args = parser.parse_args(argv)
    finalize(args.pack, args.base_constraints, args.app_version)


if __name__ == "__main__":
    main()
