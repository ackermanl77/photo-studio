from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError


@dataclass
class RequestReport:
    request_id: str
    status: str
    message: str
    source_name: str | None = None


def new_request_id() -> str:
    return uuid.uuid4().hex


def image_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_image_bytes(content: bytes, filename: str = "image") -> tuple[int, int, str]:
    if not content:
        raise ValueError("Image upload is empty")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            return image.width, image.height, image.format or "unknown"
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Invalid image content: {filename}") from error


def validate_project_archive(content: bytes, max_entries: int = 20, max_uncompressed_bytes: int = 250 * 1024 * 1024) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("Project file is not a valid ZIP archive") from error
    with archive:
        entries = archive.infolist()
        if len(entries) > max_entries:
            raise ValueError("Project contains too many files")
        total_size = sum(entry.file_size for entry in entries)
        if total_size > max_uncompressed_bytes:
            raise ValueError("Project expands beyond the configured size limit")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Project contains an unsafe path")
        if "project.json" not in archive.namelist():
            raise ValueError("Project manifest is missing")
        manifest = json.loads(archive.read("project.json"))
        if manifest.get("version") not in (1, 2) or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unsupported project manifest")
        return manifest
