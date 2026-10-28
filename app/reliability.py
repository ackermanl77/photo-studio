from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_PIXELS


def new_request_id() -> str:
    return uuid.uuid4().hex


def image_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_pixel_count(width: int, height: int, max_pixels: int | None = None) -> None:
    """Enforces MAX_IMAGE_PIXELS exactly. Setting PIL's own Image.MAX_IMAGE_PIXELS (app/backend.py
    does) is not enough on its own: PIL only *warns* between 1x and 2x that value and raises only
    above 2x, so an image up to twice the configured limit used to be admitted. Raises PIL's own
    DecompressionBombError so callers keep the single existing "pixel limit" error path (a 413 from
    /process)."""
    limit = MAX_IMAGE_PIXELS if max_pixels is None else max_pixels
    if width * height > limit:
        raise Image.DecompressionBombError(
            f"Image size ({width}x{height} = {width * height} pixels) exceeds limit of {limit} pixels"
        )


def validate_image_bytes(content: bytes, filename: str = "image", max_pixels: int | None = None) -> tuple[int, int, str]:
    if not content:
        raise ValueError("Image upload is empty")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height, format_name = image.width, image.height, image.format or "unknown"
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Invalid image content: {filename}") from error
    check_pixel_count(width, height, max_pixels)
    return width, height, format_name


# Entries: project.json + source + legacy layers.json + layers/stack.json + one PNG per image layer +
# masks. 200 leaves room for a large layer stack while still bounding a crafted archive.
def validate_project_archive(content: bytes, max_entries: int = 200, max_uncompressed_bytes: int = 250 * 1024 * 1024) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("Project file is not a valid ZIP archive") from error
    with archive:
        entries = archive.infolist()
        if len(entries) > max_entries:
            raise ValueError("Project contains too many files")
        for entry in entries:
            name = entry.filename
            # PurePosixPath only ever splits on "/" and only treats a leading "/" as absolute.
            # zipfile itself already normalizes "\\" to "/" (on any host where os.sep == "\\",
            # i.e. Windows -- this app's only supported runtime, per app/desktop.py) for both
            # entries it writes and ones it reads back, so a backslash can't actually reach this
            # check un-normalized here -- this line is deliberate defense-in-depth in case this
            # module ever runs somewhere that guarantee doesn't hold, not a reachable gap today.
            # A drive-letter prefix like "C:evil.png" gets no such normalization on any platform,
            # and isn't caught by is_absolute() either (it only recognizes a leading "/"), so that
            # part of this check is a real, currently-open gap being closed here.
            if "\\" in name or (len(name) >= 2 and name[1] == ":" and name[0].isalpha()):
                raise ValueError("Project contains an unsafe path")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Project contains an unsafe path")

        # entry.file_size is attacker-controlled ZIP central-directory metadata, not the actual
        # decompressed byte count -- a crafted archive can declare a tiny file_size while its
        # DEFLATE stream expands far past it (a classic zip bomb), which would sail through a
        # check that only sums declared sizes. Decompress every entry through a bounded read
        # loop instead, so real output crossing the limit aborts immediately rather than trusting
        # what the archive claims about itself.
        total_size = 0
        chunk_size = 1024 * 1024
        for entry in entries:
            with archive.open(entry) as member:
                while True:
                    chunk = member.read(chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_uncompressed_bytes:
                        raise ValueError("Project expands beyond the configured size limit")

        if "project.json" not in archive.namelist():
            raise ValueError("Project manifest is missing")
        try:
            manifest = json.loads(archive.read("project.json"))
        except json.JSONDecodeError as error:
            raise ValueError("Project manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("Project manifest must be an object")
        if manifest.get("version") not in (1, 2) or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unsupported project manifest")

        # Confirm every file the manifest points at actually exists in the archive now, rather
        # than letting a typo'd or hand-edited manifest surface as a bare KeyError deep inside
        # ProjectBundle.load()'s archive.read() calls.
        namelist = set(archive.namelist())
        files = manifest["files"]
        source_name = files.get("source")
        if not isinstance(source_name, str) or source_name not in namelist:
            raise ValueError("Project manifest references a missing source image")
        layers_name = files.get("layers")
        if layers_name is not None and layers_name not in namelist:
            raise ValueError("Project manifest references a missing layers file")
        for mask_name in files.get("masks") or []:
            if f"masks/{mask_name}" not in namelist:
                raise ValueError("Project manifest references a missing mask file")
        stack_name = files.get("layer_stack")
        if stack_name is not None and stack_name not in namelist:
            raise ValueError("Project manifest references a missing layer stack")
        for image_name in files.get("layer_images") or []:
            if f"layers/{image_name}" not in namelist:
                raise ValueError("Project manifest references a missing layer image")
        # The older, v1 PhotoProject format (app/project.py) uses these two keys instead of
        # ProjectBundle's "masks" list, and stores the full in-archive path directly rather than
        # a bare name under masks/ -- checked the same way as "source"/"layers" above.
        for legacy_key in ("removal_mask", "local_mask"):
            legacy_name = files.get(legacy_key)
            if legacy_name is not None and legacy_name not in namelist:
                raise ValueError(f"Project manifest references a missing {legacy_key} file")

        return manifest
