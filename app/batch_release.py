from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.image_processor import PhotoProcessor
from app.professional_tools import EXPORT_PROFILES, export_profile
from app.recipe_options import recipe_to_options


@dataclass
class BatchResult:
    source: Path
    output_name: str | None
    status: str
    error: str | None = None


class FileLike:
    """Minimal stand-in for Flask's FileStorage -- just enough (`.filename`, `.read()`) for
    app.backend.process_image() to accept a plain in-memory file, so a caller with a recipe (this
    module's own `recipe` path, and app.frontend_unified_app's custom-preset thumbnail preview)
    can reuse that one already-tested pipeline-ordering implementation in-process instead of
    writing a second copy of "how to apply a full recipe's worth of steps in the right order"."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    def read(self) -> bytes:
        return self._content


def render_name(source: Path, template: str = "{stem}-edited{suffix}", suffix: str = ".jpg") -> str:
    return template.format(stem=source.stem, name=source.name, suffix=suffix)


def process_files(
    paths: list[str | Path],
    preset: str | None = None,
    profile: str = "Web",
    name_template: str = "{stem}-edited{suffix}",
    progress=None,
    should_cancel=None,
    recipe: dict | None = None,
) -> tuple[list[tuple[str, bytes]], list[BatchResult]]:
    """`recipe`, when given, takes priority over `preset`: it's a saved custom preset's full recipe
    (app.custom_presets.PresetStore.get()'s return value -- every global tone/color/lens/detail
    setting, converted via app.recipe_options.recipe_to_options()) applied to every file through
    app.backend.process_image() directly -- the same pipeline-ordering code the single-photo editor
    and app.backend's HTTP endpoint both already use, so a batch application can't silently drift
    out of sync with what the UI actually did to produce that preset. `preset` alone (a built-in
    named look from app.presets.PRESETS) keeps the older, simpler apply_preset() path unchanged."""
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile}")
    options = recipe_to_options(recipe) if recipe else None
    outputs: list[tuple[str, bytes]] = []
    report: list[BatchResult] = []
    used_names: set[str] = set()
    total = len(paths)
    for index, raw_path in enumerate(paths, 1):
        if should_cancel and should_cancel():
            report.extend(BatchResult(Path(item), None, "cancelled", "Cancelled before processing") for item in paths[index - 1 :])
            break
        source = Path(raw_path)
        try:
            if options is not None:
                from app.backend import process_image

                result_image = process_image(FileLike(source.name, source.read_bytes()), dict(options))
                content, suffix, _ = export_profile(result_image, profile)
            else:
                processor = PhotoProcessor(source)
                if preset and preset != "None":
                    from app.presets import apply_preset

                    apply_preset(processor, preset)
                content, suffix, _ = export_profile(processor.image, profile)
            output_name = render_name(source, name_template, suffix)
            original_name = output_name
            counter = 2
            while output_name in used_names:
                output_name = f"{Path(original_name).stem}-{counter}{Path(original_name).suffix}"
                counter += 1
            used_names.add(output_name)
            outputs.append((output_name, content))
            report.append(BatchResult(source, output_name, "ok"))
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, cv2.error) as error:
            # KeyError/IndexError: a caller-supplied name_template with an unknown field (e.g.
            # "{bad}-edited{suffix}") or a stray positional placeholder (e.g. "{0}") raises one of
            # these from render_name()'s str.format() call. Without catching them here alongside
            # every other per-file failure mode, one bad template propagated out of process_files()
            # entirely -- aborting the whole batch and discarding the report for every remaining
            # file, instead of recording just this one file as an error the way every other
            # exception already does. cv2.error: the `recipe` path above can run Denoise/Dehaze
            # (app.backend.process_image()'s full pipeline, unlike the plain apply_preset() path),
            # which -- like app/backend.py's own /process route -- can raise it for a degenerate
            # per-file input (e.g. an image smaller than Non-Local Means' fixed search window).
            report.append(BatchResult(source, None, "error", str(error)))
        finally:
            if progress:
                progress(index, total, report[-1] if report else None)
    return outputs, report


def report_csv(report: list[BatchResult]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["source", "output", "status", "error"])
    for item in report:
        writer.writerow([item.source.name, item.output_name or "", item.status, item.error or ""])
    return output.getvalue().encode("utf-8")


def make_zip(outputs: list[tuple[str, bytes]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in outputs:
            bundle.writestr(name, content)
    return archive.getvalue()
