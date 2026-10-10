from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image

from app.pipeline import decode_source, run_pipeline
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
    on_output=None,
    gpu_export: bool = False,
) -> tuple[list[tuple[str, bytes]], list[BatchResult]]:
    """`recipe`, when given, takes priority over `preset`: it's a saved custom preset's full recipe
    (app.custom_presets.PresetStore.get()'s return value -- every global tone/color/lens/detail
    setting, converted via app.recipe_options.recipe_to_options()) applied to every file through
    app.pipeline.run_pipeline() -- the same pipeline-ordering code the single-photo editor, the
    backend's HTTP endpoint and the CLI all use, so a batch application can't silently drift out of
    sync with what the UI actually did to produce that preset. `preset` alone (a built-in named look
    from app.presets.PRESETS) runs through the same pipeline as {"preset": preset}.

    `paths` may also hold in-memory files (anything with `.filename` and `.read()`, e.g. FileLike):
    the frontend's Batch Processor passes the uploads directly -- it used to copy them into a temp
    folder under their original names, so two uploads with the same name overwrote each other and
    the files were never deleted. `on_output(name, content)`, when given, receives each finished
    file instead of it being kept in the returned list (stream_to_zip() writes them straight into a
    ZIP on disk, so a large batch isn't held in memory)."""
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
        in_memory = hasattr(raw_path, "read") and hasattr(raw_path, "filename")
        source = Path(raw_path.filename if in_memory else raw_path)
        try:
            # Both branches run app.pipeline.run_pipeline() -- the one step-order implementation
            # the backend and CLI share -- and the result carries the source's EXIF/ICC in .info,
            # which export_profile() writes unless the profile strips metadata.
            run_options = dict(options) if options is not None else ({"preset": preset} if preset and preset != "None" else {})
            content_bytes = raw_path.read() if in_memory else source.read_bytes()
            result_image = run_pipeline(decode_source(content_bytes, source.name), run_options)
            content, suffix, _ = export_profile(result_image, profile, gpu_export=gpu_export)
            output_name = render_name(source, name_template, suffix)
            original_name = output_name
            counter = 2
            while output_name in used_names:
                output_name = f"{Path(original_name).stem}-{counter}{Path(original_name).suffix}"
                counter += 1
            used_names.add(output_name)
            if on_output:
                on_output(output_name, content)
            else:
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


def stream_to_zip(items: list, zip_path: str | Path, **kwargs) -> list[BatchResult]:
    """process_files() with every output written into the ZIP at `zip_path` as soon as it's ready
    (plus report.csv), instead of collecting the whole batch in memory first."""
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        _outputs, report = process_files(items, on_output=bundle.writestr, **kwargs)
        bundle.writestr("report.csv", report_csv(report))
    return report


def make_zip(outputs: list[tuple[str, bytes]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in outputs:
            bundle.writestr(name, content)
    return archive.getvalue()


def export_multiple(
    image: Image.Image,
    profile_names: list[str],
    base_filename: str,
    watermark_text: str | None = None,
    watermark_position: str = "Bottom Right",
    watermark_opacity: int = 50,
    watermark_font_size: int = 32,
    exif: bytes | None = None,
    remove_gps: bool = False,
    gpu_export: bool = False,
    watermark_font: str = "",
) -> bytes:
    """One photo, several export variants (Web/Social/Print/Archive -- see EXPORT_PROFILES) in a
    single ZIP -- the "Export As" multi-size workflow every major editor offers, distinct from
    process_files()'s "one recipe, many photos" batch above. Reuses export_profile() (already the
    single source of truth for what each named profile's format/quality/max_size/metadata-stripping
    -- and now watermarking -- actually is) and make_zip(), rather than a third copy of either.
    The watermark_* args are optional passthroughs to export_profile()/apply_watermark(): a blank
    or omitted watermark_text leaves every variant unwatermarked, exactly as before this arg
    existed."""
    if not profile_names:
        raise ValueError("Select at least one export size")
    outputs = []
    for name in profile_names:
        content, extension, _mime = export_profile(
            image,
            name,
            watermark_text=watermark_text,
            watermark_position=watermark_position,
            watermark_opacity=watermark_opacity,
            watermark_font_size=watermark_font_size,
            watermark_font=watermark_font,
            exif=exif,
            remove_gps=remove_gps,
            gpu_export=gpu_export,
        )
        outputs.append((f"{base_filename}-{name.lower()}{extension}", content))
    return make_zip(outputs)
