"""Presets & Library tab: saved custom presets (apply, save, delete, import/export), and the
Library Catalog, Batch Processor and System Health tools."""
from __future__ import annotations

import functools
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.batch_release import FileLike, stream_to_zip
from app.config import user_data_dir, user_data_file
from app.custom_presets import PresetStore
from app.library import index_folder, save_rating, thumbnail
from app.major_features import PhotoCatalog
from app.painted_masks import inline_painted_masks
from app.preset_exchange import export_presets, import_presets
from app.presets import PRESETS
from app.professional_tools import EXPORT_PROFILES
from app.raw_import import uploader_types
from app.recipe_options import without_per_photo_only_keys
from app.reliability import image_digest
from app.ui.processing import custom_preset_thumbnail
from app.ui.project_io import open_library_photo
from app.ui.state import build_recipe, request_rerun, tab_visible
from app.ui.theme import static_text
from app.i18n import t, t_option


def presets_section(ctx) -> None:
    st.markdown(t("**Preset Library**"))
    p_store = ctx.preset_store = PresetStore()
    # An unreadable presets.json must not take the page down: degrade to "no presets" plus a way out.
    try:
        p_names = p_store.names()
        p_store_ok = True
    except ValueError:
        p_names, p_store_ok = [], False
        st.error(t("⚠️ The saved presets file can't be read: `{0}`").format(t(p_store.path)))
        if st.button(t("Move it aside and start fresh"), key="p-reset-btn"):
            backup = p_store.reset()
            st.toast(t("Moved the unreadable file to {0}").format(t(backup.name)), icon="🗂️")
            request_rerun()
    ctx.preset_names = p_names
    st.caption(t("Saved: {0}").format(t(', '.join(p_names) if p_names else 'None')))
    if p_names:
        st.markdown(t("**Load a Preset**"))
        p_load_name = st.selectbox(t("Preset to load"), p_names, key="p-load-sel", label_visibility="collapsed", format_func=t_option)
        if tab_visible():
            st.image(custom_preset_thumbnail(ctx.source_digest, ctx.source, p_store.get(p_load_name)), width=96, caption=p_load_name)
        p_load_col, p_delete_col = st.columns(2)
        if p_load_col.button(t("📂 Apply"), key="p-load-btn", use_container_width=True):
            # A preset is a recipe like any other: applied through _pending_restore. AI Upscale is
            # never carried by a preset (without_per_photo_only_keys).
            st.session_state["_pending_restore"] = without_per_photo_only_keys(p_store.get(p_load_name))
            st.toast(t("Applied preset '{0}'").format(t(p_load_name)), icon="📂")
            request_rerun()
        if p_delete_col.button(t("🗑️ Delete"), key="p-delete-btn", use_container_width=True):
            p_store.delete(p_load_name)
            st.toast(t("Deleted preset '{0}'").format(t(p_load_name)), icon="🗑️")
            request_rerun()

    p_new_name = st.text_input(t("New Preset Name"), key="p-name-in")
    if st.button(t("Save Current Look as Preset"), key="p-save-btn", disabled=not p_store_ok):
        if p_new_name.strip():
            # The full recipe (every control plus mask selections, painted masks inlined), without
            # the layer stack or per-photo-only keys.
            p_store.save(p_new_name.strip(), without_per_photo_only_keys(inline_painted_masks(build_recipe())))
            st.toast(t("Preset '{0}' saved").format(t(p_new_name.strip())), icon="💾")
            request_rerun()
        else:
            st.toast(t("Type a name before saving a preset"), icon="⚠️")

    st.markdown(t("---"))
    st.markdown(t("**Export / Import Presets**"))
    if p_names:
        st.download_button(
            t("⬇️ Export all presets"),
            export_presets(p_store),
            file_name="photo-studio-presets.json",
            mime="application/json",
            key="preset-export-btn",
        )
    else:
        st.caption(t("No saved presets to export yet."))
    preset_import_file = st.file_uploader(t("Import presets (.json)"), type=["json"], key="preset-import-upload")
    if preset_import_file and st.button(t("Import"), key="preset-import-btn"):
        try:
            imported = import_presets(p_store, preset_import_file.getvalue())
        except ValueError as error:
            st.toast(t("Could not import presets: {0}").format(t(error)), icon="⚠️")
        else:
            if imported:
                st.toast(t("Imported {0} preset(s)").format(t(len(imported))), icon="📥")
                request_rerun()
            else:
                # Existing names win -- nothing changed, so nothing needs a rerun.
                st.toast(t("No new presets to import — names already exist"), icon="ℹ️")
    st.caption(t("↶ Undo / ↷ Redo and 📌 Named Snapshots are always available in the bar above the canvas; Save Project is in ⬇ Export."))


def library_section(ctx) -> None:
    more_mode = st.radio(t("Tool"), ["Library Catalog", "Batch Processor", "System Health"], horizontal=True, label_visibility="collapsed", key="more-mode", format_func=t_option)
    if more_mode == "Library Catalog":
        lib_folder = st.text_input(t("Folder"), str(Path.home() / "Pictures"), key="lib-f-in")
        # The per-user data folder (migrated from the old ~/.photo-studio on first use).
        lib_db = user_data_file("catalog.sqlite")
        if st.button(t("Scan Folder"), key="lib-btn"):
            lib_progress_bar = st.progress(0.0)
            lib_progress_status = st.empty()

            def _update_scan_progress(index: int, total: int, info) -> None:
                lib_progress_bar.progress(index / total if total else 1.0)
                lib_progress_status.caption(t("Scanning {0}/{1} — {2}").format(t(index), t(total), t(info.get('name', ''))))

            st.session_state.records = index_folder(lib_folder, lib_db, True, progress=_update_scan_progress)
            st.session_state["lib-cursor"] = 0
            lib_progress_bar.progress(1.0)
            lib_progress_status.caption(t("Done — indexed {0} photo(s)").format(t(len(st.session_state.records))))
            st.toast(t("Indexed {0} photo(s)").format(t(len(st.session_state.records))), icon="✅")
        # The catalog is shown without a rescan (it used to appear only after "Scan Folder"
        # in the same session): photos catalogued earlier that are still on disk.
        rec = st.session_state.get("records") or [
            {"path": entry["path"], "name": Path(entry["path"]).name}
            for entry in PhotoCatalog(lib_db).search() if Path(entry["path"]).is_file()
        ]
        q = st.text_input(t("Search"), key="lib-q-in")
        r_min = st.slider(t("Min Stars"), 0, 5, 0, key="lib-r-in")
        search_results = PhotoCatalog(lib_db).search(q, r_min) if rec else []
        if rec and not search_results:
            st.caption(t("No photos match this search / star filter."))
        if search_results and tab_visible():
            # Thumbnail grid (cached JPEG thumbnails under the user data folder, RAW via its
            # embedded preview) with an Edit button that opens the photo in the editor.
            st.caption(t("✏️ Edit opens the photo in the editor with the current adjustments kept (↺ resets them)."))
            grid_cols = st.columns(4)
            for grid_index, entry in enumerate(search_results[:24]):
                entry_path = Path(entry["path"])
                with grid_cols[grid_index % 4]:
                    try:
                        st.image(str(thumbnail(entry_path, user_data_dir() / "thumbs")), caption=f"{'⭐' * int(entry['rating'])} {entry_path.name}", use_container_width=True)
                    except (OSError, ValueError):
                        st.caption(t("{0} (no preview)").format(t(entry_path.name)))
                    if st.button(t("✏️ Edit"), key=f"lib-edit-{image_digest(str(entry_path).encode('utf-8'))[:16]}", use_container_width=True):
                        open_library_photo(entry_path)
                        request_rerun()
        elif search_results:
            st.dataframe(search_results, use_container_width=True)

        if rec:
            st.markdown(t("**Rate & Flag**"))
            st.caption(t("Keyboard: 1-5 rate, 0 clears, X toggles reject, ←/→ move Prev/Next."))
            cursor = max(0, min(st.session_state.get("lib-cursor", 0), len(rec) - 1))
            current_path = rec[cursor]["path"]
            catalog = PhotoCatalog(lib_db)
            existing = catalog.get(current_path)
            is_rejected = "rejected" in existing["tags"]

            st.caption(t(f"{cursor + 1} / {len(rec)} — {Path(current_path).name}" + (" — 🚫 rejected" if is_rejected else "")))
            write_xmp = st.checkbox(
                t("Also write ratings and tags to XMP sidecars"), key="lib-xmp-write",
                help=t("Saves each rating, reject flag and tag change in a .xmp file next to the photo, which "
                     "Lightroom, Capture One, Bridge and darktable read. Existing sidecars keep everything else "
                     "they hold. Ratings already in sidecars are always read when a folder is first scanned."),
            )

            def _save(rating: int, tags) -> None:
                problem = save_rating(catalog, current_path, rating, tags, write_xmp=write_xmp)
                if problem:
                    st.toast(t(problem), icon="⚠️")
            try:
                # The cached thumbnail, not the full-size file decoded on every rerun.
                st.image(str(thumbnail(current_path, user_data_dir() / "thumbs")), width=240)
            except Exception:
                st.caption(t("Preview unavailable."))
            if Path(current_path).is_file() and st.button(t("✏️ Edit this photo"), key="lib-edit-current-btn"):
                open_library_photo(Path(current_path))
                request_rerun()

            with st.container(key="lib-rate-controls"):
                nav_prev, nav_next = st.columns(2)
                if nav_prev.button(t("⬅ Prev"), key="lib-prev-btn", use_container_width=True):
                    st.session_state["lib-cursor"] = (cursor - 1) % len(rec)
                    request_rerun()
                if nav_next.button(t("Next ➡"), key="lib-next-btn", use_container_width=True):
                    st.session_state["lib-cursor"] = (cursor + 1) % len(rec)
                    request_rerun()

                st.caption(t("Rating (keys 1-5, 0 to clear, X to reject, ←/→ to move)"))
                star_cols = st.columns(5)
                for i in range(5):
                    label = "⭐" if i < existing["rating"] else "☆"
                    if star_cols[i].button(t(label), key=f"lib-star-{i + 1}", use_container_width=True):
                        _save(i + 1, existing["tags"])
                        request_rerun()

                clear_col, reject_col = st.columns(2)
                if clear_col.button(t("Clear rating"), key="lib-clear-btn", use_container_width=True):
                    _save(0, existing["tags"])
                    request_rerun()
                reject_label = "🚫 Un-reject" if is_rejected else "🚫 Reject"
                if reject_col.button(t(reject_label), key="lib-reject-btn", use_container_width=True):
                    new_tags = [t for t in existing["tags"] if t != "rejected"] if is_rejected else existing["tags"] + ["rejected"]
                    _save(existing["rating"], new_tags)
                    request_rerun()

                visible_tags = ", ".join(t for t in existing["tags"] if t != "rejected")
                tags_text = st.text_input(t("Tags (comma-separated)"), value=visible_tags, key=f"lib-tags-{cursor}")
                if st.button(t("Save tags"), key="lib-tags-save-btn"):
                    new_tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                    if is_rejected:
                        new_tags.append("rejected")
                    _save(existing["rating"], new_tags)
                    st.toast(t("Tags saved"), icon="🏷️")

            components.html(static_text("library_shortcuts.html"), height=0)
    elif more_mode == "Batch Processor":
        # Every supported format, RAW included (app.raw_import.SUPPORTED_EXTENSIONS) -- this
        # used to accept only JPG/PNG.
        b_uploaded = st.file_uploader(t("Batch Files"), type=uploader_types(), accept_multiple_files=True, key="b-up")
        b_prof = st.selectbox(t("Profile"), list(EXPORT_PROFILES), key="b-p-sel", format_func=t_option)
        b_gpu_export = st.checkbox(
            t("Use GPU for profile resizing"),
            value=False,
            key="b-gpu-export",
            disabled=b_prof not in ("Web", "Social"),
            help=t("Optional CUDA resize for Web/Social output profiles. Needs CUDA-enabled PyTorch; if unavailable or if CUDA fails, the export falls back to CPU."),
        )
        # "None" + built-in named looks (app.presets.PRESETS, a handful of fixed fields) +
        # every saved custom preset (app.custom_presets.PresetStore, a full recipe --
        # everything the Presets tab above can save, minus mask_layers/removal_mask_layers/
        # layer, which are photo-specific and don't transfer across different photos in a
        # batch -- see app.recipe_options.recipe_to_options()'s docstring). The " (custom)"
        # suffix is just this dropdown's own label, stripped back off below before the
        # lookup -- p_store.names() themselves are never touched.
        b_preset_options = ["None", *sorted(PRESETS), *[f"{name} (custom)" for name in ctx.preset_names]]
        b_preset_choice = st.selectbox(
            t("Preset"), b_preset_options, key="b-preset-sel",
            help=t("A saved custom preset applies its full recipe (tone/color/detail/lens) to every photo in the batch -- the same thing Lightroom calls syncing develop settings across a shoot."), format_func=t_option,
        )
        if b_preset_choice.endswith(" (custom)"):
            # Only the custom presets lacked any visual preview at all before -- the 4
            # built-in looks already show themselves clearly by name and don't carry a
            # per-preview pipeline-run cost, so this stays scoped to the gap that existed.
            st.image(
                custom_preset_thumbnail(ctx.source_digest, ctx.source, ctx.preset_store.get(b_preset_choice[: -len(" (custom)")])),
                width=96,
            )
        run_col, stop_col = st.columns([2, 1])
        # Clicking Stop while a batch runs makes Streamlit rerun this script, which ends the
        # running batch at its next progress update (the files finished so far stay in the ZIP).
        stop_col.button(t("⏹ Stop"), key="b-stop-btn", use_container_width=True, help=t("Stop the running batch"))
        if b_uploaded and run_col.button(t("Run Batch"), key="b-run-btn", use_container_width=True):
            if b_preset_choice.endswith(" (custom)"):
                b_preset, b_recipe = None, ctx.preset_store.get(b_preset_choice[: -len(" (custom)")])
            else:
                b_preset, b_recipe = (b_preset_choice if b_preset_choice != "None" else None), None

            b_progress_bar = st.progress(0.0)
            b_progress_status = st.empty()

            def _update_batch_progress(index: int, total: int, last_result) -> None:
                b_progress_bar.progress(index / total if total else 1.0)
                label = f"Processing {index}/{total}"
                if last_result is not None:
                    label += f" — {last_result.source.name} ({last_result.status})"
                b_progress_status.caption(t(label))

            # Straight from the uploaded bytes (no temp files) into a ZIP on disk under the
            # user data folder; only the newest few are kept.
            b_out_dir = user_data_dir() / "batch-output"
            b_out_dir.mkdir(parents=True, exist_ok=True)
            for old_zip in sorted(b_out_dir.glob("batch-*.zip"))[:-2]:
                old_zip.unlink(missing_ok=True)
            b_zip_path = b_out_dir / f"batch-{uuid.uuid4().hex[:12]}.zip"
            rep = stream_to_zip(
                [FileLike(f.name, f.getvalue()) for f in b_uploaded], b_zip_path,
                preset=b_preset, profile=b_prof, progress=_update_batch_progress, recipe=b_recipe,
                gpu_export=b_gpu_export,
            )
            b_progress_bar.progress(1.0)
            b_progress_status.caption(t("Done — {0}/{1} succeeded").format(t(sum((1 for x in rep if x.status == 'ok'))), t(len(rep))))
            st.session_state["_batch_result"] = (str(b_zip_path), [{"source": x.source.name, "status": x.status, "error": x.error or ""} for x in rep])
        batch_result = st.session_state.get("_batch_result")
        if batch_result and Path(batch_result[0]).exists():
            st.dataframe(batch_result[1], use_container_width=True)
            st.download_button(
                t("Download ZIP"), functools.partial(Path(batch_result[0]).read_bytes), "batch.zip", "application/zip", key="b-dl",
            )
    else:
        from app.diagnostics import check_environment, logs_folder, open_folder
        st.dataframe([{"check": c.name, "status": c.status, "message": c.message} for c in check_environment()], use_container_width=True)
        st.caption(t("Logs: `{0}`").format(t(logs_folder())))
        if st.button(t("📂 Open logs folder"), key="diag-open-logs", help=t("Show the app's log files (useful to attach to a bug report)")):
            try:
                open_folder(logs_folder())
            except OSError as error:
                st.toast(t("Couldn't open the folder: {0}").format(t(error)), icon="⚠️")
