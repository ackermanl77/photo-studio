"""Presets & Library tab: saved custom presets (apply, save, delete, import/export), and the
Library Catalog, Batch Processor and System Health tools."""
from __future__ import annotations

import functools
import uuid
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.batch_release import FileLike, stream_to_zip
from app.config import user_data_dir, user_data_file
from app.custom_presets import PresetStore
from app.culling import is_likely_blurry
from app.library import index_folder, save_rating, thumbnail
from app.major_features import COLOR_LABELS, PhotoCatalog
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


LABEL_DOTS = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢", "Blue": "🔵", "Purple": "🟣"}
SORT_LABELS = {"rating": "Rating", "sharpest": "Sharpest first", "blurriest": "Blurriest first", "name": "Name"}


def grid_caption(entry: dict) -> str:
    """A thumbnail's caption: stars, colour label, a blur warning, the file name."""
    parts = ["⭐" * int(entry["rating"])]
    if entry.get("label"):
        parts.append(LABEL_DOTS[entry["label"]])
    if is_likely_blurry(entry.get("focus")):
        parts.append("〰️")
    parts.append(Path(entry["path"]).name)
    return " ".join(part for part in parts if part)


def score_caption(entry: dict) -> str:
    text = t("Focus {0}/100 · clipped highlights {1} % · shadows {2} %").format(
        f"{entry['focus']:.0f}", f"{entry['highlights']:.1f}", f"{entry['shadows']:.1f}")
    if is_likely_blurry(entry["focus"]):
        text += " · " + t("〰️ likely blurry")
    return text


# ---- Library organisation (F21): saved searches, collections, virtual copies, two-up compare ----
def apply_pending_search(collection_names) -> None:
    """A saved search chosen on the last run fills the filter widgets (before they are created)."""
    pending = st.session_state.pop("_lib_pending_search", None)
    if pending is None:
        return
    label = pending.get("label")
    st.session_state["lib-q-in"] = str(pending.get("text", ""))
    st.session_state["lib-r-in"] = max(0, min(5, int(pending.get("minimum_rating", 0) or 0)))
    st.session_state["lib-label-filter"] = "Any" if label is None else ("None" if label == "" else label)
    st.session_state["lib-sort"] = pending.get("sort") if pending.get("sort") in SORT_LABELS else "rating"
    collection = pending.get("collection")
    st.session_state["lib-collection-filter"] = collection if collection in collection_names else "Any"


def saved_searches_row(catalog: PhotoCatalog, current: dict) -> None:
    searches = catalog.saved_searches()
    with st.expander(t("Saved searches"), expanded=False, key="sec-lib-searches"):
        if searches:
            pick_col, apply_col, delete_col = st.columns([3, 1, 1])
            chosen = pick_col.selectbox(t("Saved search"), list(searches), key="lib-saved-search",
                                        label_visibility="collapsed")
            if apply_col.button(t("Apply"), key="lib-search-apply", use_container_width=True):
                st.session_state["_lib_pending_search"] = searches[chosen]
                request_rerun()
            if delete_col.button(t("Delete"), key="lib-search-delete", use_container_width=True):
                catalog.delete_search(chosen)
                request_rerun()
        else:
            st.caption(t("No saved searches yet: set the search and filters above, then save them here."))
        name_col, save_col = st.columns([3, 1])
        name = name_col.text_input(t("Name for the current search"), key="lib-search-name",
                                   label_visibility="collapsed", placeholder=t("Name for the current search"))
        if save_col.button(t("Save search"), key="lib-search-save", use_container_width=True):
            if name.strip():
                catalog.save_search(name, current)
                st.toast(t("Saved search '{0}'").format(name.strip()), icon="🔎")
                request_rerun()
            else:
                st.toast(t("Type a name for the search first"), icon="⚠️")


def open_virtual_copy(copy: dict) -> None:
    """Open the copy's file in the editor with the copy's own recipe."""
    path = Path(copy["path"])
    if not path.is_file():
        st.toast(t("The photo isn't there any more: {0}").format(path.name), icon="⚠️")
        return
    open_library_photo(path)
    st.session_state["_pending_restore"] = copy["recipe"]
    st.session_state["_library_open_copy"] = copy["id"]
    request_rerun()


def organise_section(catalog: PhotoCatalog, current_path: str) -> None:
    """Collections and virtual copies of the photo under the cursor."""
    name = Path(current_path).name
    st.markdown(t("**Collections**"))
    member_of = catalog.collections_of(current_path)
    if member_of:
        chip_cols = st.columns(min(4, len(member_of)))
        for index, collection in enumerate(member_of):
            if chip_cols[index % len(chip_cols)].button(f"✕ {collection}", key=f"lib-coll-rm-{index}",
                                                       help=t("Remove this photo from the collection")):
                catalog.remove_from_collection(collection, current_path)
                request_rerun()
    else:
        st.caption(t("This photo isn't in a collection."))
    known = list(catalog.collections())
    add_col, new_col, button_col = st.columns([2, 2, 1])
    existing_choice = add_col.selectbox(t("Add to collection"), ["", *[c for c in known if c not in member_of]],
                                        key="lib-coll-pick", format_func=lambda c: c or t("Choose…"))
    new_name = new_col.text_input(t("…or a new one"), key="lib-coll-new")
    if button_col.button(t("➕ Add"), key="lib-coll-add", use_container_width=True):
        target = new_name.strip() or existing_choice
        if target:
            catalog.add_to_collection(target, current_path)
            st.toast(t("Added {0} to '{1}'").format(name, target), icon="🗂️")
            request_rerun()
        else:
            st.toast(t("Choose a collection or type a new name"), icon="⚠️")

    st.markdown(t("**Virtual copies**"))
    st.caption(t("Another set of adjustments for the same file -- e.g. a colour and a black-and-white version -- "
                 "without duplicating it."))
    open_here = st.session_state.get("_library_open_path") == current_path
    for copy in catalog.virtual_copies(current_path):
        row = st.columns([3, 1, 1, 1])
        current_mark = " ✔" if open_here and st.session_state.get("_library_open_copy") == copy["id"] else ""
        row[0].write(f"🎞 {copy['name']}{current_mark}")
        if row[1].button(t("✏️ Edit"), key=f"lib-vc-open-{copy['id']}", use_container_width=True):
            open_virtual_copy(copy)
        if row[2].button(t("💾 Update"), key=f"lib-vc-update-{copy['id']}", use_container_width=True,
                         disabled=not open_here, help=t("Save the editor's current adjustments into this copy")):
            catalog.update_virtual_copy(copy["id"], inline_painted_masks(build_recipe()))
            st.toast(t("Updated '{0}'").format(copy["name"]), icon="💾")
        if row[3].button(t("🗑️"), key=f"lib-vc-delete-{copy['id']}", use_container_width=True,
                         help=t("Delete this virtual copy (the photo itself is kept)")):
            catalog.delete_virtual_copy(copy["id"])
            request_rerun()
    name_col, save_col = st.columns([3, 2])
    copy_name = name_col.text_input(t("Name for a new virtual copy"), key="lib-vc-name",
                                    placeholder=t("e.g. Black & white"))
    if save_col.button(t("🎞 Save edit as virtual copy"), key="lib-vc-save", use_container_width=True,
                       disabled=not open_here,
                       help=t("Open this photo in the editor first (✏️ Edit), then save its adjustments here")):
        copy_id = catalog.add_virtual_copy(current_path, copy_name or t("Virtual copy"),
                                           inline_painted_masks(build_recipe()))
        st.session_state["_library_open_copy"] = copy_id
        st.toast(t("Saved a virtual copy of {0}").format(name), icon="🎞")
        request_rerun()


def compare_section(catalog: PhotoCatalog, rec: list, cursor: int) -> None:
    """Two-up compare: the photo under the cursor beside another, larger, with their scores."""
    if len(rec) < 2 or not st.toggle(t("Two-up compare"), key="lib-compare"):
        return
    names = [entry["path"] for entry in rec]
    default = names[(cursor + 1) % len(names)]
    other = st.selectbox(t("Compare with"), [p for i, p in enumerate(names) if i != cursor], key="lib-compare-with",
                         index=max(0, [p for i, p in enumerate(names) if i != cursor].index(default)),
                         format_func=lambda p: Path(p).name)
    columns = st.columns(2)
    for side, (column, path) in enumerate(zip(columns, (names[cursor], other))):
        entry = catalog.get(path)
        with column:
            try:
                st.image(str(thumbnail(path, user_data_dir() / "thumbs-large", size=(900, 900))),
                         caption=grid_caption(entry), use_container_width=True)
            except (OSError, ValueError):
                st.caption(t("{0} (no preview)").format(Path(path).name))
            if entry["focus"] is not None:
                st.caption(score_caption(entry))
            rejected = "rejected" in entry["tags"]
            star_col, reject_col = st.columns(2)
            if star_col.button(t("⭐ +1"), key=f"lib-cmp-star-{side}", use_container_width=True,
                               help=t("One more star")):
                save_rating(catalog, path, min(5, int(entry["rating"]) + 1), entry["tags"],
                            write_xmp=bool(st.session_state.get("lib-xmp-write")))
                request_rerun()
            if reject_col.button(t("🚫 Un-reject") if rejected else t("🚫 Reject"), key=f"lib-cmp-reject-{side}",
                                 use_container_width=True):
                tags = [x for x in entry["tags"] if x != "rejected"] if rejected else [*entry["tags"], "rejected"]
                save_rating(catalog, path, int(entry["rating"]), tags,
                            write_xmp=bool(st.session_state.get("lib-xmp-write")))
                request_rerun()
    if st.button(t("⇄ Make the right one current"), key="lib-compare-swap"):
        st.session_state["lib-cursor"] = names.index(other)
        request_rerun()


def duplicates_section(catalog: PhotoCatalog, rec: list) -> None:
    """🔁 Find duplicates (F29): near-identical photos grouped by perceptual hash, the sharpest kept."""
    from app.duplicates import STRICTNESS, catalog_hashes, find_duplicate_groups

    with st.expander(t("🔁 Find duplicates"), expanded=False, key="sec-lib-duplicates"):
        st.caption(t("Groups copies and near-identical frames of the catalogued photos (re-saved, resized, burst shots). "
                     "The sharpest of each group is suggested to keep; the others can be rejected in one click."))
        strictness = st.select_slider(t("Match"), list(STRICTNESS), value="Near duplicates", key="lib-dup-strictness",
                                      format_func=t_option)
        if st.button(t("Find duplicates"), key="lib-dup-btn", disabled=len(rec) < 2):
            bar = st.progress(0.0)
            entries = catalog_hashes(catalog, [entry["path"] for entry in rec],
                                     progress=lambda index, total: bar.progress(index / total if total else 1.0))
            groups = find_duplicate_groups(entries, STRICTNESS[strictness])
            st.session_state["_lib_dup_groups"] = [{"paths": g.paths, "keep": g.keep, "distances": g.distances} for g in groups]
        groups = st.session_state.get("_lib_dup_groups")
        if groups is None:
            return
        if not groups:
            st.caption(t("No duplicates found."))
            return
        st.caption(t("{0} group(s), {1} photo(s) that could go.").format(len(groups), sum(len(g["paths"]) - 1 for g in groups)))
        write_xmp = bool(st.session_state.get("lib-xmp-write"))

        def reject(paths) -> None:
            for path in paths:
                entry = catalog.get(path)
                if "rejected" not in entry["tags"]:
                    save_rating(catalog, path, int(entry["rating"]), [*entry["tags"], "rejected"], write_xmp=write_xmp)

        if st.button(t("🚫 Reject all but the sharpest, in every group"), key="lib-dup-reject-all"):
            for group in groups:
                reject([p for p in group["paths"] if p != group["keep"]])
            st.toast(t("Rejected {0} photo(s)").format(sum(len(g["paths"]) - 1 for g in groups)), icon="🚫")
        for index, group in enumerate(groups[:20]):
            st.markdown(t("**Group {0}**").format(index + 1))
            columns = st.columns(min(4, len(group["paths"])))
            for position, path in enumerate(group["paths"][:4]):
                entry = catalog.get(path)
                keep = path == group["keep"]
                label = (t("✅ Keep") if keep else t("{0} bit(s) apart").format(group["distances"].get(path, 0)))
                with columns[position]:
                    if tab_visible():
                        try:
                            st.image(str(thumbnail(path, user_data_dir() / "thumbs")), use_container_width=True)
                        except (OSError, ValueError):
                            pass
                    st.caption(f"{Path(path).name} · {label}" + (f" · {score_caption(entry)}" if entry["focus"] is not None else "")
                               + (f" · {t('rejected')}" if "rejected" in entry["tags"] else ""))
            if st.button(t("🚫 Reject the others"), key=f"lib-dup-reject-{index}"):
                reject([p for p in group["paths"] if p != group["keep"]])
                request_rerun()


# ---- HDR merge and panorama (F18) ----
HDR_METHOD_LABELS = {"fusion": "Exposure fusion (natural, no exposure times needed)",
                     "debevec": "HDR merge + tone map (stronger; needs EXIF exposure times)"}


def merge_section() -> None:
    """Merge bracketed exposures into one HDR photo, or stitch overlapping frames into a panorama;
    the result opens in the editor as a new 16-bit source."""
    from app.merge import MAX_INPUTS, MergeError, encode_16bit_tiff, load_source, merge_hdr, preview_image, stitch_panorama

    kind = st.radio(t("Merge"), ["HDR", "Panorama"], horizontal=True, key="merge-kind", format_func=t_option)
    if kind == "HDR":
        st.caption(t("Two or more shots of the same scene at different exposures (a bracket, ideally from a "
                     "tripod). Small hand-held shifts are aligned."))
    else:
        st.caption(t("Two or more overlapping shots (about a third overlap), taken by turning the camera from "
                     "one spot."))
    files = st.file_uploader(t("Photos to merge"), type=uploader_types(), accept_multiple_files=True, key="merge-up")
    if kind == "HDR":
        method = st.selectbox(t("Method"), list(HDR_METHOD_LABELS), key="merge-hdr-method",
                              format_func=lambda key: t(HDR_METHOD_LABELS[key]))
        align = st.checkbox(t("Align the photos"), value=True, key="merge-align")
    else:
        crop = st.checkbox(t("Crop to the stitched area"), value=True, key="merge-crop",
                           help=t("Cuts away the ragged, empty edge a panorama has after warping"))
        scans = st.checkbox(t("Flat subject (documents, a mural)"), value=False, key="merge-scans",
                            help=t("For photos taken by moving the camera along a flat subject instead of turning it"))
    count = len(files or [])
    if st.button(t("🔀 Merge"), key="merge-run", disabled=not 2 <= count <= MAX_INPUTS, use_container_width=True):
        try:
            with st.spinner(t("Merging {0} photos…").format(count)):
                sources = [load_source(f.getvalue(), f.name) for f in files]
                pixels = (merge_hdr(sources, method, align=align) if kind == "HDR"
                          else stitch_panorama(sources, crop=crop, scans=scans))
                stem = Path(files[0].name).stem
                name = f"{stem}-{'hdr' if kind == 'HDR' else 'pano'}.tif"
                preview = BytesIO()
                preview_image(pixels).save(preview, format="JPEG", quality=90)
                st.session_state["_merge_result"] = (name, encode_16bit_tiff(pixels), preview.getvalue(),
                                                     f"{pixels.shape[1]} × {pixels.shape[0]}")
        except MergeError as error:
            st.error(t(str(error)))
        except (OSError, ValueError) as error:
            st.error(t("Couldn't read the photos: {0}").format(error))
    elif count == 1 or count > MAX_INPUTS:
        st.caption(t("Choose 2 to {0} photos.").format(MAX_INPUTS))
    result = st.session_state.get("_merge_result")
    if result:
        name, tiff, preview, size = result
        st.image(preview, caption=f"{name} · {size}", use_container_width=True)
        open_col, download_col = st.columns(2)
        if open_col.button(t("✏️ Open in the editor"), key="merge-open", use_container_width=True,
                           help=t("Opens the result as a new 16-bit photo; the current adjustments are kept (↺ resets them)")):
            st.session_state["restored_source_bytes"] = tiff
            st.session_state["restored_source_name"] = name
            st.session_state["_main_upload_generation"] = st.session_state.get("_main_upload_generation", 0) + 1
            st.session_state.pop("_library_open_path", None)
            request_rerun()
        download_col.download_button(t("⬇️ Download 16-bit TIFF"), tiff, file_name=name, mime="image/tiff",
                                     key="merge-download", use_container_width=True)


def library_section(ctx) -> None:
    more_mode = st.radio(t("Tool"), ["Library Catalog", "Batch Processor", "HDR & Panorama", "System Health"], horizontal=True, label_visibility="collapsed", key="more-mode", format_func=t_option)
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
        catalog_all = PhotoCatalog(lib_db)
        collection_names = catalog_all.collections()
        apply_pending_search(collection_names)
        q = st.text_input(t("Search"), key="lib-q-in")
        filter_cols = st.columns([2, 2, 2, 2])
        with filter_cols[0]:
            st.session_state.setdefault("lib-r-in", 0)
            r_min = st.slider(t("Min Stars"), 0, 5, key="lib-r-in")
        with filter_cols[1]:
            label_filter = st.selectbox(t("Colour label"), ["Any", "None", *COLOR_LABELS], key="lib-label-filter",
                                        format_func=lambda name: t(name) if name in ("Any", "None") else f"{LABEL_DOTS[name]} {t(name)}")
        with filter_cols[2]:
            sort = st.selectbox(t("Sort by"), list(SORT_LABELS), key="lib-sort", format_func=lambda key: t(SORT_LABELS[key]),
                                help=t("Focus is measured when a folder is scanned: the sharpness of the photo's sharpest part. "
                                       "Blurriest first finds missed focus and camera shake."))
        with filter_cols[3]:
            collection_filter = st.selectbox(
                t("Collection"), ["Any", *collection_names], key="lib-collection-filter",
                format_func=lambda name: t("Any") if name == "Any" else f"{name} ({collection_names.get(name, 0)})")
        label_arg = None if label_filter == "Any" else ("" if label_filter == "None" else label_filter)
        collection_arg = None if collection_filter == "Any" else collection_filter
        saved_searches_row(catalog_all, {"text": q, "minimum_rating": r_min, "label": label_arg, "sort": sort,
                                         "collection": collection_arg})
        search_results = catalog_all.search(q, r_min, label=label_arg, sort=sort, collection=collection_arg) if rec else []
        if rec and not search_results:
            st.caption(t("No photos match this search / star filter."))
        if search_results and tab_visible():
            # Thumbnail grid (cached JPEG thumbnails under the user data folder, RAW via its
            # embedded preview) with an Edit button that opens the photo in the editor.
            st.caption(t("✏️ Edit opens the photo in the editor with the current adjustments kept (↺ resets them)."))
            grid_cols = st.columns(4)
            copies_by_path: dict[str, list] = {}
            for copy in catalog_all.virtual_copies():
                copies_by_path.setdefault(copy["path"], []).append(copy)
            tiles = []
            for entry in search_results[:24]:
                tiles.append((entry, None))
                tiles.extend((entry, copy) for copy in copies_by_path.get(entry["path"], []))
            for grid_index, (entry, copy) in enumerate(tiles[:32]):
                entry_path = Path(entry["path"])
                with grid_cols[grid_index % 4]:
                    caption = grid_caption(entry) if copy is None else f"🎞 {copy['name']} · {entry_path.name}"
                    try:
                        st.image(str(thumbnail(entry_path, user_data_dir() / "thumbs")), caption=caption, use_container_width=True)
                    except (OSError, ValueError):
                        st.caption(t("{0} (no preview)").format(t(entry_path.name)))
                    if copy is None:
                        if st.button(t("✏️ Edit"), key=f"lib-edit-{image_digest(str(entry_path).encode('utf-8'))[:16]}", use_container_width=True):
                            open_library_photo(entry_path)
                            request_rerun()
                    elif st.button(t("✏️ Edit copy"), key=f"lib-vc-edit-{copy['id']}", use_container_width=True,
                                   help=t("Opens the photo with this virtual copy's adjustments")):
                        open_virtual_copy(copy)
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
            if existing["focus"] is not None:
                st.caption(score_caption(existing))
            write_xmp = st.checkbox(
                t("Also write ratings and tags to XMP sidecars"), key="lib-xmp-write",
                help=t("Saves each rating, reject flag and tag change in a .xmp file next to the photo, which "
                     "Lightroom, Capture One, Bridge and darktable read. Existing sidecars keep everything else "
                     "they hold. Ratings already in sidecars are always read when a folder is first scanned."),
            )

            def _save(rating: int, tags, label: str | None = None) -> None:
                problem = save_rating(catalog, current_path, rating, tags, write_xmp=write_xmp, label=label)
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

                # Colour labels (F21): keys 6-9 = Red/Yellow/Green/Blue, as in Lightroom. Keep this button
                # order in sync with app/ui/static/library_shortcuts.html.
                st.caption(t("Colour label (keys 6-9)"))
                label_cols = st.columns(len(COLOR_LABELS) + 1)
                for column, name in zip(label_cols, ("", *COLOR_LABELS)):
                    mark = "✔ " if existing["label"] == name else ""
                    text = mark + (t("None") if not name else LABEL_DOTS[name])
                    if column.button(text, key=f"lib-label-{name or 'none'}", use_container_width=True,
                                     help=t(name) if name else t("No colour label")):
                        _save(existing["rating"], existing["tags"], label=name)
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
            organise_section(catalog, current_path)
            compare_section(catalog, rec, cursor)
            duplicates_section(catalog, rec)
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
    elif more_mode == "HDR & Panorama":
        merge_section()
    else:
        from app.diagnostics import check_environment, logs_folder, open_folder
        st.dataframe([{"check": c.name, "status": c.status, "message": c.message} for c in check_environment()], use_container_width=True)
        st.caption(t("Logs: `{0}`").format(t(logs_folder())))
        if st.button(t("📂 Open logs folder"), key="diag-open-logs", help=t("Show the app's log files (useful to attach to a bug report)")):
            try:
                open_folder(logs_folder())
            except OSError as error:
                st.toast(t("Couldn't open the folder: {0}").format(t(error)), icon="⚠️")
