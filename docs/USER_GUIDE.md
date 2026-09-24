# Photo Studio — User Guide

Photo Studio is a professional, CPU-only photo editor: RAW/JPEG/PNG/WEBP/TIFF import, a full
tone/color/detail pipeline, local masking, object removal, layers, batch processing, and a photo
library — all running locally, no cloud, no GPU required.

This guide walks through every tab and control in the unified workspace. For what's changed
recently and known limitations, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## 1. Getting Started

**Desktop app (recommended):** run `release\PhotoStudio.exe` (or `PhotoStudio.exe` from an
extracted `PhotoStudio-<version>-Windows.7z`) — a native window, nothing to install.

**From source:** `streamlit run app/frontend_unified_app.py` opens it in your browser instead. Both
give you the exact same workspace.

**Opening a photo:** drag a file onto the uploader at the top, or click it to browse. Supported
formats: JPG, PNG, WEBP, TIFF, and camera RAW (`.cr2`, `.cr3`, `.nef`, `.arw`, `.dng` — RAW support
needs the optional `rawpy` package; the app tells you clearly if it's missing). Your original file
is never overwritten — every edit lives in the app until you explicitly export or save.

**Opening a saved project:** expand "📁 Open a saved project (.photoproject)" above the uploader to
reopen a `.photoproject` file you saved earlier (see [Projects](#projects) below).

---

## 2. The Workspace

The screen is split in two:

- **Left/Tool panel** — eleven tabs (Tone, Lens, Crop, Color, Detail, Repair, Layers, Masks,
  Portrait, Presets, Batch & Library) covering every adjustment.
- **Right/Canvas** — the live preview, with Undo/Redo/Reset, zoom, Split View, a histogram, a
  clipping warning, and export controls always available above and below it, regardless of which
  tool tab is open.

On a narrow window the canvas moves above the tool panel so you always see the photo first.

Every control updates the same live preview — there's no separate "apply" step. Changing a slider
sends the edit to the local processing backend and the canvas updates in place.

---

## 3. Tone

The starting point for every edit.

- **✨ Auto Enhance** — one click applies a balanced Exposure/Contrast/Saturation/Denoise/Sharpen
  pass as a starting point; touches nothing else (crop, masks, curve, temperature, …).
- **Preset Look** — a row of thumbnails (None, plus every built-in look) previews each one on your
  actual photo before you pick it.
- **Exposure**, **Contrast**, **Highlights**, **Shadows** — the four basic tonal sliders.

## 4. Lens

Corrects distortion from the lens/shooting angle. Not calibrated to your specific camera or
lens — the "Lens Profile" dropdown offers generic starting points by lens category (Wide-angle,
Standard zoom, Telephoto, Fisheye), which you fine-tune from there.

- **Lens Profile** + **Apply Profile Defaults** — loads that category's typical starting values;
  you can still adjust every slider afterward.
- **Distortion** — negative corrects barrel distortion (lines bowing outward); positive corrects
  pincushion (bowing inward).
- **Vignette Amount** / **Vignette Midpoint** — corrective darkening/brightening toward the
  corners, and how far in from the edge it starts.
- **Chromatic Aberration** — corrects red/cyan and blue/yellow color fringing near sharp edges.
- **Vertical / Horizontal Keystone** — corrects converging lines from the shooting angle (e.g. a
  building shot from below). Not part of a lens profile — this depends on how you held the camera,
  not what lens you used.

## 5. Crop

- **Aspect Ratio** — "Original" (no crop), "Free-form", or a named ratio (Square, 4:5, 5:4, 3:2,
  16:9, 9:16).
- **Free-form** shows a draggable rectangle directly on the photo — drag a corner to resize, drag
  the body to move it. A named ratio center-crops automatically. If a grid or guide is active (see
  [Guides & Grid](#canvas-controls) below), the corners snap to those lines as you drag them close —
  moving the whole rectangle doesn't snap, only resizing from a corner does.

Crop is applied last in the pipeline (after every other tab), but sits early in the tab order so
you frame the shot before fine-tuning color — the same workflow most photo editors use.

## 6. Color

- **Temperature** / **Tint** — the global white-balance sliders.
- **Saturation** — a flat, whole-image saturation boost.
- **Vibrance** — a smarter alternative: protects skin tones and already-saturated colors, pushing
  washed-out/pastel colors harder instead of boosting everything equally.
- **🎯 Pick White Balance** — click a spot in the photo that should be neutral gray or white, and
  Temperature/Tint are set to whatever would neutralize it. A starting point to fine-tune from, not
  a guaranteed-exact fix.
- **Color Mixer (HSL)** — independent Hue/Saturation/Luminance sliders for each of 8 named color
  bands (Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta) — a band's icon turns green once you've
  touched it, so you can see at a glance which ones are active.
- **Color Grading (Split Toning)** — tint Shadows/Midtones/Highlights toward independent hues, plus
  a Balance slider shifting where shadows end and highlights begin.
- **RGB Tone Curve** — drag any of the 5 fixed points (Blacks/Shadows/Midtones/Highlights/Whites)
  vertically, or click an empty spot on the curve to add a fully free-form point you can drag
  anywhere (double-click to remove it). The sliders below give exact numeric values for the 5 fixed
  points. Must be enabled via the checkbox to take effect.
- **🎯 Soft Proof: Gamut Warning** — for a photo with an embedded color profile, highlights in
  magenta any pixel that gets clipped when converted into this app's sRGB working space. Not a
  simulation of a specific printer profile.

## 7. Detail

- **Texture** / **Clarity** — local contrast and micro-detail sliders.
- **Dehaze** — reduces atmospheric haze (a classical algorithm, not AI). Pushed high, it can crush
  shadows or halo around strong edges.
- **Denoise** — smooths sensor noise while preserving edges (Non-Local Means, a real upgrade over a
  flat blur).
- **Color Noise** — an independent chroma-only denoise boost on top of Denoise, for cleaning up
  color speckle without softening detail as much. Has no effect while Denoise is 0.
- **Sharpen (Unsharp Mask)** — a checkbox to enable output sharpening.
- **✨ Effects** (expander):
  - **Post-Crop Vignette** — a creative vignette applied to the final, already-cropped frame
    (separate from the Lens tab's corrective one, which runs before Crop — using both at once
    stacks their effect, which is expected).
  - **Film Grain** — adds a monochromatic noise texture; a fresh random pattern every edit/export
    (so repeated exports don't look identically grainy).

## 8. Repair (object removal)

Select a blemish or unwanted object with the same selection tools as Masks (see
[Selection Types](#selection-types-masks--repair) below), and it's painted over by blending in
nearby texture (inpainting).

- **Repair Softness** — how far the repair blends into the surrounding area; higher softens a wider
  edge. Disabled until at least one selection is enabled.
- **👁️ Preview removal selection** — shows the selected region in yellow (Masks' own preview uses
  red, so the two are never confused).

## 9. Layers

Build a stack of any number of layers — images, text, vector shapes, and freeform paths — each
with its own opacity, blend mode, and optional effects, composited on the processing backend when
the image renders.

**Adding a layer** — "➕ Add a layer" offers four types:

| Type | What you get |
|---|---|
| **Image** | Upload a PNG/JPG/WEBP to composite on top of your photo. |
| **Text** | Type content, then set font size, color, alignment (left/center/right), and position (as a % of the photo, placing the text block's own center). |
| **Shape** | A vector Rectangle, Ellipse, Line, Polygon, or Star — position, size, and rotation (except Ellipse, which doesn't rotate), plus its own fill color/opacity and outline color/width. Polygon and Star also get a Sides/Points count; Star gets an Inner Radius controlling how sharp its points are. |
| **Pen** | Click points directly on the photo below to trace a freeform path, then **➕ Add Path Layer**. See [The Pen Tool](#the-pen-tool) below. |

**The layer stack** — each layer's own expander (click its name to open it) splits into three
sub-tabs:

- **🎨 Content** — the kind-specific fields (Text's content/font/color/position, Shape's
  type/size/rotation/fill/outline, the Pen editor, or an Image layer's thumbnail), plus a live
  preview for Text/Shape/Pen layers.
- **✨ Styles** — **Layer Styles**: Drop Shadow, Outer Glow, Stroke, and Color Overlay, any
  combination, on *any* layer type. Each is independent of the others (widening the Stroke doesn't
  change where the Glow falls, for example) and computed from that layer's own shape, so they only
  show up where the layer actually has transparent edges — a photo used as a full-canvas background
  layer has no "outside its own shape" for a shadow or glow to occupy, the same way Photoshop's own
  layer effects work.
- **⚙️ Settings** — **Name**, **Visible**, **Opacity**, and **Blend Mode** (26 blend modes in
  total: Normal; the Darken group: Darken/Multiply/Color Burn/Linear Burn/Darker Color; the Lighten
  group: Lighten/Screen/Color Dodge/Linear Dodge/Lighter Color; the Contrast group: Overlay/Soft
  Light/Hard Light/Vivid Light/Linear Light/Pin Light/Hard Mix; the Comparative group:
  Difference/Exclusion/Subtract/Divide; and Hue/Saturation/Color/Luminosity — the same set
  Photoshop's own Layers panel offers), plus **⬆️ Up** / **⬇️ Down** to reorder the stack (the last
  layer listed sits on top), **📄 Duplicate**, and **🗑️ Remove**.

### The Pen Tool

A true Bezier-curve path, not just straight-line shapes:

- **Click** an empty spot on the photo to add a new point.
- **Drag a point** to move it.
- **Drag its amber handle** to curve the path through that point — the handle sets one shared
  tangent direction used on both sides of the point (a "smooth point," in vector-editor terms).
  Leave a point's handle at zero length for a plain straight corner instead.
- **Double-click a point** to remove it.
- **Closed Path** — connects the last point back to the first, and unlocks **Fill** (color +
  opacity) in addition to **Stroke**. An open path can only be stroked — there's no well-defined
  "inside" to fill without a closed boundary.
- **🗑️ Clear Points** — starts over.

One handle per point means a curve's direction can't change abruptly on one side versus the
other the way Photoshop's Alt-drag "corner point" allows — see
[Known Limitations](#known-limitations).

**A note on Text/Shape/Path layers and Save Project:** these three layer types stay fully editable
while you're working, but **Save Project flattens them into a plain image** — reopening a
`.photoproject` later shows them as an ordinary Image layer, not editable text/shape/path anymore.
An Image layer (an uploaded photo) doesn't have this limitation, since it was already a plain image
to begin with.

## 10. Masks (local adjustments)

Build any number of named selections and combine them (union) into one mask; the Target sliders
below only affect pixels inside that combined selection.

- **➕ Add a selection** — see [Selection Types](#selection-types-masks--repair) below for the six
  kinds available.
- Each added layer can be enabled/disabled, inverted, given an edge-softening radius, and (for most
  types) an opacity — all editable after adding it, and removable.
- **👁️ Preview combined mask overlay** — shows the union of every enabled layer in red.
- **Target Exposure**, **Target Contrast**, **Target Saturation** — the actual local adjustment,
  applied only where the mask is active. Disabled until at least one mask layer is enabled.

### Selection Types (Masks & Repair)

Both the Masks tab and the Repair tab share the same six selection tools:

| Type | How it works |
|---|---|
| **Luminance** | Selects pixels within a brightness range (Min/Max Luma, 0–255). |
| **Color** | Pick a target color and a tolerance; selects similar-colored pixels anywhere in the frame. |
| **Horizontal / Vertical gradient** | A smooth linear fade across the frame. |
| **Radial** | A smooth fade outward from a center point. |
| **Painted** | Paint directly on the photo — freehand brush (with an eraser), a drag-out radial, or a drag-out gradient. Reopen it later with "🎨 Edit Painting" to continue painting on the same selection instead of starting over. |
| **Magic Wand** | Click a point on the photo; selects similar colors around it. Contiguous mode (default) only grows outward from that connected region; unchecked, it selects every matching-color pixel anywhere in the frame. Tolerance controls how similar a color has to be. |

Every selection (except Painted/Magic Wand's own click-based commit) also gets a name, an edge-soften
radius, and an invert checkbox.

## 11. Portrait

- **Skin Smoothing** — 0 leaves skin untouched (the default); increase to smooth detected skin
  tones.
- The tab also reports how many faces it detected in the photo (informational only — not required
  for smoothing to work, which is tone-based rather than face-located).

## 12. Presets

- **Load a Preset** — pick a saved custom preset from the dropdown; a live thumbnail preview shows
  what it would look like on your current photo before you apply it. **📂 Apply** loads it,
  **🗑️ Delete** removes it.
- **Save Current Look as Preset** — type a name and save. This captures your *entire* current
  recipe (every Tone/Lens/Color/Detail/Mask setting), not just a handful of basic sliders — the one
  thing deliberately excluded is the layer overlay image itself, since a preset is a reusable style,
  not tied to one specific photo.
- **Export / Import Presets** — download every saved preset as a shareable `.json` file, or upload
  one to import (names that already exist locally are skipped, not overwritten).
- **Project File** — see [Projects](#projects) below.

Undo/Redo/Reset live in the canvas header bar above, not in this tab — see
[Canvas Controls](#canvas-controls) below.

## 13. Batch & Library

Three modes, selected via the radio buttons at the top of the tab.

### Library Catalog

- **Folder** + **Scan Folder** — indexes every JPG/PNG/WEBP/TIFF in a folder (recursively), showing
  a live progress bar as it works. Scanning doesn't touch the files themselves — it just builds a
  local searchable catalog with thumbnails.
- **Search** + **Min Stars** — filters the indexed photos by filename/tag text and a minimum star
  rating. If nothing matches, the app says so instead of showing a silent empty table.
- **Rate & Flag** — after a scan, step through your photos one at a time:
  - **⬅ Prev** / **Next ➡** to move between them.
  - Five star buttons to rate (or **Clear rating**).
  - **🚫 Reject** to flag a photo (stored as a tag, doesn't affect its rating).
  - A comma-separated **Tags** field, with **Save tags**.
  - **Keyboard shortcuts** (work whenever this panel is visible and you're not typing in a text
    field): **1–5** to rate, **0** to clear the rating, **X** to toggle reject, **←/→** to move
    Prev/Next.

### Batch Processor

- **Batch Files** — upload multiple photos at once.
- **Profile** — an export profile (size/quality preset) applied to every file.
- **Preset** — apply either a built-in named look, or any saved custom preset (suffixed
  "(custom)") to every photo in the batch — the same idea as Lightroom's "Sync Settings" across a
  shoot. A custom preset's mask layers, removal selections, and layer overlay are intentionally
  *not* carried into the batch (they're tied to one specific photo's pixels) — only its global
  tone/color/detail/lens settings transfer.
- **Run Batch** — shows a live progress bar and per-file status as it processes, ending in a
  results table and a **Download ZIP** button.

### System Health

A read-only diagnostics table (`app/diagnostics.py`) — useful for confirming optional dependencies
like `rawpy` (RAW import) are actually installed and working.

---

## Canvas Controls

Always visible above/below the preview, on every tab:

- **↶ Undo** / **↷ Redo** — step backward/forward through your edit history. Also available
  anywhere via **Ctrl+Z** (undo) and **Ctrl+Shift+Z** or **Ctrl+Y** (redo) — these work even right
  after dragging a slider, not just when a button has focus. They're suppressed only while you're
  actually typing in a text field, so a normal Ctrl+Z there still does ordinary text-undo.
- **↺ Reset** — restores every adjustment to its default. Deliberately leaves an attached layer or
  RAW source alone — only the slider/checkbox values reset.
- **Zoom View** — Fit, 50%, 100%, 150%, 200%, or 400%. Above 100%, Pan Horizontal/Vertical sliders
  appear to move around the zoomed view.
- **⇄ Split View** — a draggable before/after comparison slider over the same zoomed/panned view.
- **👁️ Original** — shows the untouched source image instead of your edit (disabled while Split
  View is active, since that already shows both).
- **🧭 Guides & Grid** — a composition aid, purely visual: it's never sent through the processing
  backend and never appears in an exported file.
  - **Grid overlay** — None, Rule of Thirds, Grid 4x4, or Golden Ratio.
  - **Custom guides** — add any number of horizontal or vertical guide lines at a chosen position
    (%), each removable individually.
  - The free-form Crop rectangle's corner handles (see [Crop](#5-crop)) snap to whichever grid
    lines/guides are active.
- **Histogram** — a live per-channel (Red/Green/Blue) chart of the current edited image.
- **⚠️ Show clipping** — highlights blocked shadows (blue) and blown highlights (red) directly on
  the canvas.
- **Export Format / Quality / Save** — choose JPEG, PNG, WEBP, or TIFF (Quality only applies to
  JPEG/WEBP), then download the edited result.
- **📦 Export Multiple Sizes** — check any combination of the named export profiles (Web, Social,
  Print, Archive) and click **Prepare ZIP**, then **⬇️ Download ZIP** — gets you the current photo
  at several sizes/formats in one click, instead of downloading each one separately. This is
  different from the Batch Processor (below): this is *one photo, several output variants*; Batch
  is *one recipe, many photos*.

---

## Projects

Save everything — the source photo, every adjustment, your full layer stack, and your mask/removal
selections — into one `.photoproject` file you can reopen later:

- **💾 Save Project** (Presets tab) packages the current state and gives you a download button.
- **📁 Open a saved project** (top of the app, above the uploader) restores a `.photoproject` back
  into the workspace exactly as you left it. A corrupted or hand-edited file shows a clear error
  instead of crashing the app.
- **Text, Shape, and Pen path layers are flattened into plain images when saved** — reopening the
  project shows them as ordinary Image layers, no longer editable as text/shapes/paths. Everything
  else (every layer's position in the stack, opacity, blend mode, and any Layer Styles) round-trips
  correctly.

---

## Keyboard Shortcuts Reference

| Shortcut | Action | Where |
|---|---|---|
| Ctrl+Z | Undo | Anywhere (except while typing in a text field) |
| Ctrl+Shift+Z or Ctrl+Y | Redo | Anywhere (except while typing in a text field) |
| 1–5 | Rate current photo | Library Catalog, Rate & Flag panel |
| 0 | Clear rating | Library Catalog, Rate & Flag panel |
| X | Toggle Reject | Library Catalog, Rate & Flag panel |
| ← / → | Previous / Next photo | Library Catalog, Rate & Flag panel |

---

## Known Limitations

This app is CPU-first and offline by design — no cloud processing, no GPU dependency. That means a
few things are deliberately simpler than a cloud/AI-powered tool: lens and denoise are classical
algorithms (not neural networks), color proofing checks against this app's own working space (not
a specific printer profile), and there's no automatic subject/sky detection.

A few more specific to Layers:

- **Text layers use whatever font is installed on the machine running the app** — there's no font
  picker or custom font upload, so the same project can render with a different typeface on a
  different computer.
- **The Pen tool's curve handles are symmetric** — one handle per point, shared by both directions
  the curve travels through it. There's no independent, non-mirrored handle pair for a curve that
  changes direction sharply on one side (a plain straight corner is still fully supported).
- **Ellipse shapes don't rotate.**

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md)'s "Known Issues" section for the full, current list.
