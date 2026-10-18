# Photo Studio — User Guide

Photo Studio is a professional, CPU-first photo editor: RAW/JPEG/PNG/WEBP/TIFF import, a full
tone/color/detail pipeline, local masking, object removal, layers, batch processing, and a photo
library — all running locally, no cloud. Core editing and file encoding need no GPU; an optional
CUDA path can accelerate final export resizing and sharpening.

This guide walks through every tab and control in the unified workspace. For known limitations, see
[`PROJECT_STATUS.md`](PROJECT_STATUS.md); for what changed recently, [`CHANGELOG.md`](CHANGELOG.md).

---

## 1. Getting Started

**Desktop app (recommended):** run `PhotoStudio.exe` from an extracted
`PhotoStudio-<version>-Windows.7z` (or `release\PhotoStudio.exe` after a local build) — a native
window, nothing else to install. Or run the installer, `PhotoStudio-Setup-<version>.exe` (see
[Installing](#installing) below).

**From source:** `streamlit run app/frontend_unified_app.py` opens it in your browser instead. Both
give you the exact same workspace.

**Opening a photo:** drag a file onto the uploader at the top, or click it to browse. Supported
formats: JPG, PNG, WEBP, TIFF (including 16-bit PNG/TIFF), and camera RAW (`.cr2`, `.cr3`, `.nef`,
`.arw`, `.dng`, `.raf`, `.rw2`, `.orf`, `.pef`, `.srw`) — RAW support is included in the desktop
app; from source it needs the optional `rawpy` package (`requirements-raw.txt`), and the app tells
you clearly if it's missing). Your original file
is never overwritten — every edit lives in the app until you explicitly export or save.

**Opening a saved project:** expand "📁 Open a saved project (.photoproject)" above the uploader to
reopen a `.photoproject` file you saved earlier (see [Projects](#projects) below).

---

## 2. The Workspace

The screen is split in two:

- **Left/Tool panel** — nine tabs, grouped by how you work:

  | Tab | What's in it |
  |---|---|
  | **☼ Basic** | ✨ Auto Enhance, 🎞️ Preset Look, Tone (exposure, contrast, highlights, shadows, whites, blacks), Color (white balance, saturation, vibrance, 🎯 Pick White Balance) and Presence (texture, clarity, dehaze) |
  | **🎨 Curves & Color** | RGB Tone Curve, Color Mixer (HSL), Black & White, Color Grading, Soft Proof |
  | **🔬 Detail** | Denoise, Color Noise, Sharpen |
  | **⛶ Geometry** | Rotate & Straighten, Lens Corrections, Crop / Aspect Ratio |
  | **◉ Local** | three sub-tabs: ◉ Masks, 🩹 Repair, ◌ Portrait |
  | **◈ Layers** | image, text, shape and pen-path layers |
  | **✨ Effects** | post-crop vignette, film grain |
  | **⬇ Export** | export settings and presets, 🏷️ Watermark, GPU export transforms, AI Upscale, 📦 Export Multiple Sizes, 💾 Save Project |
  | **✦ Presets & Library** | two sub-tabs: ✦ Presets, ☰ Library & Batch (Library Catalog, Batch Processor, System Health) |

  Every section shows **● modified** next to its title (or a ● in its expander's title) as soon as
  one of its controls differs from the default, so you can see at a glance where a photo has been
  edited; most sections also have their own **↺ Reset** button.
- **Right/Canvas** — the live preview, with Undo/Redo/Reset, zoom, Split View, a histogram, a
  clipping warning, and the Save button always available above and below it, regardless of which
  tool tab is open.

On a narrow window the canvas moves above the tool panel so you always see the photo first.

Every control updates the same live preview — there's no separate "apply" step. Changing a slider
sends the edit to the local processing backend and the canvas updates in place.

---

## 3. Tone (☼ Basic)

The starting point for every edit.

- **✨ Auto Enhance** — measures *this* photo and sets Exposure, the black/white points (tone
  curve), a gray-world white balance, Vibrance for a flat image, and Denoise only for a noisy one; a
  message lists what it changed. Crop, masks, HSL and everything else stay as you left them.
- **Preset Look** — a row of thumbnails (None, plus every built-in look) previews each one on your
  actual photo before you pick it.
- **Exposure (EV)** — in stops of light, −5 to +5: +1 doubles the light, −1 halves it. Bright
  highlights roll off smoothly instead of clipping.
- **Contrast**, **Highlights**, **Shadows** — −100 to +100. Highlights/Shadows work on regions (an
  edge-aware base layer), so lifting a shadowed face keeps its texture and doesn't flatten the bright
  background next to it.
- **Whites**, **Blacks** — −100 to +100: set the white and black points without opening the curve.
  Negative Blacks deepens the darkest tones into black, positive Whites pushes the brightest into
  white; the midtones barely move.
- **↺ Reset Tone** — puts just this section back to its defaults. Lens, Rotate & Straighten, Color,
  Detail, Effects and the mask targets have the same button.

Older presets and projects (saved with the earlier 0.5–1.5 multiplier sliders) are converted
automatically when you open them.

## 4. Lens (⛶ Geometry)

Corrects distortion from the lens/shooting angle.

- **📷 From the photo's lens** — reads the camera, lens, focal length and aperture from the photo's
  EXIF, looks them up in the lensfun lens database (installed with the app; offline) and sets
  Distortion, Vignette and Chromatic Aberration to the closest match of that lens's calibration at
  this focal length and aperture. The caption says which lens it matched, or why it couldn't (no
  EXIF focal length -- RAW files don't carry their EXIF through here --, or a lens lensfun doesn't
  know). The sliders are simpler than lensfun's own models, so this is a close approximation (most
  of the distortion, not complex "mustache" shapes); fine-tune from there.
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

## 5. Crop, Rotate & Straighten (⛶ Geometry)

**Rotate & Straighten** (top of the tab):

- **⟲ Rotate left** / **⟳ Rotate right** — 90° turns (the rotation is also shown and selectable).
- **⇆ Flip horizontal** / **⇅ Flip vertical**.
- **Straighten (°)** — −45° to +45° to level a tilted horizon; **Auto** detects the photo's
  dominant horizontal/vertical lines and sets the angle for you.
- **Constrain crop** (on by default) — crops away the black or smeared corners that Distortion,
  Keystone and Straighten would otherwise leave.

**Crop / Aspect Ratio**:

- **Aspect Ratio** — "Original" (no crop), "Free-form", "Custom" (type any W:H), or a named ratio
  (Square, 4:5, 5:4, 3:2, 16:9, 9:16). **Swap orientation** turns a ratio portrait ↔ landscape.
- **Free-form** shows a draggable rectangle directly on the photo — drag a corner to resize, drag
  the body to move it — plus **Left / Top / Width / Height** fields to type the box in pixels. A
  named ratio center-crops automatically. If a grid or guide is active (see
  [Guides & Grid](#canvas-controls) below), the corners snap to those lines as you drag them close —
  moving the whole rectangle doesn't snap, only resizing from a corner does.

Crop is applied last in the pipeline (after every other tab), but sits early in the tab order so
you frame the shot before fine-tuning color — the same workflow most photo editors use.

## 6. Color (☼ Basic, and 🎨 Curves & Color)

Temperature, Tint, Saturation, Vibrance and the white-balance picker sit in **☼ Basic → Color**; the
tone curve, HSL mixer, color grading and soft proof are on **🎨 Curves & Color**.

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
- **⚫ Black & White** — "Convert to Black & White", then one slider per colour band brightens or
  darkens the gray that colour turns into (darken Blue for a dramatic sky, brighten Orange for
  lighter skin). Color Grading still applies afterwards, so it can tone the result (sepia, split
  toning). CLI: `--black-white Blue=-60,Orange=20`.
- **Color Grading (Split Toning)** — tint Shadows/Midtones/Highlights toward independent hues, plus
  a Balance slider shifting where shadows end and highlights begin.
- **RGB Tone Curve** — drag any of the 5 fixed points (Blacks/Shadows/Midtones/Highlights/Whites)
  vertically, or click an empty spot on the curve to add a fully free-form point you can drag
  anywhere (double-click to remove it). The sliders below give exact numeric values for the 5 fixed
  points. Must be enabled via the checkbox to take effect. The **RGB / Red / Green / Blue** tabs
  switch the editor between the master curve and one curve per colour channel (five points each,
  no extra points): lift Blue in the shadows for a cool cast, pull Red down in the highlights, and
  so on. They apply after the master curve, while the curve is enabled. CLI: `--curve-red`,
  `--curve-green`, `--curve-blue` (same `x:y,...` format as `--curve`).
- **🎯 Soft Proof: Gamut Warning** — for a photo with an embedded color profile, highlights in
  magenta any pixel that gets clipped when converted into this app's sRGB working space. Not a
  simulation of a specific printer profile.

## 7. Detail (🔬 Detail), Presence (☼ Basic) and Effects (✨ Effects)

- **Texture** / **Clarity** — local contrast and micro-detail sliders (☼ Basic → Presence).
- **Dehaze** (☼ Basic → Presence) — reduces atmospheric haze (a classical algorithm, not AI). Pushed high, it can crush
  shadows or halo around strong edges.
- **Denoise** — smooths sensor noise while preserving edges (Non-Local Means, a real upgrade over a
  flat blur).
- **Color Noise** — an independent chroma-only denoise boost on top of Denoise, for cleaning up
  color speckle without softening detail as much. Has no effect while Denoise is 0.
- **Sharpen (Unsharp Mask)** — a checkbox to enable output sharpening.
- **✨ Effects** (its own tab):
  - **Post-Crop Vignette** — a creative vignette applied to the final, already-cropped frame
    (separate from the Lens corrections' corrective one, which runs before Crop — using both at once
    stacks their effect, which is expected).
  - **Film Grain** — adds a monochromatic noise texture; a fresh random pattern every edit/export
    (so repeated exports don't look identically grainy).

## 8. Repair — object removal (◉ Local → 🩹 Repair)

Select a blemish or unwanted object with the same selection tools as Masks (see
[Selection Types](#selection-types-masks--repair) below), and it's repaired by the chosen method:

- **Repair Method** — **Fill** paints the area over from its edges (inpainting): good for wires and
  small objects on smooth areas such as sky. **Heal** looks for the best-matching patch nearby,
  copies it over each selected spot and blends its colour and brightness into the surroundings, so
  real texture (skin, sand, grass) comes back instead of a smooth smear. A spot at the photo's edge,
  or one too large for a clean nearby patch, is filled instead.
- **Repair Softness** — how far the repair blends into the surrounding area; higher softens a wider
  edge. Disabled until at least one selection is enabled.
- **👁️ Preview removal selection** — shows the selected region in yellow (Masks' own preview uses
  red, so the two are never confused).

## 9. Layers (◈ Layers)

Build a stack of any number of layers — images, text, vector shapes, and freeform paths — each
with its own opacity, blend mode, and optional effects, composited on the processing backend when
the image renders.

**Adding a layer** — "➕ Add a layer" offers four types:

| Type | What you get |
|---|---|
| **Image** | Upload a PNG/JPG/WEBP to composite on top of your photo. It keeps its proportions (fitted inside the photo at 100 %) and can be moved, scaled and rotated; "Stretch to fill the photo" makes it cover the whole frame instead. |
| **Text** | Type content, then pick a **Font** (any font installed on this computer, or one you added under **🔤 Fonts**; "Automatic" is the default) and set font size (a % of the photo's short edge, so it looks the same at any resolution), color, alignment (left/center/right), and position (as a % of the photo, placing the text block's own center). |
| **Shape** | A vector Rectangle, Ellipse, Line, Polygon, or Star — position, size, and rotation, plus its own fill color/opacity and outline color/width. Polygon and Star also get a Sides/Points count; Star gets an Inner Radius controlling how sharp its points are. |
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
  layer listed sits on top), **📄 Duplicate**, and **🗑️ Remove**. **Layer mask** shows the layer
  only where one of the ◉ Masks selections is (its own softness, invert and opacity apply), and
  **Invert layer mask** flips it — e.g. a texture layer that covers only the sky. Build the
  selection on the Masks tab; leave its target sliders at 0 if it should only mask the layer.

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

Text, Shape and Pen path layers stay editable after **Save Project** and reopening the file (see
[Projects](#projects)).

## 10. Masks — local adjustments (◉ Local → ◉ Masks)

Build any number of named selections and combine them into one mask; the Target sliders below
only affect pixels inside that combined selection. A selection can also have its **own
adjustments** instead (see below) — so you can brighten a face and darken the sky in one edit.

- **➕ Add a selection** — see [Selection Types](#selection-types-masks--repair) below for the six
  kinds available.
- Each added layer can be enabled/disabled, inverted, given an edge-softening radius, and (for most
  types) an opacity — all editable after adding it, and removable.
- **Combine with the selections above** — Add (union), Subtract (cut its area out) or Intersect
  (keep only the overlap).
- **Own adjustments** — this selection gets its own Exposure (EV), Contrast, Saturation, Clarity,
  Temperature and Tint, applied only inside it, instead of joining the combined mask.
- **👁️ Preview combined mask overlay** — shows the combined selection as a tint, diagonal hatching,
  or an outline (so it's readable without telling colours apart), in a colour you choose.
- **Target Exposure (EV)**, **Target Contrast**, **Target Saturation** — the shared local
  adjustment, applied only where the combined mask is active. Disabled until at least one mask layer
  without its own adjustments is enabled.

### Selection Types (Masks & Repair)

Masks and Repair (both under ◉ Local) share the same selection tools:

| Type | How it works |
|---|---|
| **Luminance** | Selects pixels within a brightness range (Min/Max Luma, 0–255). |
| **Color** | Pick a target color and a tolerance; selects similar-colored pixels anywhere in the frame. |
| **Horizontal / Vertical gradient** | A smooth linear fade across the frame. |
| **Radial** | A smooth fade outward from a center point — set its Center X/Y, Radius and Feather (also editable afterwards). |
| **Painted** | Paint directly on the photo — freehand brush (with an eraser), a drag-out radial, or a drag-out gradient. Reopen it later with "🎨 Edit Painting" to continue painting on the same selection instead of starting over. |
| **Magic Wand** | Click a point on the photo; selects similar colors around it. Contiguous mode (default) only grows outward from that connected region; unchecked, it selects every matching-color pixel anywhere in the frame. Tolerance controls how similar a color has to be, and **Sample size** (Point, 3×3, 5×5, 11×11) averages the color around the click — steadier than a single pixel on noisy or textured areas. |

Every selection (except Painted/Magic Wand's own click-based commit) also gets a name, an edge-soften
radius, and an invert checkbox.

## 11. Portrait (◉ Local → ◌ Portrait)

- **Skin Smoothing** — 0 leaves skin untouched (the default); increase to smooth detected skin
  tones.
- **Only around detected faces** (on by default) — smooths skin tones only around the faces the
  app finds, so a wall, wooden floor or sand in skin-like colours elsewhere in the photo stays
  sharp. When no face is found (or face detection isn't available), every skin-toned area is
  smoothed, as with the option off.
- The tab also reports how many faces it detected in the photo.

## 12. Presets (✦ Presets & Library → ✦ Presets)

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

## 13. Library & Batch (✦ Presets & Library → ☰ Library & Batch)

Three modes, selected via the radio buttons at the top of the tab.

### Library Catalog

- **Folder** + **Scan Folder** — indexes every JPG/PNG/WEBP/TIFF and camera RAW file in a folder
  (recursively), showing a live progress bar as it works. Scanning doesn't touch the files
  themselves, and rescanning keeps the ratings and tags you've already given. The catalog is shown
  the next time you open the app, without scanning again. A photo scanned for the first time takes
  its rating, reject flag and keywords from an **XMP sidecar** next to it (`IMG_0001.xmp` or
  `photo.jpg.xmp`, as Lightroom, Capture One, Bridge and darktable write them), if it has one, and
  a rescan picks up a sidecar another program changed since.
- **Thumbnail grid** — the matching photos as thumbnails (cached, RAW files via their embedded
  preview), each with **✏️ Edit** to open it in the editor. Your current adjustments are kept, so a
  look carries over to the next photo of a shoot; **↺** resets them.
- **Search** + **Min Stars** — filters the indexed photos by filename/tag text and a minimum star
  rating. If nothing matches, the app says so instead of showing a silent empty table.
- **Rate & Flag** — after a scan, step through your photos one at a time:
  - **⬅ Prev** / **Next ➡** to move between them.
  - Five star buttons to rate (or **Clear rating**).
  - **🚫 Reject** to flag a photo (stored as a tag, doesn't affect its rating).
  - A comma-separated **Tags** field, with **Save tags**.
  - **Also write ratings and tags to XMP sidecars** — off by default; when on, every rating,
    reject and tag change is also saved in the photo's `.xmp` sidecar (rejected = rating -1), so
    other photo applications see your cull. An existing sidecar keeps everything else it holds
    (develop settings included); a damaged one is left alone and a warning says so.
  - **✏️ Edit this photo** — opens the current photo in the editor.
  - **Keyboard shortcuts** (work whenever this panel is visible and you're not typing in a text
    field): **1–5** to rate, **0** to clear the rating, **X** to toggle reject, **←/→** to move
    Prev/Next.

### Batch Processor

- **Batch Files** — upload multiple photos at once — any format the editor opens, RAW included.
  Files are processed straight from the upload (two files with the same name are both kept).
- **Profile** — an export profile (size/quality preset) applied to every file.
- **Preset** — apply either a built-in named look, or any saved custom preset (suffixed
  "(custom)") to every photo in the batch — the same idea as Lightroom's "Sync Settings" across a
  shoot. A custom preset's mask layers, removal selections, and layer overlay are intentionally
  *not* carried into the batch (they're tied to one specific photo's pixels) — only its global
  tone/color/detail/lens settings transfer.
- **Run Batch** — shows a live progress bar and per-file status as it processes, writing each
  result into a ZIP (with a `report.csv`) as it goes, ending in a results table and a **Download ZIP**
  button. **⏹ Stop** ends a running batch; the files finished so far stay in the ZIP.

### System Health

A read-only diagnostics table (`app/diagnostics.py`) — useful for confirming optional dependencies
like `rawpy` (RAW import) are actually installed and working. **📂 Open logs folder** opens the
folder holding the app's log files (its path is shown above the button) — attach them to a bug
report.

---

## Canvas Controls

Always visible above/below the preview, on every tab:

- **↶ Undo** / **↷ Redo** — step backward/forward through your edit history (the last 200 steps).
  Also available via **Ctrl+Z** (undo) and **Ctrl+Shift+Z** (redo).
- **📌 History & Snapshots** — the **History** list shows the last 30 steps, newest first, each named
  after what it changed ("Exposure (EV) 0.3", "Masks", "Contrast, Clarity"); click one to jump
  there, exactly as if you pressed Undo/Redo until it (the later steps stay until you make a new
  edit). **Named Snapshots** bookmark the current look under a name.
- **↺ Reset** — restores every adjustment to its default. Deliberately leaves an attached layer or
  RAW source alone — only the slider/checkbox values reset.
- **Zoom View** — **Fit** (the whole photo), **100% (1:1)** — the photo's real pixels, the view to
  judge sharpening, noise, grain and mask edges in (rendered at full resolution even for a large
  photo) — and 200%/400%, which enlarge those pixels. While zoomed, **drag the photo** to pan (or
  click it and use the arrow keys; Shift moves further), click the small **navigator** overview to
  center the view there, or use the Pan sliders.
- **Fast preview, then full resolution** — a large photo (long edge over 2560 px) is previewed from a
  2048 px copy so sliders stay quick. When you stop adjusting for a moment, the canvas line under the
  zoom buttons reads "Rendering full resolution…" and the preview is replaced by a full-resolution
  render ("✨ Full-resolution view"); touching a control goes back to the fast preview. Save then
  reuses that render instead of computing it again.
- **🔍 1:1 / Fit (Z)**, **👁 Before / After (B)**, **# Grid (G)** — one-click toggles, also on the
  keyboard.
- **⇄ Split View** — a draggable before/after comparison slider over the same zoomed/panned view.
- **👁️ Original** — shows the untouched source image instead of your edit (disabled while Split
  View is active, since that already shows both).
- **🧭 Guides & Grid** — a composition aid, purely visual: it's never sent through the processing
  backend and never appears in an exported file.
  - **Grid overlay** — None, Rule of Thirds, Grid 4x4, or Golden Ratio.
  - **Custom guides** — add any number of horizontal or vertical guide lines at a chosen position
    (%), each removable individually.
  - The free-form Crop rectangle's corner handles (see [Crop](#5-crop-rotate--straighten--geometry)) snap to whichever grid
    lines/guides are active.
- **Histogram** — a live per-channel (Red/Green/Blue) chart of the current edited image.
- **⚠️ Show clipping** — highlights blocked shadows (blue) and blown highlights (red) directly on
  the canvas. **Clipping margin** also flags tones within that many levels of pure black or white.
- **⬇ Export tab → Export settings** — resize by long edge, output sharpening (screen / matte / glossy), which
  metadata to keep (all, copyright & camera only, or none, plus Remove GPS — on by default) and,
  when the optional **GPU pack** is installed (see [AI Upscale and the GPU pack](#ai-upscale-and-the-gpu-pack)),
  AI Upscale 2x/4x. **File name** is a template:
  `{stem}` (the original's name), `{date}`, `{width}`, `{height}`, `{format}`. **Bit depth** (PNG and
  TIFF): **16-bit** writes a genuine 16-bit file — the edit is rendered at full resolution in floating
  point from the full-precision original (16-bit PNG/TIFF scans and camera RAW keep their extra
  detail), so skies and gradients stay free of banding when you keep editing or print. It looks the
  same as the preview (at most about 1 % brighter after very heavy edits). Both 16-bit PNG and TIFF
  keep the colour profile and the camera metadata (exposure details and GPS included, following the
  metadata choice above). **Color space**: **sRGB** (the default -- right for the web and most
  screens) or **Display P3**, the wider space of current phones, Macs and many monitors. A Display P3
  export is rendered from the original file in a Display P3 working space, so colours sRGB can't show
  -- the saturated greens, reds and oranges a camera RAW or a wide-gamut photo can hold -- are kept
  instead of being clipped; the file carries a Display P3 profile. The preview on screen is still
  sRGB, so those colours look a little less vivid there than in the exported file on a P3 screen. The watermark has its own **Font** choice. **Save preset** keeps
  these settings under a name; pick one and **Apply** to restore them. The file is only produced when
  you click Save (or press **Ctrl+E**), and it carries the original photo's camera metadata and colour
  profile.
- **🏷️ Watermark** (⬇ Export tab) — optional text stamped only on exported files.
- **Export Format / Quality / Save** (under the canvas) — choose JPEG, PNG, WEBP, or TIFF (Quality only applies to
  JPEG/WEBP), then download the edited result.
- **📦 Export Multiple Sizes** (⬇ Export tab) — check any combination of the named export profiles (Web, Social,
  Print, Archive; Web and Social carry no EXIF, and **Remove GPS location** strips the location from
  Print/Archive too) and click **Prepare ZIP**, then **⬇️ Download ZIP** — gets you the current photo
  at several sizes/formats in one click, instead of downloading each one separately. This is
  different from the Batch Processor (below): this is *one photo, several output variants*; Batch
  is *one recipe, many photos*.

---

## Projects

Save everything — the source photo, every adjustment, your full layer stack, and your mask/removal
selections — into one `.photoproject` file you can reopen later:

- **💾 Save Project (.photoproject)** (⬇ Export tab) downloads the project in one click. If the
  project would be too large to reopen (very large image layers), the button is disabled with a
  warning.
- **📁 Open a saved project** (top of the app, above the uploader) restores a `.photoproject` back
  into the workspace exactly as you left it. A corrupted or hand-edited file shows a clear error
  instead of crashing the app.
- **Every layer stays editable** — Text, Shape and Pen path layers reopen as Text/Shape/Path layers,
  Image layers keep their placement, and every layer's stack position, opacity, blend mode and Layer
  Styles round-trip. (Projects saved by older versions reopen with those layers as plain images.)
- On the desktop app, double-clicking a `.photoproject` file (when the installer's file association
  was chosen) opens it directly.

---

## AI Upscale and the GPU pack

AI Upscale (Real-ESRGAN, 2x/4x on export) needs PyTorch, which is large, so the desktop release
ships it separately:

- **PhotoStudio-<version>-Windows.7z** — the CPU base app, with no PyTorch dependency.
- **PhotoStudio-<version>-GPU-pack.7z** — optional. Extract it **next to PhotoStudio.exe** so a
  `gpu-pack` folder sits beside the exe (or into `%LOCALAPPDATA%\PhotoStudio\gpu-pack`), then
  restart the app. The installer offers the same pack as an optional "GPU pack" component when it
  was built with one.
- **⬇ Export → AI Upscale** appears once the pack is found; ☰ Library & Batch → System Health shows
  where it was found, or why a pack was ignored (for example one built for a different version).
  An NVIDIA GPU makes it fast; it also runs, slowly, on the CPU.
- **Use GPU for export resize/sharpen** (⬇ Export) opts into CUDA for final export resizing and
  output sharpening. It requires CUDA-enabled PyTorch from the GPU pack (or a compatible source
  install); file encoding and the editing pipeline remain on the CPU. If CUDA is missing or a GPU
  operation fails, the export logs a warning and retries on the CPU. The option is off by default.
- The **Batch Processor** has a separate **Use GPU for profile resizing** option for its Web/Social
  profiles. It uses the same optional CUDA support and CPU fallback; Print/Archive do not resize.
- From source, install `requirements-gpu.txt` instead (see its header for the CUDA build of torch).

---

## Installing

- **Portable:** extract `PhotoStudio-<version>-Windows.7z` anywhere and run `PhotoStudio.exe`. The
  Microsoft Edge WebView2 runtime the window needs is bundled (`WebView2Runtime` next to the exe),
  so nothing else has to be installed.
- **Installer:** `PhotoStudio-Setup-<version>.exe` asks whether to install **for all users**
  (needs administrator rights, goes to Program Files) or **just for you** (no administrator rights,
  goes to `%LOCALAPPDATA%\Programs`). It can associate `.photoproject` files with Photo Studio,
  creates Start-menu and desktop shortcuts, and — when it was built with one — offers the **GPU
  pack** as an optional component. Uninstalling asks whether to delete your presets, catalog and
  logs too; answering No (or a silent uninstall) keeps them.
- **Your files:** settings, custom presets, export presets, the library catalog, logs and a GPU
  pack you extracted there live in `%LOCALAPPDATA%\PhotoStudio`, never in the install folder.
  Photos are only read; edits are written only where you save or export them.

---

## Command Line

The same edit pipeline runs without the window (from source: `python -m app`), for one photo or a
whole folder:

```powershell
python -m app --input photo.jpg --output out.jpg --exposure-ev 0.5 --contrast 1.1 --vibrance 20
python -m app --input-dir .\shoot --output-dir .\edited --preset travel-clean --quality 90
python -m app --input scan.tif --output scan-edit.tif --bit-depth 16 --curve "0:8,128:140,255:250"
python -m app --input photo.jpg --output square.jpg --rotate 90 --straighten 1.5 --aspect 1 1
python -m app --input photo.jpg --output clean.jpg --remove-mask object-mask.png
```

- Tone/colour: `--exposure-ev` (stops), `--contrast`/`--saturation`/`--brightness` (multipliers,
  1.0 = unchanged), `--highlights`/`--shadows`, `--temperature`/`--tint`, `--vibrance`,
  `--texture`/`--clarity`, `--dehaze`, `--curve "in:out,..."`, `--denoise 0-9` (+ `--denoise-color`),
  `--sharpen`, `--auto-enhance`, `--preset <built-in look>`.
- Geometry: `--rotate 90|180|270`, `--straighten <degrees>`, `--constrain-crop`,
  `--crop LEFT TOP RIGHT BOTTOM` (pixels), `--aspect W H` (centred crop).
- Local edits (single photo): `--mask mask.png` with `--mask-exposure`/`--mask-contrast`/
  `--mask-saturation` (multipliers; white = edited, black = protected, grey = partial), and
  `--remove-mask mask.png` for object removal.
- Saved looks: `--recipe preset.json` (a custom preset, or an exported presets file with
  `--recipe-name`); `--options options.json` takes a raw pipeline options object.
- Output: `--quality 1-100` (JPEG/WEBP), `--metadata all|camera|none`, `--remove-gps`,
  `--bit-depth 16` (PNG/TIFF), `--gpu-export` (CUDA resize/sharpen when the GPU pack's torch is
  available), `--ai-upscale 2|4` (needs the GPU pack).

`python -m app --help` lists every option with its range.

---

## Keyboard Shortcuts Reference

| Shortcut | Action | Where |
|---|---|---|
| Ctrl+Z | Undo | Anywhere |
| Ctrl+Shift+Z | Redo | Anywhere |
| Ctrl+E | Save (export) | Anywhere |
| Z | 1:1 / Fit | Anywhere |
| B | Before / After | Anywhere |
| G | Grid on/off | Anywhere |
| ← ↑ → ↓ (Shift = 10×) | Nudge the focused crop corner or crop box (1 px steps) | ⛶ Geometry → Crop, free-form crop editor |
| ← ↑ → ↓ / Alt+arrows / Delete | Move the selected point / its curve handle / remove it | Layers tab, Pen editor |
| ↑ / ↓ (Shift = 10×) | Raise / lower the focused curve point | 🎨 Curves & Color, curve editor (Tab to a point) |
| [ / ] (Shift = 10×) | Smaller / larger brush | Masks & Repair, paint editor |
| ← ↑ → ↓ / Alt+arrows | Move the last radial or gradient / only its end point | Masks & Repair, paint editor |
| Drag, or ← ↑ → ↓ | Pan the zoomed view (Shift = half a screen) | Canvas at 100 % and above |
| 1–5 | Rate current photo | Library Catalog, Rate & Flag panel |
| 0 | Clear rating | Library Catalog, Rate & Flag panel |
| X | Toggle Reject | Library Catalog, Rate & Flag panel |
| ← / → | Previous / Next photo | Library Catalog, Rate & Flag panel |

---

## Known Limitations

This app is CPU-first and offline by design — no cloud processing or required GPU dependency. GPU
export transforms and AI Upscale are optional. That means a few things are deliberately simpler
than a cloud/AI-powered tool: lens and denoise are classical
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
