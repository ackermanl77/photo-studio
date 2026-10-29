// Creates the small PSD/PSB test documents in this folder for tests/test_psd.py (app/psd.py's reader),
// with the Photoshop it runs in: File > Scripts > Browse..., or from PowerShell
//   (New-Object -ComObject Photoshop.Application).DoJavaScriptFile("<this file>")
// Only new documents are touched; the document that was active is re-activated at the end.
// The committed files were made by Photoshop 2026 (27.11) on Windows.
#target photoshop
var OUT = File($.fileName).parent.fsName.replace(/\\/g, "/") + "/";  // next to this script
var cTID = function (s) { return charIDToTypeID(s); };
var sTID = function (s) { return stringIDToTypeID(s); };
var log = [];
var original = app.documents.length ? app.activeDocument : null;
var oldDialogs = app.displayDialogs;
var oldUnits = app.preferences.rulerUnits;
app.displayDialogs = DialogModes.NO;
app.preferences.rulerUnits = Units.PIXELS;
new Folder(OUT).create();

function rgb(r, g, b) { var c = new SolidColor(); c.rgb.red = r; c.rgb.green = g; c.rgb.blue = b; return c; }

function fillRect(doc, x0, y0, x1, y1, color) {
    doc.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    doc.selection.fill(color);
    doc.selection.deselect();
}

function addRevealSelectionMask() {
    // Layer > Layer Mask > Reveal Selection, on the active layer.
    var desc = new ActionDescriptor();
    desc.putClass(cTID("Nw  "), cTID("Chnl"));
    var ref = new ActionReference();
    ref.putEnumerated(cTID("Chnl"), cTID("Chnl"), cTID("Msk "));
    desc.putReference(cTID("At  "), ref);
    desc.putEnumerated(cTID("Usng"), cTID("UsrM"), cTID("RvlS"));
    executeAction(cTID("Mk  "), desc, DialogModes.NO);
}

function addHueSaturationLayer() {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putClass(sTID("adjustmentLayer"));
    desc.putReference(cTID("null"), ref);
    var layerDesc = new ActionDescriptor();
    var typeDesc = new ActionDescriptor();
    typeDesc.putBoolean(sTID("colorize"), false);
    layerDesc.putObject(cTID("Type"), sTID("hueSaturation"), typeDesc);
    desc.putObject(cTID("Usng"), sTID("adjustmentLayer"), layerDesc);
    executeAction(cTID("Mk  "), desc, DialogModes.NO);
}

// A layered document: Background, "Red box" (multiply 50 %), a group with "Inside group", a masked
// "Masked green" layer, a text layer "Hello PSD" and a Hue/Saturation adjustment layer.
function buildLayered(doc) {
    var red = doc.artLayers.add(); red.name = "Red box";
    fillRect(doc, 4, 4, 30, 24, rgb(255, 0, 0));
    red.opacity = 50; red.blendMode = BlendMode.MULTIPLY;

    var group = doc.layerSets.add(); group.name = "My group";
    var inside = group.artLayers.add(); inside.name = "Inside group";
    fillRect(doc, 34, 4, 60, 20, rgb(0, 0, 255));

    doc.activeLayer = doc.layers[0];
    var green = doc.artLayers.add(); green.name = "Masked green";
    green.move(doc, ElementPlacement.PLACEATBEGINNING);
    fillRect(doc, 4, 28, 60, 44, rgb(0, 200, 0));
    doc.selection.select([[4, 28], [32, 28], [32, 44], [4, 44]]);
    addRevealSelectionMask();
    doc.selection.deselect();

    var text = doc.artLayers.add(); text.kind = LayerKind.TEXT; text.name = "Hello PSD";
    text.textItem.contents = "Hello PSD";
    text.textItem.size = new UnitValue(10, "px");
    text.textItem.color = rgb(10, 20, 30);
    text.textItem.position = [new UnitValue(6, "px"), new UnitValue(16, "px")];

    addHueSaturationLayer();
}

function save(doc, name, psb) {
    var file = new File(OUT + name);
    if (psb) {
        var desc = new ActionDescriptor();
        desc.putObject(cTID("As  "), sTID("largeDocumentFormat"), new ActionDescriptor());
        desc.putPath(cTID("In  "), file);
        desc.putBoolean(cTID("Cpy "), true);
        executeAction(cTID("save"), desc, DialogModes.NO);
    } else {
        var options = new PhotoshopSaveOptions();
        options.layers = true;
        doc.saveAs(file, options, true);
    }
}

function make(name, bits, mode, layered, psb) {
    var doc = null;
    try {
        doc = app.documents.add(64, 48, 72, "fixture-" + name, NewDocumentMode.RGB, DocumentFill.WHITE, 1,
            bits === 16 ? BitsPerChannelType.SIXTEEN : BitsPerChannelType.EIGHT);
        if (layered) { buildLayered(doc); }
        if (mode === "cmyk") { doc.changeMode(ChangeMode.CMYK); }
        if (mode === "lab") { doc.changeMode(ChangeMode.LAB); }
        if (mode === "gray") { doc.changeMode(ChangeMode.GRAYSCALE); }
        save(doc, name, psb);
        log.push("ok " + name);
    } catch (e) {
        log.push("FAILED " + name + ": " + e);
    }
    if (doc) { try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (e2) { log.push("close failed " + name); } }
}

make("rgb8_layers.psd", 8, "rgb", true, false);
make("rgb16_layers.psd", 16, "rgb", true, false);
make("gray16_layers.psd", 16, "gray", true, false);
make("cmyk8_layers.psd", 8, "cmyk", true, false);
make("lab8_layers.psd", 8, "lab", true, false);
make("rgb8_layers.psb", 8, "rgb", true, true);

app.displayDialogs = oldDialogs;
app.preferences.rulerUnits = oldUnits;
if (original) { try { app.activeDocument = original; } catch (e3) { log.push("could not re-activate the original document"); } }
log.join("\n");
