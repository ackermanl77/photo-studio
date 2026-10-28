"""XMP sidecar files for the Library's ratings, reject flag and tags (report item F8).

Lightroom, Capture One, Bridge and darktable keep a photo's star rating in ``xmp:Rating``
(``-1`` = rejected) and its keywords in ``dc:subject`` -- in a ``.xmp`` file next to the photo.
The Library reads those when it catalogues a photo for the first time, and can write them back
when a photo is rated or tagged, so a cull done here shows up in those applications and the
other way round.

Sidecar name: ``IMG_0001.xmp`` next to a RAW file (the Lightroom / Capture One convention) and
``photo.jpg.xmp`` next to anything else (darktable's), so a RAW+JPEG pair doesn't share one file.
An existing sidecar with either name is used as found. Writing changes only the rating and the
keyword list: everything else in an existing sidecar (develop settings, other namespaces) is kept.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import atomic_write_text
from app.raw_import import is_raw_path

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
DC_NS = "http://purl.org/dc/elements/1.1/"
_STANDARD_PREFIXES = {"x": X_NS, "rdf": RDF_NS, "xmp": XMP_NS, "dc": DC_NS}

#: The catalog tag that stands for the reject flag (xmp:Rating -1 in a sidecar).
REJECTED_TAG = "rejected"


def _q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def sidecar_candidates(photo: str | Path) -> list[Path]:
    """Where a sidecar for ``photo`` may be, the name this module writes first."""
    photo = Path(photo)
    stem_name, full_name = photo.with_suffix(".xmp"), photo.with_name(photo.name + ".xmp")
    return [stem_name, full_name] if is_raw_path(photo) else [full_name, stem_name]


def find_sidecar(photo: str | Path) -> Path | None:
    return next((path for path in sidecar_candidates(photo) if path.is_file()), None)


def _descriptions(root: ET.Element) -> list[ET.Element]:
    return list(root.iter(_q(RDF_NS, "Description")))


def read_sidecar(photo: str | Path, content: bytes | None = None) -> dict | None:
    """``{"rating": 0-5, "rejected": bool, "tags": [...]}`` from ``photo``'s sidecar, or None when
    there is none or it can't be read. `content`, when supplied, is the already-read sidecar
    snapshot used by the catalog to fingerprint and parse the same bytes. A rating outside 0-5
    other than -1 is clamped."""
    path = find_sidecar(photo)
    if path is None:
        return None
    try:
        root = ET.fromstring(content) if content is not None else ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    rating = None
    label = ""
    tags: list[str] = []
    for description in _descriptions(root):
        raw_label = description.get(_q(XMP_NS, "Label"))
        label_element = description.find(_q(XMP_NS, "Label"))
        if raw_label is None and label_element is not None:
            raw_label = label_element.text
        if raw_label and not label:
            label = raw_label.strip()
        raw = description.get(_q(XMP_NS, "Rating"))
        element = description.find(_q(XMP_NS, "Rating"))
        if raw is None and element is not None:
            raw = element.text
        if raw is not None and rating is None:
            try:
                rating = int(round(float(raw)))
            except ValueError:
                pass
        subject = description.find(_q(DC_NS, "subject"))
        if subject is not None:
            tags += [li.text.strip() for li in subject.iter(_q(RDF_NS, "li")) if li.text and li.text.strip()]
    rejected = rating is not None and rating < 0
    return {
        "rating": 0 if rating is None or rejected else max(0, min(5, rating)),
        "rejected": rejected,
        "tags": [tag for tag in dict.fromkeys(tags) if tag != REJECTED_TAG],
        "label": label,
    }


def _register_namespaces(path: Path) -> None:
    """Keep an existing file's own prefixes (crs:, lr:, darktable:, ...) when it is rewritten --
    ElementTree would otherwise rename them ns0, ns1, ..."""
    for prefix, uri in _STANDARD_PREFIXES.items():
        ET.register_namespace(prefix, uri)
    try:
        for _event, (prefix, uri) in ET.iterparse(path, events=("start-ns",)):
            if prefix:
                ET.register_namespace(prefix, uri)
    except (ET.ParseError, ValueError):
        pass


def _new_tree() -> ET.ElementTree:
    root = ET.Element(_q(X_NS, "xmpmeta"))
    rdf = ET.SubElement(root, _q(RDF_NS, "RDF"))
    ET.SubElement(rdf, _q(RDF_NS, "Description"), {_q(RDF_NS, "about"): ""})
    return ET.ElementTree(root)


def write_sidecar(photo: str | Path, rating: int, tags, rejected: bool = False, label: str | None = None) -> Path:
    """Write ``rating`` (0-5; ``-1`` when ``rejected``) and ``tags`` into ``photo``'s sidecar,
    creating it when there is none, and ``label`` into xmp:Label unless it is None ("" removes it).
    Returns the sidecar's path. Raises OSError when it can't be
    written; an existing sidecar that isn't valid XML is left alone (ValueError)."""
    path = find_sidecar(photo) or sidecar_candidates(photo)[0]
    if path.is_file():
        _register_namespaces(path)
        try:
            tree = ET.parse(path)
        except ET.ParseError as error:
            raise ValueError(f"{path.name} is not a readable XMP file: {error}") from error
    else:
        for prefix, uri in _STANDARD_PREFIXES.items():
            ET.register_namespace(prefix, uri)
        tree = _new_tree()
    descriptions = _descriptions(tree.getroot())
    if not descriptions:
        rdf = tree.getroot().find(_q(RDF_NS, "RDF"))
        if rdf is None:
            rdf = ET.SubElement(tree.getroot(), _q(RDF_NS, "RDF"))
        descriptions = [ET.SubElement(rdf, _q(RDF_NS, "Description"), {_q(RDF_NS, "about"): ""})]
    value = "-1" if rejected else str(max(0, min(5, int(rating))))
    # One rating and one keyword list: drop them from every description, then set them on the first.
    for description in descriptions:
        description.attrib.pop(_q(XMP_NS, "Rating"), None)
        for name in ((XMP_NS, "Rating"), (DC_NS, "subject")):
            for element in description.findall(_q(*name)):
                description.remove(element)
    first = descriptions[0]
    first.set(_q(XMP_NS, "Rating"), value)
    if label is not None:
        for description in descriptions:
            description.attrib.pop(_q(XMP_NS, "Label"), None)
            for element in description.findall(_q(XMP_NS, "Label")):
                description.remove(element)
        if label:
            first.set(_q(XMP_NS, "Label"), label)
    keywords = [tag for tag in dict.fromkeys(t.strip() for t in tags if t and t.strip()) if tag != REJECTED_TAG]
    if keywords:
        bag = ET.SubElement(ET.SubElement(first, _q(DC_NS, "subject")), _q(RDF_NS, "Bag"))
        for keyword in keywords:
            ET.SubElement(bag, _q(RDF_NS, "li")).text = keyword
    ET.indent(tree, space=" ")
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=False)
    atomic_write_text(path, buffer.getvalue().decode("utf-8") + "\n")
    return path
