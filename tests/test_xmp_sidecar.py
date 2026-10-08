"""XMP sidecars for Library ratings, the reject flag and tags (report item F8)."""

from app.library import index_folder, save_rating
from app.major_features import PhotoCatalog
from app.xmp_sidecar import find_sidecar, read_sidecar, sidecar_candidates, write_sidecar
from tests.synthetic_assets import photo_like

LIGHTROOM_SIDECAR = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
   crs:Exposure2012="+0.35"
   crs:WhiteBalance="As Shot">
   <xmp:Rating>3</xmp:Rating>
   <dc:subject>
    <rdf:Bag>
     <rdf:li>beach</rdf:li>
     <rdf:li>family</rdf:li>
    </rdf:Bag>
   </dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def _photo(tmp_path, name="shot.jpg"):
    path = tmp_path / name
    photo_like((32, 24)).save(path) if not name.lower().endswith(".dng") else path.write_bytes(b"not really raw")
    return path


def test_new_sidecar_round_trip_and_naming(tmp_path):
    jpeg = _photo(tmp_path)
    assert read_sidecar(jpeg) is None
    path = write_sidecar(jpeg, 4, ["Sea", "sunset", "Sea", "rejected"])
    assert path == tmp_path / "shot.jpg.xmp"  # darktable-style for non-RAW: no clash with a RAW twin
    assert read_sidecar(jpeg) == {"rating": 4, "rejected": False, "tags": ["Sea", "sunset"]}
    write_sidecar(jpeg, 4, ["Sea"], rejected=True)
    assert read_sidecar(jpeg) == {"rating": 0, "rejected": True, "tags": ["Sea"]}
    assert 'xmp:Rating="-1"' in path.read_text(encoding="utf-8")
    raw = tmp_path / "IMG_0001.dng"
    assert sidecar_candidates(raw)[0] == tmp_path / "IMG_0001.xmp"  # Lightroom / Capture One convention


def test_an_existing_lightroom_sidecar_keeps_its_develop_settings(tmp_path):
    raw = _photo(tmp_path, "IMG_0002.dng")
    sidecar = tmp_path / "IMG_0002.xmp"
    sidecar.write_text(LIGHTROOM_SIDECAR, encoding="utf-8")
    assert read_sidecar(raw) == {"rating": 3, "rejected": False, "tags": ["beach", "family"]}
    assert write_sidecar(raw, 5, ["beach"]) == sidecar
    text = sidecar.read_text(encoding="utf-8")
    assert 'crs:Exposure2012="+0.35"' in text and 'crs:WhiteBalance="As Shot"' in text  # prefixes kept
    assert "ns0:" not in text and text.count("Rating") == 1
    assert read_sidecar(raw) == {"rating": 5, "rejected": False, "tags": ["beach"]}


def test_a_photo_named_with_the_other_convention_is_found(tmp_path):
    jpeg = _photo(tmp_path, "other.jpg")
    (tmp_path / "other.xmp").write_text(LIGHTROOM_SIDECAR, encoding="utf-8")
    assert find_sidecar(jpeg) == tmp_path / "other.xmp"
    assert write_sidecar(jpeg, 1, []) == tmp_path / "other.xmp"
    assert read_sidecar(jpeg)["tags"] == []


def test_a_broken_sidecar_is_not_overwritten(tmp_path):
    jpeg = _photo(tmp_path)
    broken = tmp_path / "shot.jpg.xmp"
    broken.write_text("<x:xmpmeta", encoding="utf-8")
    assert read_sidecar(jpeg) is None
    catalog = PhotoCatalog(tmp_path / "catalog.sqlite")
    problem = save_rating(catalog, jpeg, 2, ["x"], write_xmp=True)
    assert problem and "XMP" in problem
    assert broken.read_text(encoding="utf-8") == "<x:xmpmeta"
    assert catalog.get(jpeg)["rating"] == 2  # the catalog is updated anyway


def test_scanning_imports_sidecars_only_for_new_photos(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    jpeg = _photo(folder, "a.jpg")
    (folder / "a.jpg.xmp").write_text(LIGHTROOM_SIDECAR.replace("<xmp:Rating>3", "<xmp:Rating>-1"), encoding="utf-8")
    database = tmp_path / "catalog.sqlite"
    index_folder(folder, database)
    catalog = PhotoCatalog(database)
    assert catalog.get(jpeg)["rating"] == 0
    assert sorted(catalog.get(jpeg)["tags"]) == ["beach", "family", "rejected"]
    catalog.upsert(jpeg, rating=5, tags=["keeper"])
    index_folder(folder, database)  # a rescan doesn't bring the sidecar's values back
    assert catalog.get(jpeg) == {"path": str(jpeg), "rating": 5, "tags": ["keeper"]}


def test_save_rating_writes_the_sidecar_only_when_asked(tmp_path):
    jpeg = _photo(tmp_path)
    catalog = PhotoCatalog(tmp_path / "catalog.sqlite")
    assert save_rating(catalog, jpeg, 3, ["a"]) is None
    assert find_sidecar(jpeg) is None
    assert save_rating(catalog, jpeg, 3, ["a", "rejected"], write_xmp=True) is None
    assert read_sidecar(jpeg) == {"rating": 0, "rejected": True, "tags": ["a"]}
