import json

from app.custom_presets import PresetStore
from app.preset_exchange import export_presets, import_presets


def test_preset_exchange_round_trip(tmp_path):
    first = PresetStore(tmp_path / "first.json")
    second = PresetStore(tmp_path / "second.json")
    first.save("Warm", {"temperature": 15})

    imported = import_presets(second, export_presets(first))

    assert imported == ["Warm"]
    assert second.get("Warm") == {"temperature": 15}


def test_preset_exchange_does_not_overwrite_by_default(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("Look", {"contrast": 1.1})
    imported = import_presets(store, json.dumps({"version": 1, "presets": {"Look": {"contrast": 1.5}}}))

    assert imported == []
    assert store.get("Look") == {"contrast": 1.1}
