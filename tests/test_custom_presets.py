from app.custom_presets import PresetStore


def test_custom_preset_store_round_trip_and_delete(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    recipe = {"exposure": 1.1, "contrast": 1.2}

    store.save("My Look", recipe)
    assert store.names() == ["My Look"]
    assert store.get("My Look") == recipe
    store.delete("My Look")
    assert store.names() == []
