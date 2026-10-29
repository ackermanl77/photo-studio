"""Photoshop filter plug-ins (PS20, app.filter_plugins): the plugin_filters option, the recipe entry,
the helper protocol (run with a stand-in for PhotoStudio.FilterHost.exe), the result cache and the
pipeline step. The real helper is exercised when scripts/build_filter_host.ps1 has built it."""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import filter_plugins
from app.edit_schema import OptionsError, parse_options, sanitize_recipe
from app.filter_plugins import FilterCancelled, FilterPluginError, apply_filters, clean_steps, configure
from app.pipeline import run_pipeline

PARAMS = base64.b64encode(b"saved settings").decode("ascii")
STEP = {"file": r"C:\Filters\Invert.8bf", "entry": "InvertEntry", "title": "Invert", "params": PARAMS}


def test_clean_steps_accepts_valid_steps_and_refuses_the_rest():
    assert clean_steps([STEP]) == [STEP]
    for bad in (
        "not a list",
        [{**STEP, "file": "invert.dll"}],
        [{**STEP, "entry": ""}],
        [{**STEP, "params": ""}],
        [{**STEP, "params": "not base64!"}],
        [{**STEP, "extra": 1}],
        [STEP] * (filter_plugins.MAX_FILTERS + 1),
        [{**STEP, "params": base64.b64encode(b"x" * (filter_plugins.MAX_PARAMS_BYTES + 1)).decode()}],
    ):
        with pytest.raises(ValueError):
            clean_steps(bad)


def test_the_option_is_validated_like_every_other():
    assert parse_options({"plugin_filters": [STEP]})["plugin_filters"] == [STEP]
    assert parse_options({"plugin_filters": []})["plugin_filters"] is None
    with pytest.raises(OptionsError, match="plugin_filters"):
        parse_options({"plugin_filters": [{"file": "a.8bf"}]})


def test_the_recipe_keeps_valid_steps_with_their_ids_and_drops_a_broken_list():
    clean = sanitize_recipe({"plugin_filters": [{**STEP, "id": "abc"}]})
    assert clean["plugin_filters"] == [{**STEP, "id": "abc"}]
    assert sanitize_recipe({"plugin_filters": [{**STEP, "id": "abc"}, {"file": "x.8bf"}]})["plugin_filters"] == []
    assert sanitize_recipe({"plugin_filters": "nonsense"})["plugin_filters"] == []


class _FakeHost:
    """Stands in for PhotoStudio.FilterHost.exe: inverts the input PNG, records its arguments and
    writes back its settings, or exits with a chosen code."""

    def __init__(self, exit_code=0, params_out=b"new settings"):
        self.calls, self.exit_code, self.params_out = [], exit_code, params_out

    def __call__(self, args, timeout):
        self.calls.append(list(args))
        options = {args[i][2:]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--") and args[i] != "--ui"}
        if self.exit_code == 0:
            with Image.open(options["input"]) as source:
                inverted = np.asarray(source.convert("RGBA")).copy()
            inverted[..., :3] = 255 - inverted[..., :3]
            inverted[..., 3] = 255  # plug-ins often return opaque pixels: the alpha must come from the source
            Image.fromarray(inverted, mode="RGBA").save(options["output"])
            Path(options["params-out"]).write_bytes(self.params_out)
        return subprocess.CompletedProcess(args, self.exit_code, b"", b"Boom: the plug-in failed" if self.exit_code == 2 else b"")


@pytest.fixture()
def fake_host(monkeypatch):
    host = _FakeHost()
    monkeypatch.setattr(filter_plugins, "_run", host)
    filter_plugins._cache.clear()
    return host


def test_apply_replays_the_saved_settings_without_a_dialog(fake_host):
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    result = apply_filters(image, [STEP])
    assert result.mode == "RGB" and result.getpixel((0, 0)) == (245, 235, 225)
    args = fake_host.calls[0]
    assert "--ui" not in args and args[args.index("--entry") + 1] == "InvertEntry"
    assert "--params-in" in args


def test_a_repeated_input_comes_from_the_cache(fake_host):
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    first = apply_filters(image, [STEP])
    second = apply_filters(image, [STEP])
    assert len(fake_host.calls) == 1 and second.tobytes() == first.tobytes()
    apply_filters(Image.new("RGB", (8, 6), (11, 20, 30)), [STEP])  # other pixels: runs again
    assert len(fake_host.calls) == 2


def test_transparency_is_kept_from_the_source(fake_host):
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (0, 0, 0, 255))
    result = apply_filters(image, [STEP])
    assert result.mode == "RGBA" and result.getpixel((0, 0))[3] == 0 and result.getpixel((1, 1)) == (255, 255, 255, 255)


def test_configure_shows_the_dialog_and_returns_the_new_settings(fake_host):
    step = configure(Image.new("RGB", (4, 4)), {k: v for k, v in STEP.items() if k != "params"})
    assert "--ui" in fake_host.calls[0] and "--params-in" not in fake_host.calls[0]
    assert base64.b64decode(step["params"]) == b"new settings"


def test_a_cancelled_dialog_and_a_failing_plugin_raise(monkeypatch):
    monkeypatch.setattr(filter_plugins, "_run", _FakeHost(exit_code=3))
    with pytest.raises(FilterCancelled):
        configure(Image.new("RGB", (4, 4)), STEP)
    monkeypatch.setattr(filter_plugins, "_run", _FakeHost(exit_code=2))
    filter_plugins._cache.clear()
    with pytest.raises(FilterPluginError, match="plug-in failed"):
        apply_filters(Image.new("RGB", (4, 4)), [STEP])


def test_the_pipeline_runs_the_filters_after_the_effects(fake_host):
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    result = run_pipeline(image, parse_options({"plugin_filters": [STEP]}))
    assert result.getpixel((5, 5)) == (245, 235, 225)


def test_without_the_helper_it_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(filter_plugins, "host_path", lambda: None)
    assert not filter_plugins.available()
    with pytest.raises(FilterPluginError, match="build_filter_host"):
        filter_plugins.list_filters(Path(__file__).parent)


@pytest.mark.skipif(sys.platform != "win32" or filter_plugins.host_path() is None,
                    reason="the filter host isn't built (scripts/build_filter_host.ps1)")
def test_the_real_helper_lists_a_folder_without_plugins(tmp_path):
    assert filter_plugins.list_filters(tmp_path) == []
    (tmp_path / "broken.8bf").write_bytes(b"MZ not really a plug-in")
    assert filter_plugins.list_filters(tmp_path) == []  # an unreadable plug-in is skipped, not fatal
