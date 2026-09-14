from pathlib import Path

from app.release_preflight import preflight_ok, run_preflight


def test_release_preflight_checks_project_inputs(tmp_path):
    for relative in ["app/frontend_unified_app.py", "app/desktop.py", "PhotoStudio.spec", "requirements.txt", "requirements-desktop.txt"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")

    checks = run_preflight(tmp_path, run_tests=False)

    assert any(check.name == "PhotoStudio.spec" and check.status == "ok" for check in checks)
    assert preflight_ok(checks)
