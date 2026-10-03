from app.diagnostics import check_environment


def test_environment_diagnostics_report_required_runtime():
    checks = check_environment()
    names = {check.name for check in checks}
    assert "Python" in names
    assert "PIL" in names
    assert "numpy" in names
    assert {check.status for check in checks} <= {"ok", "missing", "optional", "warning", "error"}
