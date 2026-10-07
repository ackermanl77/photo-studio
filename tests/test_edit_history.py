from app.edit_history import EditHistory


def test_record_dedupes_and_undo_redo_cycle_correctly():
    history = EditHistory()
    history.record({"exposure": 1.0})
    history.record({"exposure": 1.0})  # unchanged -- must not create a second step
    history.record({"exposure": 1.2})

    assert len(history.snapshots) == 2
    assert history.can_undo
    assert not history.can_redo

    assert history.undo() == {"exposure": 1.0}
    assert history.can_redo
    assert history.redo() == {"exposure": 1.2}


def test_record_after_undo_truncates_the_redo_branch():
    history = EditHistory()
    history.record({"exposure": 1.0})
    history.record({"exposure": 1.2})
    history.undo()
    history.record({"exposure": 1.5})  # a fresh edit after undo -- the old 1.2 step is gone

    assert not history.can_redo
    assert history.snapshots == [{"exposure": 1.0}, {"exposure": 1.5}]


def test_save_snapshot_stores_a_deep_copy_independent_of_the_caller():
    history = EditHistory()
    options = {"exposure": 1.1, "mask_layers": [{"id": "a"}]}

    history.save_snapshot("Look 1", options)
    options["exposure"] = 9.9
    options["mask_layers"].append({"id": "b"})

    restored = history.restore_snapshot("Look 1")
    assert restored == {"exposure": 1.1, "mask_layers": [{"id": "a"}]}


def test_save_snapshot_rejects_an_empty_or_whitespace_name():
    history = EditHistory()
    try:
        history.save_snapshot("   ", {"exposure": 1.0})
        assert False, "expected a ValueError"
    except ValueError:
        pass
    assert history.snapshot_names == []


def test_restore_snapshot_of_an_unknown_name_returns_none():
    history = EditHistory()
    assert history.restore_snapshot("Nope") is None


def test_restoring_a_snapshot_does_not_consume_it_or_touch_undo_redo_history():
    """A snapshot is a bookmark, not a one-time jump: restoring it any number of times keeps
    returning the same saved state, and never mutates the separate linear Undo/Redo step list
    (self.snapshots/self.cursor) -- only the frontend's own subsequent history.record() call
    (driven by _build_recipe() on the next rerun) folds the restored state into Undo/Redo, the
    same way applying a preset does."""
    history = EditHistory()
    history.record({"exposure": 1.0})
    history.record({"exposure": 1.2})
    steps_before = list(history.snapshots)

    history.save_snapshot("Bookmark", {"exposure": 5.0})
    assert history.restore_snapshot("Bookmark") == {"exposure": 5.0}
    assert history.restore_snapshot("Bookmark") == {"exposure": 5.0}  # restorable again
    assert "Bookmark" in history.snapshot_names

    assert history.snapshots == steps_before  # untouched by save/restore


def test_delete_snapshot_removes_it_and_is_a_no_op_for_an_unknown_name():
    history = EditHistory()
    history.save_snapshot("Temp", {"exposure": 1.0})

    history.delete_snapshot("Temp")
    assert history.snapshot_names == []
    history.delete_snapshot("Temp")  # no-op, must not raise

    assert history.restore_snapshot("Temp") is None


def test_snapshot_names_reflects_insertion_and_survives_unrelated_undo_redo():
    history = EditHistory()
    history.save_snapshot("A", {"exposure": 1.0})
    history.save_snapshot("B", {"exposure": 2.0})

    history.record({"exposure": 1.0})
    history.record({"exposure": 3.0})
    history.undo()

    assert set(history.snapshot_names) == {"A", "B"}


def test_history_steps_describe_what_each_step_changed():
    from app.edit_history import describe_change
    from app.edit_schema import recipe_defaults

    base = recipe_defaults()
    assert describe_change(None, base) == "Opened"
    assert describe_change(base, dict(base, exposure_ev=0.3)) == "Exposure (EV) 0.3"
    assert describe_change(base, dict(base, black_white=True)) == "Convert to Black & White on"
    assert describe_change(base, dict(base, contrast_amount=10, clarity=5)) == "Contrast, Clarity"
    assert describe_change(base, dict(base, hsl_blue_sat=-20)) == "HSL Blue Saturation -20"
    assert describe_change(base, dict(base, curve_mid=140, curve_white=250)) == "Tone curve"
    assert describe_change(base, dict(base, mask_layers=[{"id": "a"}])) == "Masks"
    assert describe_change(base, dict(base, exposure_ev=1.0, whites=5, tint=3)) == "Exposure (EV) +2 more"


def test_jump_moves_the_cursor_without_dropping_the_redo_branch():
    history = EditHistory()
    for value in (0.0, 0.5, 1.0):
        history.record({"exposure_ev": value})
    assert history.jump(0) == {"exposure_ev": 0.0} and history.cursor == 0 and history.can_redo
    history.record({"exposure_ev": 0.0})  # the frontend records the restored recipe -- unchanged
    assert len(history.snapshots) == 3
    assert history.jump(99) == {"exposure_ev": 0.0}  # out of range: stays put
