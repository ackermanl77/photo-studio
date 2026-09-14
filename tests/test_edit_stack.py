from PIL import Image

from app.edit_stack import EditStack
from app.image_processor import PhotoProcessor


def test_edit_stack_reorders_disables_and_round_trips(tmp_path):
    source = tmp_path / "stack.png"
    Image.new("RGB", (12, 12), (100, 110, 120)).save(source)
    stack = EditStack()
    stack.add("Exposure", "apply_exposure", {"value": 1.2})
    stack.add("Contrast", "apply_contrast", {"value": 1.15})
    stack.move("Contrast", -1).set_enabled("Contrast", False)

    processor = stack.apply(PhotoProcessor(source))
    restored = EditStack.from_json(stack.to_json())

    assert [operation.name for operation in stack.operations] == ["Contrast", "Exposure"]
    assert restored.operations[0].enabled is False
    assert processor.image.size == (12, 12)


def test_edit_stack_can_remove_operations():
    stack = EditStack().add("Tone", "apply_contrast", {"value": 1.1})
    stack.remove("Tone")
    assert stack.operations == []
