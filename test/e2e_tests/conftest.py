import pytest
from pathlib import Path
import sys

root = Path(__file__).parent.parent.parent
example_path = (root / "examples/").as_posix()
fall_detection_lib_path = (root / "Human-Falling-Detect-Tracks").as_posix()


@pytest.fixture
def setup_example_path():
    sys.path.append(example_path)
    yield
    sys.path.remove(example_path)


@pytest.fixture
def setup_fall_detection_lib_path():
    sys.path.append(fall_detection_lib_path)
    yield
    sys.path.remove(fall_detection_lib_path)
