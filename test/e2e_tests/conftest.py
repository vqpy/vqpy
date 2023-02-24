import pytest
from pathlib import Path
import sys

root = Path(__file__).parent.parent.parent
example_path = (root / "examples/").as_posix()


@pytest.fixture
def setup_example_path():
    sys.path.append(example_path)
    yield
    sys.path.remove(example_path)
