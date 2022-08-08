from pathlib import Path
import setuptools


def get_version() -> str:
    root = Path(__file__).parent.parent
    return open(root / "version.txt", "r").read().strip()


def get_description():
    return open("README.md", "r", encoding="utf-8").read()


setuptools.setup(
    name="vqpy",
    version=get_version(),
    description="A language for video analytics",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    package_dir={"vqpy": "vqpy"},
    packages=['vqpy'],
    python_requires=">=3.7",
)
