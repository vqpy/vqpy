from pathlib import Path
import setuptools
from setuptools import find_packages


def get_version() -> str:
    root = Path(__file__).parent
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
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["cython",
                      "torch",
                      "torchvision",
                      "loguru",
                      "tqdm",
                      "onnxruntime",
                      "shapely",
                      "scipy",
                      "scikit-learn==1.1.2",
                      "numpy<1.24.0",
                      "requests",
                      "opencv-python",
                      ],
)
