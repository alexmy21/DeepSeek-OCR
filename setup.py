import os
from setuptools import setup, find_packages

setup(
    name="deepseek-ocr-hllset",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "numpy<2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "gputil>=1.4.0",
        "psutil>=5.9.0",
    ],
)
