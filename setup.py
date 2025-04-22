from setuptools import setup, find_packages

setup(
    name="facefusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "opencv-python>=4.10.0.84",
        "onnx>=1.17.0",
        "onnxruntime>=1.20.1",
        "scipy>=1.14.1",
        "tqdm>=4.67.1",
        "psutil>=6.1.1",
        "pydantic>=2.10.6",
    ],
    python_requires=">=3.8",
)
