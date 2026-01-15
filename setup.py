from setuptools import setup, find_packages

setup(
    name="cat-tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.2",
        "opencv-python>=4.6.0",
        "onnxruntime>=1.14.0",
        "scipy>=1.10.0",
        "filterpy>=1.4.5",
        "matplotlib",
    ],
    python_requires=">=3.7",
    description="Multi-cat tracking system with color-based identification",
    author="Patrick",
)
