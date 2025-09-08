from pathlib import Path
from setuptools import setup, find_packages

# WARNING: This package prefers GPU usage for ML components but falls back to CPU.
# CI tests are CPU-only; for GPU support, install GPU-enabled TensorFlow/PyTorch.

# Readme for long description
HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="easyQuake",
    version="2.0.0",
    author="Jake Walter",
    author_email="jakeiwalter@gmail.com",
    description="Simplified machine-learning driven earthquake detection, location, and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakewalter/easyQuake",
    packages=find_packages(exclude=("tests", "docs", "examples", "build", "*.egg-info")),
    # Make the default install include the full ML stack so `pip install .`
    # matches the user's typical environment. Provide a `lite` extra for
    # users who want a minimal install instead.
    install_requires=[
        # Core runtime + full ML stack (default)
        "obspy>=1.3",
        "pandas>=1.5",
        "tqdm",
        # TensorFlow (flexible lower bound)
        "tensorflow>=2.12",
        # TensorFlow and recent protobufs can be incompatible with protobuf 4.x; keep an upper bound
        "protobuf>=3.20,<4",
        # PyTorch stack (platform-specific). Users on systems with special CUDA
        # requirements may prefer to install PyTorch via conda, but including
        # a baseline here makes the default install conveniently complete.
        "torch>=1.13",
        "torchvision",
        "torchmetrics",
    ],
    extras_require={
        # Minimal/lite install: core functionality without heavy ML frameworks
        "lite": ["obspy>=1.3", "pandas>=1.5", "tqdm"],
        # Individual options (for users who prefer to install only one ML framework)
        "tf": ["tensorflow>=2.12"],
        "torch": ["torch>=1.13", "torchvision", "torchmetrics"],
        # Convenience combined extras
        "ml": ["tensorflow>=2.12", "torch>=1.13", "torchvision", "torchmetrics"],
        # Development and test extras
        "test": ["pytest"],
        "dev": ["pytest", "black", "flake8"],
        "full": ["tensorflow>=2.12", "torch>=1.13", "torchvision", "torchmetrics", "pytest", "black", "flake8"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            # removed `create_new_project` which referenced an external package not bundled here
            "gpd_predict=easyQuake.gpd_predict.gpd_predict:main",
            "mseed_predictor=easyQuake.EQTransformer.mseed_predictor:main",
            "phasenet_predict=easyQuake.phasenet.phasenet_predict:main",
            "run_seisbench=easyQuake.seisbench.run_seisbench:main",
            # Provide convenient aliases for the seisbench runner
            "seisbench_predict=easyQuake.seisbench.run_seisbench:main",
            "seisbench_run=easyQuake.seisbench.run_seisbench:main",
        ]
    },
    license="MIT",
    project_urls={
        "Source": "https://github.com/jakewalter/easyQuake",
        "Documentation": "https://github.com/jakewalter/easyQuake/blob/master/README.md",
        "Tracker": "https://github.com/jakewalter/easyQuake/issues",
    },
    keywords="seismology earthquake detection phasenet eqtransformer",
    zip_safe=False,
)
