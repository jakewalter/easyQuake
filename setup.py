import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easyQuake",
    version="1.4.0",
    author="Jake Walter",
    author_email="jakeiwalter@gmail.com",
    description="Simplified machine-learning driven earthquake detection, location, and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakewalter/easyQuake",
    packages=setuptools.find_packages(),
    install_requires=[
        'obspy',
        'pandas',
        'tensorflow>=2.12',  # Use modern TensorFlow
        'tqdm',
        'protobuf>=3.20',
        'torch',
        'torchmetrics',
        'torchvision',
    ],
    tests_require=[
        'pytest',
    ],
    test_suite='pytest',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'create_new_project=ilifu_user_management.create_new_project:main',
            'gpd_predict=easyQuake.gpd_predict.gpd_predict:main',
            'mseed_predictor=easyQuake.EQTransformer.mseed_predictor:main',
            'phasenet_predict=easyQuake.phasenet.phasenet_predict:main',
            'run_seisbench=easyQuake.seisbench.run_seisbench:main',
        ]
    },
)
