import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easyQuake",
    version="0.7.5",
    author="Jake Walter",
    author_email="jakeiwalter@gmail.com",
    description="Simplified machine-learning driven earthquake detection, location, and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakewalter/easyQuake",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
