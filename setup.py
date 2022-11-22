from setuptools import find_packages, setup

import os
import re


# Recommendations from https://packaging.python.org/
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def read(*parts):
    with open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="mace_jax",
    version=find_version("mace_jax", "__version__.py"),
    description="Machine learning model for interatomic potentials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilyes319/mace-jax",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "torch",
        "e3nn_jax",
        "numpy",
        "ase",
        "prettytable",
        "roundmantissa",
        "jraph",
        "gin-config",
        "unique_names_generator",
        # for plotting:
        "matplotlib",
        "pandas",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    license="ASL",
    license_files=["ASL.md"],
)
