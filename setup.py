# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages

if os.path.exists("requirements.txt"):
    with open("requirements.txt") as fh:
        requirements = [r for r in fh.read().split("\n") if ";" not in r]
else:
    requirements = []

if os.path.exists("requirements.txt"):
    with open("README.md", "r") as fh:
        long_description = fh.read()
else:
    long_description = "Description could not be found."

setup(
    name="snlp",
    version='0.1.0',
    author="meghdadFar",
    author_email="meghdad.farahmand@gmail.com",
    description="Statistical NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meghdadFar/snlp",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    scripts=['bin/downloads.py'],
    python_requires='>=3.7',
)