# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

if os.path.exists("README.md"):
    with open("README.md", "r") as f:
        long_description = f.read()
else:
    long_description = "Description could not be found."

if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        requirements = f.read().split("\n")
else:
    requirements = []

setup(
    name="snlp",
    version='0.1.3.post4',
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
    python_requires='>=3.9.9',
)