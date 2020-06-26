import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snlp",
    version="0.0.8",
    author="meghdadFar",
    author_email="meghdad.farahmand@gmail.com",
    description="Statistical NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meghdadFar/snlp",
    packages=setuptools.find_packages(),
    install_requires = ['wordcloud==1.7.0'],
    scripts=['bin/downloads.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)