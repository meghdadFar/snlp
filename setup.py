import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snlp", # Replace with your own username
    version="0.0.2",
    author="meghdadFar",
    author_email="meghdad.farahmand@gmail.com",
    description="Statistical NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meghdadFar/snlp",
    packages=setuptools.find_packages(),
    install_requires = ['pandas==1.0.3', 'scikit-learn==0.22.2.post1', 
    'matplotlib==3.2.1', 'scipy==1.4.1', 'torch==1.4.0', 'torchtext==0.5.0',
    'nltk==3.5', 'tqdm==4.45.0', 'fasttext==0.9.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)