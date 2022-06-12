import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, "README.md"), "r").read()

if __name__ == "__main__":
    setup(
        name="hatespace",
        version="0.0.0",
        description="Novel Archetypal Analysis NLP on hateful text corpora",
        long_description=README_FILE,
        long_description_content_type="text/markdown",
        url="https://github.com/GenerallyIntelligent/hatespace",
        author="GenerallyIntelligent",
        author_email="tannersims@generallyintelligent.me",
        license="MIT",
        packages=find_packages(),
        install_requires=[
            "torch >=1.10.0, <2.0.0",
            "geomloss",
            "html2text >=2020.0.0",
            "transformers >=4.0.0"
        ],
    )