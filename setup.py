from setuptools import setup, find_packages
from os import path

name = "lightning-module-enhanced"
version = "0.25"
description = "Lightning Module Enhanced"
url = "https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced"

loc = path.abspath(path.dirname(__file__))
with open(f"{loc}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

required = ["pytorch_lightning==2.2.0.post0", "torch>=2.1.1", "torchinfo>=1.6.5", "torchmetrics>=0.11.4", "overrides>=6.1.0",
            "matplotlib>=3.5.1", "pandas>=2.0.0", "colorama>=0.4.6", "tensorboardX>=2.5.1", "pool-resources>=0.2.3"]

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    packages=find_packages(),
    install_requires=required,
    dependency_links=[],
    license="WTFPL",
    python_requires=">=3.8"
)
