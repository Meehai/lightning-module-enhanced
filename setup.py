from setuptools import setup, find_packages
from os import path

def get_reqs(requirements):
    # Do not add to required lines pointing to Git repositories
    required, dependency_links = [], []
    EGG_MARK = "#egg="
    for line in requirements:
        if line.startswith("-e git:") or line.startswith("-e git+") or \
                line.startswith("git:") or line.startswith("git+"):
            line = line.lstrip("-e ")  # in case that is using "-e"
            if EGG_MARK in line:
                package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
                repository = line[:line.find(EGG_MARK)]
                required.append(f"{package_name} @ {repository}")
                dependency_links.append(line)
            else:
                print("Dependency to a git repository should have the format:")
                print("git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name")
        else:
            required.append(line)
    return required, dependency_links

name = "lightning-module-enhanced"
version = "0.21"
description = "Lightning Module Enhanced"
url = "https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced"

loc = path.abspath(path.dirname(__file__))
with open(f"{loc}/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open(f"{loc}/requirements.txt") as f:
        requirements = f.read().splitlines()
    required, dependency_links = get_reqs(requirements)
except:
    required, dependency_links = [], []

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    packages=find_packages(),
    install_requires=required,
    dependency_links=dependency_links,
    license="WTFPL",
    python_requires=">=3.8"
)
