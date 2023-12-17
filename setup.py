from setuptools import find_packages,setup
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "LandCoverAI"
AUTHOR_USER_NAME = "debasishdj"
SRC_REPO = "LCover"
AUTHOR_EMAIL = "debasish.jyotishi07@gmail.com"


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirement
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=[req.strip() for req in file_obj.readlines()]

        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A python package for semantic segmentation of Land Cover",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements('requirements.txt')
)