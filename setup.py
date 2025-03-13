from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    packages = []
    with open(file_path, "r+") as packs:
        packages = packs.readlines()
        packages = [req.replace("\n", "") for req in packages]
    
        if HYPEN_E_DOT in packages:
            packages.remove(HYPEN_E_DOT)
    return packages

setup(
    name = "mlproject",
    version = "0.0.1",
    author = "MK",
    author_email = "sakthikaliappan7797@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)