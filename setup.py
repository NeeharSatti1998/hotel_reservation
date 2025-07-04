from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup (
    name = "hotel_reservation",
    version=0.1,
    author="Neehar",
    packages=find_packages(),
    install_requires= requirements
)