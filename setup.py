"""
automated-mlsecops  –  package setup
"""

from setuptools import setup, find_packages

setup(
    name="automated-mlsecops",
    version="1.0.0",
    description="Automated MLSecOps pipeline with DEA evaluation",
    packages=find_packages(include=["src", "src.*", "api", "api.*", "experiments", "experiments.*"]),
    python_requires=">=3.10",
    install_requires=[],   # see requirements.txt
)
