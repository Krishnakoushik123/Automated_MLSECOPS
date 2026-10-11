"""
automated-mlsecops  –  package setup
"""

from pathlib import Path
from setuptools import setup, find_packages


def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.is_file():
        return []
    lines = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # strip inline comments if any
            pkg = line.split("#")[0].strip()
            if pkg:
                lines.append(pkg)
    return lines


setup(
    name="automated-mlsecops",
    version="1.0.0",
    description="Automated MLSecOps pipeline with DEA evaluation",
    packages=find_packages(include=["src", "src.*", "api", "api.*", "experiments", "experiments.*"]),
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
