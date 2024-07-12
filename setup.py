# setup.py

from setuptools import setup, find_packages

setup(
    name="MRIResearchTools",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for MRI research tools",
    packages=find_packages(),
    install_requires=[
        "pydicom",
        "numpy",
        "scipy",
        "matplotlib",
    ],
    python_requires='>=3.6',
)