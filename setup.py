from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ColdSnap",
    version="0.0.2",
    author="Sebastian Proost",
    author_email="sebastian.proost@gmail.com",
    description="Create snapshots of machine learning models and their training data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raeslab/coldsnap/",
    project_urls={
        "Bug Tracker": "https://github.com/raeslab/coldsnap/issues",
    },
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.2.3",
        "scikit-learn>=1.5.2",
        "matplotlib>=3.9.2",
        "shap>=0.46.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
