from setuptools import setup, find_packages
import os

setup(
    name="artiq",
    version="0.0+dev",
    author="whitequark",
    author_email="whitequark@whitequark.org",
    url="https://m-labs.github.io/pyparser",
    description="A Python parser intended for use in tooling",
    long_description=open("README.rst").read(),
    license="BSD",
    install_requires=[],
    extras_require={},
    dependency_links=[],
    packages=find_packages(exclude=['tests*']),
    namespace_packages=[],
    test_suite="pyparser.test",
    package_data={},
    ext_modules=[],
    entry_points={}
)
