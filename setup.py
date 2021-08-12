from setuptools import setup, find_packages, Command
import os

setup(
    name="pythonparser",
    version="1.4",
    author="whitequark",
    author_email="whitequark@whitequark.org",
    url="https://github.com/m-labs/pythonparser",
    description="A Python parser intended for use in tooling",
    long_description=open("README.md").read(),
    license="MIT",
    install_requires=["regex"],
    extras_require={},
    dependency_links=[],
    packages=find_packages(exclude=["tests*"]),
    namespace_packages=[],
    test_suite="pythonparser.test",
    package_data={},
    ext_modules=[],
    entry_points={},
)
