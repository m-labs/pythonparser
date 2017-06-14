from setuptools import setup, find_packages, Command
import os

class PushDocCommand(Command):
    description = "uploads the documentation to m-labs.hk"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system("rsync -avz doc/_build/html/ shell.serverraum.org:~/web/m-labs.hk/pythonparser")

setup(
    name="pythonparser",
    version="1.1",
    author="whitequark",
    author_email="whitequark@whitequark.org",
    url="https://m-labs.hk/pythonparser",
    download_url="https://github.com/m-labs/pythonparser",
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
    cmdclass={"push_doc":PushDocCommand}
)
