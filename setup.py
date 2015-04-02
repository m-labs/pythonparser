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
        os.system('rsync -avz doc/_build/html/ shell.serverraum.org:~/web/m-labs.hk/pyparser')

setup(
    name="pyparser",
    version="0.0+dev",
    author="whitequark",
    author_email="whitequark@whitequark.org",
    url="http://m-labs.hk/pyparser",
    description="A Python parser intended for use in tooling",
    long_description=open("README.rst").read(),
    license="BSD",
    install_requires=['regex'],
    extras_require={},
    dependency_links=[],
    packages=find_packages(exclude=['tests*']),
    namespace_packages=[],
    test_suite="pyparser.test",
    package_data={},
    ext_modules=[],
    entry_points={},
    cmdclass={'push_doc':PushDocCommand}
)
