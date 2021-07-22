#!/usr/bin/env python3
# coding=utf-8

from setuptools import setup, find_packages
from distutils.core import Command
from setuptools.command.install import install

# Borrowed (with revisions) from https://stackoverflow.com/questions/17001010/
# how-to-run-unittest-discover-from-python-setup-py-test/21726329#21726329


class DiscoverTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import os
        import sys
        import unittest

        # get setup.py directory
        setup_file = sys.modules["__main__"].__file__
        test_dir = os.path.join(os.path.abspath(os.path.dirname(setup_file)), "tests")
        # use the default shared TestLoader instance
        test_loader = unittest.defaultTestLoader
        # use the basic test runner that outputs to sys.stderr
        test_runner = unittest.TextTestRunner()
        # automatically discover all tests
        # NOTE: only works for python 2.7 and later
        test_suite = test_loader.discover(test_dir)
        print(test_suite)
        # run the test suite
        test_runner.run(test_suite)


class DownloadDependencies(Command):
    # Wrapper to accommodate old-style class Command
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


setup(
    name="pepsickle",
    version="0.1.0",
    description="proteasomal cleavage prediction tool",
    long_description=(
        ""),
    url="https://github.com/pdxgx/pepsickle/archive/refs/tags/v0.1.0.tar.gz",
    download_url="https://github.com/pdxgx/pepsickle.git",
    author="",
    author_email="weederb@ohsu.edu",
    license="MIT",
    packages=["pepsickle"],
    include_package_data=True,
    package_data={"pepsickle": ["*.py", "*.pickle", "*.joblib"]},
    zip_safe=False,
    install_requires=["biopython>=1.77", "numpy>=1.16.5", "torch>=1.3.1",
                      "joblib>=0.11", "scikit-learn>=0.23.2"],
    entry_points={"console_scripts": ["pepsickle=pepsickle:main"]},
    cmdclass={"download": DownloadDependencies, "test": DiscoverTest},
    keywords=["proteasome", "cleavage", "peptide", "degredation"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
