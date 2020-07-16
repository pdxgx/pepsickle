#!/usr/bin/env python3
"""
test_pepsickle.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains functions for wrapping generated proteasomal cleavage
prediction models and handling fasta protein inputs for easy model
implementation.
"""

from pepsickle import *

import unittest
import os


class TestSequenceProcessing(unittest.TestCase):
    """
    tests proper handling of proteasomal cleavage predictions from direct
    sequence input to the command line interface
    """

    def setUP(self):
        """

        Returns:

        """

    def testEpitopeModel(self):
        """

        Returns:

        """
    def testConstitutiveDigestionModel(self):
        """

        Returns:

        """
    def testImmunoDigestionModel(self):
        """

        Returns:

        """


class TestFastaProcessing(unittest.TestCase):
    """
    tests proper handling of proteasomal cleavage predictions from direct
    sequence input to the command line interface
    """

    def setUP(self):
        """

        Returns:

        """

    def testEpitopeModel(self):
        """

        Returns:

        """
    def testConstitutiveDigestionModel(self):
        """

        Returns:

        """
    def testImmunopDigestionModel(self):
        """

        Returns:

        """


class testFileOutput(unittest.TestCase):
    def setUp(self):
        """

        Returns:

        """

    def testCSV(self):
        """
        tests if output option gives correct CSV file and format
        Returns:

        """

class testThreshodling(unittest.TestCase):
    def setUp(self):
        """

        Returns:

        """
    def testThresholdRange(self):
        """

        Returns:

        """
