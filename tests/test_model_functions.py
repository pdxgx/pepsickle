#!/usr/bin/env python3
"""
test_model_functions.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains functions for wrapping generated proteasomal cleavage
prediction models and handling fasta protein inputs for easy model
implementation.
"""

from pepsickle import *
import pepsickle.sequence_featurization_tools as sft
from pepsickle.model_functions import *
import unittest
from inspect import getsourcefile
import os.path as path, sys
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
        self.seq = "MPLEQRSQHCKPEEGLEARGEALGLVGAQAPATEEQEAASSSSTLVEVTLGEVPAAE" \
                   "SPDPPQSPQGASSLPTTMNYPLWSQSYEDSSNQEEEGPSTFPDLESEFQAALSRKVA" \
                   "ELVHFLLLKYRAREPVTKAEMLGSVVGNWQYFFPVIFSKASSSLQLVFGIELMEVDP" \
                   "IGHLYIFATCLGLSYDGLLGDNQIMPKAGLLIIVLAIIAREGDCAPEEKIWEELSVL" \
                   "EVFEGREDSILGDPKKLLTQHFVQENYLEYRQVPGSDPACYEFLWGPRALVETSYVK" \
                   "VLHHMVKISGGPHISYPPLHEWVLREGEE"

    def test_epitope_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_epitope_model()
        self.assertIsInstance(cleavage_model, epitopeFullNet)
        out_df = predict_protein_cleavage_locations("None",
                                                    self.seq,
                                                    cleavage_model,
                                                    mod_type="epitope",
                                                    proteasome_type="E")
        self.assertEqual(out_df.shape, (313, 4))

    def test_constitutive_digestion_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model()
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = predict_protein_cleavage_locations("None",
                                                    self.seq,
                                                    cleavage_model,
                                                    mod_type="digestion",
                                                    proteasome_type="C")
        self.assertEqual(out_df.shape, (313, 4))

    def testImmunoDigestionModel(self):
        """

        Returns:

        """
    def testEpitopeModelHuman(self):
        """

        Returns:

        """
    def testConstitutiveDigestionModelHuman(self):
        """

        Returns:

        """
    def testImmunoDigestionModelHuman(self):
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


if __name__ == "__main__":
    unittest.main()
