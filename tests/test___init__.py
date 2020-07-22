#!/usr/bin/env python3
"""
test___init__.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains functions for wrapping generated proteasomal cleavage
prediction models and handling fasta protein inputs for easy model
implementation.
"""

from pepsickle import *
from inspect import getsourcefile
import unittest

pepsickle_dir = os.path.dirname(
    os.path.dirname((os.path.abspath(getsourcefile(lambda: 0))))
)


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
        print(self.seq)
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
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
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="digestion",
                                                    proteasome_type="C")
        self.assertEqual(out_df.shape, (313, 4))

    def test_immuno_digestion_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model()
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="digestion",
                                                    proteasome_type="I")
        self.assertEqual(out_df.shape, (313, 4))

    def test_epitope_model_human(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_epitope_model(human_only=True)
        self.assertIsInstance(cleavage_model, epitopeFullNet)
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="epitope")
        self.assertEqual(out_df.shape, (313, 4))

    def test_constitutive_digestion_model_human(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model(human_only=True)
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="digestion",
                                                    proteasome_type="C")
        self.assertEqual(out_df.shape, (313, 4))

    def test_immuno_digestion_model_human(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model(human_only=True)
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="digestion",
                                                    proteasome_type="I")
        self.assertEqual(out_df.shape, (313, 4))


class TestFastaProcessing(unittest.TestCase):
    """
    tests proper handling of proteasomal cleavage predictions from direct
    sequence input to the command line interface
    """

    def setUP(self):
        """

        Returns:

        """
        self.base_dir = os.path.join(pepsickle_dir, "tests")
        self.fasta = os.path.join(self.base_dir, "P43357.fasta")

    def test_epitope_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_epitope_model()
        self.assertIsInstance(cleavage_model, epitopeFullNet)
        out_df = process_fasta(self.fasta,
                               cleavage_model)
        self.assertEqual(out_df.shape, (313, 4))

    def test_constitutive_digestion_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model()
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = process_fasta(fasta_file=self.fasta,
                               cleavage_model=cleavage_model,
                               mod_type="digestion",
                               proteasome_type="C")
        self.assertEqual(out_df.shape, (313, 4))

    def test_immuno_digestion_model(self):
        """

        Returns:

        """
        self.setUP()
        cleavage_model = initialize_digestion_model()
        self.assertIsInstance(cleavage_model, digestionFullNet)
        out_df = process_fasta(fasta_file=self.fasta,
                               cleavage_model=cleavage_model,
                               mod_type="digestion",
                               proteasome_type="I")
        self.assertEqual(out_df.shape, (313, 4))


class testFileOutput(unittest.TestCase):
    def setUp(self):
        """

        Returns:

        """

    def test_CSV(self):
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
