#!/usr/bin/env python3
"""
test___init__.py

For issues contact Ben Weeder (weeder@ohsu.edu)

"""

from pepsickle import *
from inspect import getsourcefile
import unittest

pepsickle_dir = os.path.dirname(
    os.path.dirname((os.path.abspath(getsourcefile(lambda: 0))))
)


class TestExactModelOutput(unittest.TestCase):
    """
    tests exact output of loaded models to verify model stability
    """
    def setUP(self):
        """

        :return:
        """
        self.seq = ["MPLEQRSQHCKPE"]

    def test_epitope_model(self):
        """

        :return:
        """
        self.setUP()
        cleavage_model = initialize_epitope_model()
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_epitope_mod(cleavage_model, feature_set)
        self.assertAlmostEqual(round(pred[0], 5), round(0.18569, 5))

    def test_constitutive_digestion_model(self):
        self.setUP()
        cleavage_model = initialize_digestion_model()
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_digestion_mod(cleavage_model, feature_set,
                                     proteasome_type="C")
        self.assertAlmostEqual(round(pred[0], 5), 0.99859)

    def test_immuno_digestion_model(self):
        self.setUP()
        cleavage_model = initialize_digestion_model()
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_digestion_mod(cleavage_model, feature_set,
                                     proteasome_type="I")
        self.assertAlmostEqual(round(pred[0], 5), 0.78628)

    def test_epitope_model_human(self):
        """

        :return:
        """
        self.setUP()
        cleavage_model = initialize_epitope_model(human_only=True)
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_epitope_mod(cleavage_model, feature_set)
        self.assertAlmostEqual(round(pred[0], 5), 0.18569)

    def test_constitutive_digestion_model_human(self):
        self.setUP()
        cleavage_model = initialize_digestion_model(human_only=True)
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_digestion_mod(cleavage_model, feature_set, proteasome_type="C")
        self.assertAlmostEqual(round(pred[0], 5), 0.99778)

    def test_immuno_digestion_model_human(self):
        self.setUP()
        cleavage_model = initialize_digestion_model(human_only=True)
        feature_set = sft.generate_feature_array(self.seq)

        pred = predict_digestion_mod(cleavage_model, feature_set, proteasome_type="I")
        self.assertAlmostEqual(round(pred[0], 5), 0.00309)


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
        out_df = predict_protein_cleavage_locations(protein_id="None",
                                                    protein_seq=self.seq,
                                                    model=cleavage_model,
                                                    mod_type="epitope",
                                                    proteasome_type="E")
        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)

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

        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)

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
        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)

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
        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)

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
        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)

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
        for i, entry in enumerate(out_df):
            self.assertEqual(len(entry), 4)
        self.assertEqual(i, 312)


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
        self.assertEqual(len(out_df), 314)

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
        self.assertEqual(len(out_df), 314)

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
        self.assertEqual(len(out_df), 314)


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
