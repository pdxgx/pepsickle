#!/usr/bin/env python3
"""
__init__.py

For issues contact Ben Weeder (weeder@ohsu.edu)

"""

from pepsickle.model_functions import *
from optparse import OptionParser


def parse_args():
    parser = OptionParser()
    parser.add_option("-s", "--sequence", dest="input_seq",
                      help="option to use pepsickle in single sequence mode. "
                           "takes a string sequence as input and returns "
                           "predicted cleavage sites in standard format")
    parser.add_option("-f", "--fasta", dest="fasta",
                      help="fasta file with protein ID's and corresponding "
                           "sequences")
    parser.add_option("-o", "--out_file", dest="out_file",
                      help="name and destination for prediction outputs. If "
                           "none is provided, the output will be printed "
                           "directly")
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="prints progress during cleavage predictions")
    parser.add_option("-m", "--model", default="E", dest="model_type",
                      help="allows the use of models trained on alternative "
                           "data. Defaults to epitope, with options for "
                           "C (constitutive proteasome) and I"
                           "(immunoproteasome)")
    parser.add_option("-t", "--threshold", dest="threshold", default=0.5,
                      help="probability threshold to be used for cleavage "
                           "predictions")
    parser.add_option("--human_only", action="store_true", default=False,
                      help="uses models trained on human data only instead " 
                           "of all mammals")
    (options, args) = parser.parse_args()
    return options, args


def validate_input(options):
    assert (options.fasta or options.input_seq), \
        "input sequence or file required for model predictions"
    assert not (options.fasta and options.input_seq), \
        "input must be either an individual sequence or a fasta file, not both"
    if options.model_type:
        assert (options.model_type in ['E', 'C', 'I']), \
            "model type must be one of the following: 'E', 'C', 'I'"


def main():
    # parse args and validate expected input
    options, args = parse_args()
    validate_input(options)

    # initialize requested model
    if options.model_type == "E":
        cleavage_model = initialize_epitope_model(human_only=
                                                  options.human_only)
    elif options.model_type in ['C', 'I']:
        cleavage_model = initialize_digestion_model(human_only=
                                                    options.human_only)

    # two if statements for fasta vs. sequence input
    if options.fasta:
        if isinstance(cleavage_model, epitopeFullNet):
            out_df = process_fasta(options.fasta,
                                   cleavage_model,
                                   verbose=options.verbose,
                                   threshold=options.threshold)
        elif isinstance(cleavage_model, digestionFullNet):
            out_df = process_fasta(options.fasta,
                                   cleavage_model,
                                   verbose=options.verbose,
                                   mod_type="digestion",
                                   proteasome_type=options.model_type,
                                   threshold=options.threshold)
    elif options.input_seq:
        if isinstance(cleavage_model, epitopeFullNet):
            out_df = predict_protein_cleavage_locations("None",
                                                        options.input_seq,
                                                        cleavage_model,
                                                        mod_type="epitope",
                                                        proteasome_type=options.model_type,
                                                        threshold=options.threshold)
        elif isinstance(cleavage_model, digestionFullNet):
            out_df = predict_protein_cleavage_locations("None",
                                                        options.input_seq,
                                                        cleavage_model,
                                                        mod_type="digestion",
                                                        proteasome_type=options.model_type,
                                                        threshold=options.threshold)

    if options.out_file:
        out_df.to_csv(options.out_file, index=False)
    else:
        print(out_df)


if __name__ == "__main__":
    main()
