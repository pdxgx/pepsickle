#!/usr/bin/env python3
"""
__init__.py

For issues contact Ben Weeder (weeder@ohsu.edu)

"""

from pepsickle.model_functions import *
from optparse import OptionParser


def parse_args():
    parser = OptionParser()
    parser.add_option("-s", "--sequence",
                      help="option to use pepsickle in single sequence mode. "
                           "takes a string sequence as input and returns "
                           "predicted cleavage sites in standard format")
    parser.add_option("-f", "--fasta",
                      help="fasta file with protein ID's and corresponding "
                           "sequences")
    parser.add_option("-o", "--out",
                      help="name and destination for prediction outputs. If "
                           "none is provided, the output will be printed "
                           "directly")
    parser.add_option("-v", "--verbose", action="store_true", default=False,
                      help="prints progress during cleavage predictions for "
                           "fasta files with multiple protein sequences")
    parser.add_option("-m", "--model-type", default="epitope",
                      help="allows the use of models trained on alternative "
                           "data. Defaults to epitope based model, with options "
                           "for in-vitro based random forest model (in-vitro) "
                           "or an experimental neural network based in-vitro "
                           "model (in-vitro-2)")
    parser.add_option("-p", "--proteasome-type", default="C",
                      help="allows predictions to be made based on consitutive "
                           "proteasomal or immunoproteasomal cleavage profiles. "
                           "Note that if predictions are made using the "
                           "epitope-based model (default), predictions will be "
                           "proteasome type agnostic.")
    parser.add_option("-t", "--threshold", dest="threshold", default=0.5,
                      help="probability threshold to be used for cleavage "
                           "predictions")
    parser.add_option("--human-only", action="store_true",
                      help="uses models trained on human data only instead " 
                           "of all mammals")
    (options, args) = parser.parse_args()
    return options, args


def validate_input(options):
    assert (options.fasta or options.sequence), \
        "input sequence or file required for model predictions"
    assert not (options.fasta and options.sequence), \
        "input must be either an individual sequence or a fasta file, not both"
    if options.model_type:
        assert (options.model_type in ["epitope", "in-vitro", "in-vitro-2"]), \
            "model type must be one of the following: 'epitope', 'in-vitro', " \
            "'in-vitro-2'"
    if options.proteasome_type:
        assert (options.proteasome_type in ['C', 'I']), \
            "Proteasome type must C (constitutive) or I (immuno)"


def main():
    # parse args and validate expected input
    options, args = parse_args()
    validate_input(options)

    # initialize requested model
    if options.model_type == "epitope":
        cleavage_model = initialize_epitope_model(human_only=
                                                  options.human_only)
    elif options.model_type == 'in-vitro-2':
        cleavage_model = initialize_digestion_model(human_only=
                                                    options.human_only)
    else:
        cleavage_model = initialize_digestion_gb_model(human_only=
                                                       options.human_only)
        pass

    # two if statements for fasta vs. sequence input
    if options.fasta:
        if options.model_type == 'epitope':
            out = process_fasta(options.fasta,
                                cleavage_model,
                                verbose=options.verbose,
                                threshold=options.threshold)
        elif options.model_type == 'in-vitro-2':
            out = process_fasta(options.fasta,
                                cleavage_model,
                                verbose=options.verbose,
                                mod_type='in-vitro-2',
                                proteasome_type=options.proteasome_type,
                                threshold=options.threshold)
        elif options.model_type == 'in-vitro':
            out = process_fasta(options.fasta,
                                cleavage_model,
                                verbose=options.verbose,
                                mod_type='in-vitro',
                                proteasome_type=options.proteasome_type,
                                threshold=options.threshold)

        if options.out:
            with open(options.out, "w") as f:
                for line in out:
                    f.write(line + "\n")
            f.close()
        else:
            for line in out:
                print(line)

    elif options.sequence:
        if options.model_type == 'epitope':
            out = predict_protein_cleavage_locations(protein_id="None",
                                                     protein_seq=options.sequence,
                                                     model=cleavage_model,
                                                     mod_type="epitope",
                                                     proteasome_type=options.proteasome_type,
                                                     threshold=options.threshold)

        elif options.model_type == 'in-vitro-2':
            out = predict_protein_cleavage_locations(protein_id="None",
                                                     protein_seq=options.sequence,
                                                     model=cleavage_model,
                                                     mod_type="in-vitro-2",
                                                     proteasome_type=options.proteasome_type,
                                                     threshold=options.threshold)

        elif options.model_type == 'in-vitro':
            out = predict_protein_cleavage_locations(protein_id="None",
                                                     protein_seq=options.sequence,
                                                     model=cleavage_model,
                                                     mod_type="in-vitro",
                                                     proteasome_type=options.proteasome_type,
                                                     threshold=options.threshold)

        master_lines = ["positions\tcleav_prob\tcleaved\tprotein_id"]
        for line in format_protein_cleavage_locations(out):
            master_lines.append(line)

        if options.out:
            with open(options.out, "w") as f:
                for line in master_lines:
                    f.write(line + "\n")
            f.close()
        else:
            for line in master_lines:
                print(line)


if __name__ == "__main__":
    main()
