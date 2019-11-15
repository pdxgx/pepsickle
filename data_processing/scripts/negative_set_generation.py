#!/usr/bin/env python
'''
negative_set_generation.py

Mary Wood (mary.a.wood.91@gmail.com)


'''

from __future__ import print_function
import argparse
import os
import pickle
import random
import pandas as pd
import sequence_featurization_tools as sf

if __name__ == '__main__':

	# Set up command line parameters
	parser = argparse.ArgumentParser()
	parser.add_argument(
					'--input-file', '-i', type=str, required=True,
					help='path to input file containing epitope data'
	)
	parser.add_argument(
					'--output-dict', '-o', type=str, required=True,
					help='path to output file for storing pickled dictionary'
	)
	parser.add_argument(
					'--context-window', '-w', type=int, default=10,
					help='size of upstream/downstream peptide context'
	)
	parser.add_argument(
					'--digestion-window', '-d', type=int, default=24,
					help='size of window for proteasomal digestion'
	)
	parser.add_argument(
					'--c-terminal-exclusion', '-c', type=int, default=2,
					help='number of C terminal amino acids to exclude'
	)
	parser.add_argument(
					'--n-terminal-exclusion', '-n', type=int, default=2,
					help='number of N terminal amino acids to exclude'
)
	parser.add_argument(
					'--exclude-unknowns', '-u', type=bool, default=False,
					help='whether to exclude unknown amino acids'
	)
	args = parser.parse_args()

	# Set random seed
	random.seed(1206)

	# Read input file
	data = pd.read_csv(os.path.abspath(args.input_file))

	# Create sets to store positive, unknown and negative examples
	positives = set()
	unknowns = set()
	negatives = set()

	# Iterate through epitopes to find positive/unknown/negative cases
	for index, row in data.iterrows():

		# Extract protein sequence and start/end positions
		protein = row['Source_Sequence'].strip('"')
		start_pos = row['Starting_Position']
		end_pos = row['Ending_Position']
		
		# Get positive peptide window
		peptide_window = sf.get_peptide_window(
											protein, None, end_pos, 
											upstream=args.context_window, 
											downstream=args.context_window, 
		)
		positives.add(peptide_window)

		# Store unknowns due to digestion/N-terminal exclusion to final set
		digestion_start = start_pos - args.digestion_window
		n_exclusion_end = start_pos + args.n_terminal_exclusion
		for i in range(digestion_start, n_exclusion_end):
			peptide_window = sf.get_peptide_window(
											protein, None, i,
											upstream=args.context_window, 
											downstream=args.context_window,
			)
			unknowns.add(peptide_window)

		# Store unknowns due to C-terminal exclusion to final set
		c_exclusion_start = end_pos - args.c_terminal_exclusion
		for i in range(c_exclusion_start, end_pos):
			peptide_window = sf.get_peptide_window(
											protein, None, i,
											upstream=args.context_window, 
											downstream=args.context_window,
			)
			unknowns.add(peptide_window)

		# Store all negatives for epitope temporarily
		temp_negatives = set()
		for i in range(n_exclusion_end, c_exclusion_start):
			peptide_window = sf.get_peptide_window(
											protein, None, i,
											upstream=args.context_window, 
											downstream=args.context_window,
			)
			temp_negatives.add(peptide_window)
		# Add only one negative to final set
		new_negatives = [x for x in temp_negatives if x not in negatives]
		if new_negatives:
			negatives.add(random.choice(new_negatives))


	# Remove positive examples from unknowns + positives/unknowns from negatives
	unknown_examples = unknowns - positives
	negative_examples = negatives - unknowns - positives

	# Create feature arrays for positive and negative examples
	data_set = {'positives': {}, 'negatives': {}}
	for window in positives:
		matrix = sf.generate_feature_array(window)
		data_set['positives'][window] = matrix
	for window in negatives:
		matrix = sf.generate_feature_array(window)
		data_set['negatives'][window] = matrix

	# Store dataset to pickled dictionary
	with open(args.output_dict, 'wb') as p:
		pickle.dump(data_set, p)


