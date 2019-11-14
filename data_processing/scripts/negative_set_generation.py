#!/usr/bin/env python
'''
negative_set_generation.py

Mary Wood (mary.a.wood.91@gmail.com)


'''

from __future__ import print_function
import os
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

	data = pd.read_csv(os.path.abspath(args.input_file))

	positives = set()
	unknowns = set()
	negatives = set()
	for index, row in data.iterrows():
		# Extract protein sequence and start/end positions
		protein = row['Source_Sequence']
		start_pos = row['Starting_Position']
		end_pos = row['Ending_Position']
		# Get peptide window
		peptide_window = sf.get_peptide_window(
											protein, start_pos, end_pos, 
											upstream=args.digestion_window, 
											downstream=args.digestion_window, 
											c_terminal=True
		)
		positives.add(peptide_window)


