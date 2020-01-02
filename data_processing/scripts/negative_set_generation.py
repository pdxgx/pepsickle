#!/usr/bin/env python
'''
negative_set_generation.py

Mary Wood (mary.a.wood.91@gmail.com)


'''

from __future__ import print_function
from collections import defaultdict
from datetime import datetime
import argparse
import os
import pickle
import re
import random
import sys
import pandas as pd
import sequence_featurization_tools as sf

# Set wildcard amino acid info
wildcard_aas = ['B', 'J', 'X', 'Z']

def generate_regex(peptide):
	''' Generates a regex pattern to search ambiguous amino acids

		peptide: peptide to generate regex pattern from (string)

		Return value: regex pattern
	'''
	return peptide.replace('B', '[BDN]').replace('Z', '[EQZ]').replace('J', '[IJL]').replace('X', '[ABCDEFGHIJKLMNPQRSTUVWYXZ]').replace('*', '[*]')

def get_context_window(user_window):
	''' Gets upstream and downstream amino acid context window sizes from 
		command line

		user_window: command line parameter passed by user for amino acid 
					 context window sizes; either upstream_size,downstream_size
					 or one size for uniform window size on either side

		Return value: upstream context window size (int), downstream context
					  window size (int)
	'''
	if ',' in user_window:
		windows = user_window.split(',')
		if len(windows) == 2:
			return int(windows[0]), int(windows[1])
		else:
			raise ValueError(
			   'Only two context window sizes may be used: upstream,downstream'
			)
	else:
		return int(user_window), int(user_window)

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
					'--context-window', '-w', type=str, default="10,10",
					help='size of upstream,downstream peptide context'
	)
	parser.add_argument(
					'--trimming-window', '-t', type=int, default=24,
					help='size of window for proteasomal trimming'
	)
	parser.add_argument(
					'--c-terminal-exclusion', '-c', type=int, default=1,
					help='number of C terminal amino acids to exclude'
	)
	parser.add_argument(
					'--n-terminal-exclusion', '-n', type=int, default=1,
					help='number of N terminal amino acids to exclude'
	)
	parser.add_argument(
					'--exclude-ambiguous', '-a', required=False, action='store_true',
					help='whether to exclude ambiguous amino acids'
	)
	parser.add_argument(
					'--full-negative-set', '-f', required=False, action='store_true',
					help='whether to include all negative examples'
	)
	parser.add_argument(
					'--no-positive-n-terminus', '-p', required=False, action='store_true',
					help='do not treat N terminal cleavage site as positive if not trimmed'
	)
	args = parser.parse_args()

	print(datetime.now(), 'Loading and preprocessing data...', file=sys.stderr)

	# Set random seed
	random.seed(1206)
	
	# Get upstream/downstream context window sizes
	upstream_window, downstream_window = get_context_window(args.context_window)
	
	# Read input file and add columns for positional info
	data = pd.read_csv(os.path.abspath(args.input_file), low_memory=False)
	data['trimming_start'] = data['end_pos'] - args.trimming_window - 1
	data['n_exclusion_end'] = data['start_pos'] + args.n_terminal_exclusion + 1
	data['c_exclusion_start'] = data['end_pos'] - args.c_terminal_exclusion
	
	# Create sets to store positive, unknown and negative examples
	positives = defaultdict(set)
	unknowns = defaultdict(set)
	negatives = defaultdict(set)
	# Create sets to hold regex patterns for positives/unknowns
	positive_regex = set()
	unknown_regex = set()
	cleavage_negatives = set()

	print(datetime.now(), 'Identifying positive cases...', file=sys.stderr)

	# Iterate through epitopes to find positive/unknown cases
	for index, row in data.iterrows():
		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])

		if row['end_pos'] != len(row['full_sequence']):
			# Get positive peptide window and store regex
			peptide_window = sf.get_peptide_window(
												protein, None, row['end_pos'], 
												upstream=upstream_window, 
												downstream=downstream_window, 
			)
			wildcards = [x for x in peptide_window if x in wildcard_aas]
			if wildcards:
				if not args.exclude_ambiguous:
					positives[peptide_window].add(tuple(row[0:-3]))
					positive_regex.add(re.compile(generate_regex(peptide_window)))
			else:
				positives[peptide_window].add(tuple(row[0:-3]))

		if row['entry_source'] != 'cleavage map':
			if args.trimming_window > len(row['fragment']):
				# Store unknowns due to trimming/N-terminal exclusion to final set
				for i in range(int(row['trimming_start']), int(row['n_exclusion_end'])):
					peptide_window = sf.get_peptide_window(
													protein, None, i,
													upstream=upstream_window, 
													downstream=downstream_window,
					)
					if peptide_window is not None:
						# Adds window to unknowns if center is in valid protein space
						unknowns[peptide_window].add((tuple(row[0:-3])))
			elif row['start_pos'] > 0 and not args.no_positive_n_terminus:
				# Also store N-terminal peptide window in positives (and regex if relevant)
				peptide_window = sf.get_peptide_window(
													protein, row['start_pos'], None, 
													upstream=upstream_window, 
													downstream=downstream_window, 
													c_terminal=False
				)
				wildcards = [x for x in peptide_window if x in wildcard_aas]
				if wildcards:
					if not args.exclude_ambiguous:
						positives[peptide_window].add(tuple(row[0:-3]))
						positive_regex.add(re.compile(generate_regex(peptide_window)))
				else:
					positives[peptide_window].add(tuple(row[0:-3]))

			# Store unknowns due to C-terminal exclusion to final set
			for i in range(int(row['c_exclusion_start']), int(row['end_pos'])):
				peptide_window = sf.get_peptide_window(
												protein, None, i,
												upstream=upstream_window, 
												downstream=downstream_window,
				)
				unknowns[peptide_window].add(tuple(row[0:-3]))

		elif row['start_pos'] > 0:
			# Also store N-terminal peptide window in positives (and regex if relevant)
			peptide_window = sf.get_peptide_window(
												protein, row['start_pos'], None, 
												upstream=upstream_window, 
												downstream=downstream_window, 
												c_terminal=False
			)
			wildcards = [x for x in peptide_window if x in wildcard_aas]
			if wildcards:
				if not args.exclude_ambiguous:
					positives[peptide_window].add(tuple(row[0:-3]))
					positive_regex.add(re.compile(generate_regex(peptide_window)))
			else:
				positives[peptide_window].add(tuple(row[0:-3]))
	
	print(datetime.now(), 'Identifying negative cases from cleavage maps...', file=sys.stderr)

	# Determine cleavage map negative peptides
	cleavage_peptides = data.loc[data['entry_source'] == 'cleavage map']
	for index, row in cleavage_peptides.iterrows():
		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])
		# Store all potential negatives for epitope temporarily
		temp_negatives = set()
		for i in range(int(row['start_pos'])+1, int(row['end_pos'])):
			peptide_window = sf.get_peptide_window(
											protein, None, i+1,
											upstream=upstream_window, 
											downstream=downstream_window,
			)
			temp_negatives.add(peptide_window)
		# Process temp_negatives to find true negatives
		true_negatives = set()
		for peptide in temp_negatives:
			# Create regex if relevant
			wildcards = [x for x in peptide if x in wildcard_aas]
			if wildcards:
				if not args.exclude_ambiguous:
					r = re.compile(generate_regex(peptide))
					p_matches = [x for x in positives if r.match(x)]
					if not p_matches:
						# Peptide does not match a positive, is a true negative
						true_negatives.add(peptide)
						u_matches = [x for x in unknowns if r.match(x)]
						for pep in u_matches:
							# Remove any unknowns that match negative peptide regex
							del unknowns[pep]
			else:
				# Compare peptide to positives
				in_positives = False
				if peptide in positives:
					in_positives = True
				else: 
					for regex in positive_regex:
						if regex.match(peptide):
							in_positives = True
							break
				# Check if peptide matches unknowns if not in positives
				if not in_positives:
					true_negatives.add(peptide)
					cleavage_negatives.add(peptide)
		# Add negative(s) to negative set
		if not args.full_negative_set:
			# Add only one negative to final set
			new_negatives = [x for x in true_negatives if x not in negatives]
			if new_negatives:
				# Determine how many negatives to select
				neg_count = 0
				if row['end_pos'] != len(row['full_sequence']):
					neg_count += 1
				if row['start_pos'] != 0:
					neg_count += 1
				selected = random.sample(new_negatives, min(len(new_negatives), neg_count))
				for peptide in selected:
					negatives[peptide].add(tuple(row[0:-3]))
			# Add info for redundant negatives
			old_negatives = [x for x in true_negatives if x in negatives]
			for neg in old_negatives:
				negatives[neg].add(tuple(row[0:-3]))
		else:
			# Store all negatives
			for neg in true_negatives:
				negatives[neg].add(tuple(row[0:-3]))

	print(datetime.now(), 'Filtering unknowns...', file=sys.stderr)

	# Remove positive examples from unknowns and create unknown regexes
	for peptide in list(unknowns.keys()):
		# Check for wildcards to search against	first
		wildcards = [x for x in peptide if x in wildcard_aas]
		if wildcards:
			if not args.exclude_ambiguous:
				regex = generate_regex(peptide)
				r = re.compile(regex)
				# Compare to positives
				for pep in positives:
					if r.match(pep):
						del unknowns[peptide]
						break
				# Compare to cleavage negatives w/o wildcards
				for pep in cleavage_negatives:
					if r.match(pep):
						del unknowns[peptide]
						break
				# Create regex for peptide if still in unknowns
				if peptide in unknowns:
					unknown_regex.add(re.compile(regex))
			else:
				del unknowns[peptide]
		# Check for matches to positives and cleavage map negatives
		elif peptide in positives:
			del unknowns[peptide]
		elif peptide in cleavage_negatives:
			del unknowns[peptide]
		elif positive_regex:
			for r in positive_regex:
				if r.match(peptide):
					del unknowns[peptide]
					break

	print(datetime.now(), 'Identifying remaining negative cases...', file=sys.stderr)

	# Loop back through the data frame to get negative cases for non-cleavage map peptides
	for index, row in data.iterrows():
		# Skip cleavage map data
		if row['entry_source'] == 'cleavage map':
			continue

		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])

		# Store all potential negatives for epitope temporarily
		temp_negatives = set()
		for i in range(int(row['n_exclusion_end']), int(row['c_exclusion_start'])):
			peptide_window = sf.get_peptide_window(
											protein, None, i,
											upstream=upstream_window, 
											downstream=downstream_window,
			)
			temp_negatives.add(peptide_window)
		# Process temp negatives to find true negatives
		true_negatives = set()
		for peptide in temp_negatives:
			# Create regex if relevant
			wildcards = [x for x in peptide if x in wildcard_aas]
			if wildcards:
				if not args.exclude_ambiguous:
					r = re.compile(generate_regex(peptide))
					p_matches = [x for x in positives if r.match(x)]
					if not p_matches:
						u_matches = [x for x in unknowns if r.match(x)]
						if not u_matches:
							true_negatives.add(peptide)
			else:
				# Compare peptide to positives
				in_positives = False
				if peptide in positives:
					in_positives = True
				else: 
					for regex in positive_regex:
						if regex.match(peptide):
							in_positives = True
							break
				# Check if peptide matches unknowns if not in positives
				if not in_positives:
					in_unknowns = False
					if peptide in unknowns:
						in_unknowns = True
					else:
						for regex in unknown_regex:
							if regex.match(peptide):
								in_unknowns = True
								break
					# Only consider true negative if not in unknowns
					if not in_unknowns:
						true_negatives.add(peptide)
		# Add negative(s) to negative set
		if not args.full_negative_set:
			# Add only one negative to final set
			new_negatives = [x for x in true_negatives if x not in negatives]
			if new_negatives:
				# Determine how many negatives to select
				neg_count = 0
				if row['end_pos'] != len(row['full_sequence']):
					neg_count += 1
				if args.trimming_window > len(row['fragment']) and row['start_pos'] != 0:
					neg_count += 1
				selected = random.sample(new_negatives, min(len(new_negatives), neg_count))
				for peptide in selected:
					negatives[peptide].add(tuple(row[0:-3]))
			# Add info for redundant negatives
			old_negatives = [x for x in true_negatives if x in negatives]
			for neg in old_negatives:
				negatives[neg].add(tuple(row[0:-3]))
		else:
			# Store all negatives
			for neg in true_negatives:
				negatives[neg].add(tuple(row[0:-3]))

	# Create feature arrays for positive and negative examples
	data_set = {'positives': {}, 'negatives': {}}
	for window in positives:
		data_set['positives'][window] = positives[window]
	for window in negatives:
		data_set['negatives'][window] = negatives[window]
	# Store dataset to pickled dictionary
	with open(args.output_dict, 'wb') as p:
		pickle.dump(data_set, p)

	print(datetime.now(), 'Done!', file=sys.stderr)

