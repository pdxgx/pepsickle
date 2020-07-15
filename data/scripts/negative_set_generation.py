#!/usr/bin/env python
'''
negative_set_generation.py

Mary Wood (mary.a.wood.91@gmail.com)


'''

from __future__ import print_function
from collections import defaultdict
from datetime import datetime
import argparse
import copy
import os
import pickle
import re
import random
import sys
import numpy as np
import pandas as pd
import sequence_featurization_tools as sf

# Set wildcard amino acid info
wildcard_aas = ['B', 'J', 'X', 'Z']

def generate_regex(peptide):
	''' Generates a regex pattern to search ambiguous amino acids

		peptide: peptide to generate regex pattern from (string)

		Return value: regex pattern
	'''
	return peptide.replace(
		'B', '[BDN]').replace(
		'Z', '[EQZ]').replace(
		'J', '[IJL]').replace(
		'X', '[ABCDEFGHIJKLMNPQRSTUVWYXZ]').replace(
		'*', '[*]'
	)

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

def remove_positives(temp_negatives, positives, positive_regex, negatives, row,
					 full_negative=True):
	''' Filters out peptides in a list of potential negatives that match 
		positive peptides

		temp_negatives: set of potential negative peptides
		positives: set of positive peptides
		positive_regex: set of compiled regular expressions for positive 
						peptides with wildcard AAs
		negatives: pre-existing negative peptides (dict)
		row: row of dataframe describing the epitope
		full_negative: whether to retain all negatives (bool)

		Return values: set of peptides that don't match any positives
	'''
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
			selected = random.sample(
							new_negatives, min(len(new_negatives), neg_count)
						)
			# Add info for redundant negatives
			selected.extend([x for x in true_negatives if x in negatives])
			return selected
		else:
			return []
	else:
		# Store all negatives
		return list(true_negatives)

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
					'--context-window', '-w', type=str, default="6,6",
					help='size of upstream,downstream peptide context'
	)
	parser.add_argument(
					'--trimming-window', '-t', type=int, default=16,
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
					'--exclude-ambiguous', '-a', required=False, 
					action='store_true',
					help='whether to exclude ambiguous amino acids'
	)
	parser.add_argument(
					'--full-negative-set', '-f', required=False, 
					action='store_true',
					help='whether to include all negative examples'
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
	
	# Create dicts to store positive, unknown and negative examples
	epitope_positives = defaultdict(set)
	epitope_unknowns = defaultdict(set)
	epitope_negatives = defaultdict(set)
	proteasome_positives = defaultdict(set)
	proteasome_negatives = defaultdict(set)
	immunoproteasome_positives = defaultdict(set)
	immunoproteasome_negatives = defaultdict(set)
	mixed_positives = defaultdict(set)
	mixed_negatives = defaultdict(set)
	# Create sets to hold regex patterns for positives/unknowns
	epitope_positive_regex = set()
	epitope_unknown_regex = set()
	proteasome_positive_regex = set()
	immunoproteasome_positive_regex = set()
	mixed_positive_regex = set()

	print(datetime.now(), 'Identifying positive cases...', file=sys.stderr)

	# Iterate through epitopes to find positive/unknown cases
	for index, row in data.iterrows():
		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])
		try:
			exclusions = row['exclusions'].split(';')
		except:
			exclusions = []
		if row['end_pos'] != len(row['full_sequence']):
			# Get positive peptide window and store regex if relevant
			peptide_window = sf.get_peptide_window(
												protein, None, row['end_pos'], 
												upstream=upstream_window, 
												downstream=downstream_window, 
			)
			wildcards = [x for x in peptide_window if x in wildcard_aas]
			if row['entry_source'] != 'cleavage_map':
				epitope_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					epitope_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
			elif row['Proteasome'] == 'C' and str(row['end_pos']-1) not in exclusions:
				proteasome_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					proteasome_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
			elif row['Proteasome'] == 'I' and str(row['end_pos']-1) not in exclusions:
				immunoproteasome_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					immunoproteasome_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
			elif row['Proteasome'] == 'M' and str(row['end_pos']-1) not in exclusions:
				mixed_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					mixed_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
		if row['entry_source'] != 'cleavage_map':
			# Store unknowns for non-cleavage map data
			if args.trimming_window > len(row['fragment']):
				# Store unknowns due to trimming/N-terminal exclusion
				for i in range(
								int(row['trimming_start']), 
								int(row['n_exclusion_end'])
							  ):
					peptide_window = sf.get_peptide_window(
												protein, None, i,
												upstream=upstream_window, 
												downstream=downstream_window,
					)
					if peptide_window is not None:
						# Adds window to unknowns if valid
						epitope_unknowns[peptide_window].add((tuple(row[0:-3])))

			# Store unknowns due to C-terminal exclusion to final set
			for i in range(int(row['c_exclusion_start']), int(row['end_pos'])):
				peptide_window = sf.get_peptide_window(
												protein, None, i,
												upstream=upstream_window, 
												downstream=downstream_window,
				)
				epitope_unknowns[peptide_window].add(tuple(row[0:-3]))
		elif row['start_pos'] > 0 and str(row['start_pos']) not in exclusions:
			# Also store N-terminal peptide window in positives
			peptide_window = sf.get_peptide_window(
											protein, row['start_pos'], None, 
											upstream=upstream_window, 
											downstream=downstream_window, 
											c_terminal=False
			)
			wildcards = [x for x in peptide_window if x in wildcard_aas]
			if row['Proteasome'] == 'C':
				proteasome_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					proteasome_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
			elif row['Proteasome'] == 'I':
				immunoproteasome_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					immunoproteasome_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
			elif row['Proteasome'] == 'M':
				mixed_positives[peptide_window].add(tuple(row[0:-3]))
				if wildcards:
					mixed_positive_regex.add(
									re.compile(generate_regex(peptide_window))
					)
	
	print(
			datetime.now(), 'Identifying negative cases from cleavage maps...', 
			file=sys.stderr
	)

	# Determine cleavage map negative peptides
	cleavage_peptides = data.loc[data['entry_source'] == 'cleavage_map']
	# Compile cleavage positives for filtering potential mixed negatives
	all_cleavage_positives = copy.copy(proteasome_positives)
	for peptide in immunoproteasome_positives:
		all_cleavage_positives[peptide].update(immunoproteasome_positives[peptide])
	for peptide in mixed_positives:
		all_cleavage_positives[peptide].update(mixed_positives[peptide])
	all_cleavage_regex = proteasome_positive_regex.copy()
	all_cleavage_regex.update(immunoproteasome_positive_regex)
	all_cleavage_regex.update(mixed_positive_regex)
	for index, row in cleavage_peptides.iterrows():
		try:
			exclusions = row['exclusions'].split(';')
		except:
			exclusions = []
		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])
		# Store all potential negatives for epitope temporarily
		temp_negatives = set()
		for i in range(int(row['start_pos']), int(row['end_pos'])):
			if str(i) not in exclusions:
				peptide_window = sf.get_peptide_window(
												protein, None, i+1,
												upstream=upstream_window, 
												downstream=downstream_window,
				)
				temp_negatives.add(peptide_window)
		# Process temp_negatives to find true negatives
		if row['Proteasome'] == 'C':
			true_negatives = remove_positives(
											temp_negatives, 
											proteasome_positives, 
											proteasome_positive_regex,
											proteasome_negatives, row,
											args.full_negative_set
			)
			for neg in true_negatives:
				proteasome_negatives[neg].add(tuple(row[0:-3]))
		elif row['Proteasome'] == 'I':
			true_negatives = remove_positives(
										temp_negatives, 
										immunoproteasome_positives, 
										immunoproteasome_positive_regex,
										immunoproteasome_negatives, row,
										args.full_negative_set
			)
			for neg in true_negatives:
				immunoproteasome_negatives[neg].add(tuple(row[0:-3]))
		elif row['Proteasome'] == 'M':
			true_negatives = remove_positives(
										temp_negatives, 
										all_cleavage_positives, 
										all_cleavage_regex,
										mixed_negatives, row,
										args.full_negative_set
			)
			for neg in true_negatives:
				mixed_negatives[neg].add(tuple(row[0:-3]))

	print(datetime.now(), 'Filtering unknowns...', file=sys.stderr)

	# Remove positive examples from unknowns and create unknown regexes
	for peptide in list(epitope_unknowns.keys()):
		# Check for wildcards to search against	first
		wildcards = [x for x in peptide if x in wildcard_aas]
		if wildcards:
			if not args.exclude_ambiguous:
				regex = generate_regex(peptide)
				r = re.compile(regex)
				# Compare to positives
				for pep in epitope_positives:
					if r.match(pep):
						del epitope_unknowns[peptide]
						break
				# Create regex for peptide if still in unknowns
				if peptide in epitope_unknowns:
					epitope_unknown_regex.add(re.compile(regex))
			else:
				del epitope_unknowns[peptide]
		# Check for matches to positives and cleavage map negatives
		elif peptide in epitope_positives:
			del epitope_unknowns[peptide]
		elif epitope_positive_regex:
			for r in epitope_positive_regex:
				if r.match(peptide):
					del epitope_unknowns[peptide]
					break

	print(
			datetime.now(), 'Identifying remaining negative cases...', 
			file=sys.stderr
	)

	# Get negative cases for non-cleavage map peptides
	for index, row in data.iterrows():
		# Skip cleavage map data
		if row['entry_source'] == 'cleavage_map':
			continue
		# Extract protein sequence
		protein = ''.join([row['full_sequence'][0:int(row['start_pos'])],
						   row['fragment'],
						   row['full_sequence'][int(row['end_pos']):]])

		# Store all potential negatives for epitope temporarily
		temp_negatives = set()
		for i in range(
							int(row['n_exclusion_end']), 
							int(row['c_exclusion_start'])
					   ):
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
					p_matches = [x for x in epitope_positives if r.match(x)]
					if not p_matches:
						u_matches = [x for x in epitope_unknowns if r.match(x)]
						if not u_matches:
							true_negatives.add(peptide)
			else:
				# Compare peptide to positives
				in_positives = False
				if peptide in epitope_positives:
					in_positives = True
				else: 
					for regex in epitope_positive_regex:
						if regex.match(peptide):
							in_positives = True
							break
				# Check if peptide matches unknowns if not in positives
				if not in_positives:
					in_unknowns = False
					if peptide in epitope_unknowns:
						in_unknowns = True
					else:
						for regex in epitope_unknown_regex:
							if regex.match(peptide):
								in_unknowns = True
								break
					# Only consider true negative if not in unknowns
					if not in_unknowns:
						true_negatives.add(peptide)
		# Add negative(s) to negative set
		if not args.full_negative_set:
			# Add only one negative to final set
			new_negatives = [
						x for x in true_negatives if x not in epitope_negatives
			]
			if new_negatives:
				# Determine how many negatives to select
				neg_count = 0
				if row['end_pos'] != len(row['full_sequence']):
					neg_count += 1
				if args.trimming_window > len(row['fragment']):
					if row['start_pos'] != 0:
						neg_count += 1
				selected = random.sample(
							new_negatives, min(len(new_negatives), neg_count)
						   )
				for peptide in selected:
					epitope_negatives[peptide].add(tuple(row[0:-3]))
			# Add info for redundant negatives
			old_negatives = [
							x for x in true_negatives if x in epitope_negatives
			]
			for neg in old_negatives:
				epitope_negatives[neg].add(tuple(row[0:-3]))
		else:
			# Store all negatives
			for neg in true_negatives:
				epitope_negatives[neg].add(tuple(row[0:-3]))

	# Create feature arrays for positive and negative examples for each set
	data_set = {
					'epitope': {'positives': {}, 'negatives': {}, 'unknowns': {}}, 
					'pepsickle': {'positives': {}, 'negatives': {}},
			   }
	# Store epitope positives/negatves
	for window in epitope_positives:
		data_set['epitope']['positives'][window] = epitope_positives[window]
	for window in epitope_negatives:
		data_set['epitope']['negatives'][window] = epitope_negatives[window]
	for window in epitope_unknowns:
		data_set['epitope']['unknowns'][window] = epitope_unknowns[window]
	# Store constitutive pepsickle positives/negatives
	for window in proteasome_positives:
		data_set['pepsickle']['positives'][window] = proteasome_positives[window]
	for window in proteasome_negatives:
		data_set['pepsickle']['negatives'][window] = proteasome_negatives[window]
	# Store immunoproteasome positives/negatives
	for window in immunoproteasome_positives:
		if window in data_set['pepsickle']['positives']:
			data_set['pepsickle']['positives'][window].update(immunoproteasome_positives[window])
		else:
			data_set['pepsickle']['positives'][window] = immunoproteasome_positives[window]
	for window in immunoproteasome_negatives:
		if window in data_set['pepsickle']['negatives']:
			data_set['pepsickle']['negatives'][window].update(immunoproteasome_negatives[window])
		else:
			data_set['pepsickle']['negatives'][window] = immunoproteasome_negatives[window]
	for window in mixed_positives:
		if window in data_set['pepsickle']['positives']:
			data_set['pepsickle']['positives'][window].update(mixed_positives[window])
		else:
			data_set['pepsickle']['positives'][window] = mixed_positives[window]
	for window in mixed_negatives:
		if window in data_set['pepsickle']['negatives']:
			data_set['pepsickle']['negatives'][window].update(mixed_negatives[window])
		else:
			data_set['pepsickle']['negatives'][window] = mixed_negatives[window]
	# Store dataset to pickled dictionary
	with open(args.output_dict, 'wb') as p:
		pickle.dump(data_set, p)

	print(datetime.now(), 'Done!', file=sys.stderr)
