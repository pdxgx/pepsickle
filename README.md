# pepsickle [![Build Status](https://travis-ci.com/pdxgx/pepsickle.svg?token=MwZdsoYXNWVDeSqTyWLs&branch=master)](https://travis-ci.com/github/pdxgx/pepsickle)
A context aware tool for proteasomal cleavage predictions


## About
`pepsickle` is an open-source command line tool for  proteasomal cleavage prediction. `pepsickle` is designed with flexibility in mind allows for the use of either direct amino acid input or the `FASTA` files. Predictions can also be determined based on a variety of available models including those trained on: *in-vivo* epitope data (default), *in-vitro* constitutive proteasome data, or *in-vitro* immunoproteasome data. For information on available models and how they were trained, see the [companion paper in *XXXX*]() highlighting this tool, as well as the accompanying [paper repo](https://github.com/pdxgx/pepsickle-paper) with code for training and reproduction.

## License 
`pepsickle` is licensed under the [MIT](https://choosealicense.com/licenses/mit/) license. See [LICENSE](https://github.com/pdxgx/pepsickle/blob/master/LICENSE) for more details.

## Support


## Requirements
`pepsickle` relies on `Python 3` and a few other required packages. A complete list of dependencies can be found in [requirements.txt](https://github.com/pdxgx/pepsickle/blob/master/requirements.txt)

## Installation
Installing `pepsickle` is easy! If you already have [Python 3](https://www.python.org/downloads/), `pepsickle` can simply be installed via the command line using `pip`: 

`pip install pepsickle`

We also recommend using a version control system like [Anaconda](https://docs.anaconda.com/anaconda/install/) to make sure version requirements for pepsickle don't interfere with other packages in use.

## Use
`pepsickle` allows for multiple methods of use. By default, predictions are made based on a model trained using *in-vivo* epitope data. 

During predictions, the upstream and downstream amino acid contexts are used and we therefore recommend including at least 8 amino acids on each side of any sites of interest. If less than the recommended context is given (such as in the case of residues near the beginning or end of a protein sequence) `pepsickle` will auto-pad inputs, however padding can explicitly be added for internal residues using the value `X`. `X`'s submitted to the prediction model are interpreted as the presence of an amino acid sequence with unkonwn identity, while auto-padding is interpreted as the absence of amino acid context all together. 

For predictions on single short amino acid sequences, `pepsickle` can be run
using the `-s` option:

`pepsickle -s VSGLEQLESIINFEKLTEWTSSNV`

For long peptide sequences or to run multiple sequences at once, `pepsickle`can be run using the fasta file `-f` option:

`pepsickle -f /PATH/TO/FASTA.fasta`

For an example of a `FASTA` formatted file, see the [test fasta](https://github.com/pdxgx/pepsickle/blob/master/tests/P43357.fasta) used for this package.

By default, output will be printed to the screen, however output can easily be routed to a file location by using the `-o` option:

`pepsickle -s VSGLEQLESIINFEKLTEWTSSNV -o /PATH/TO/OUTPUT.txt`

Output is in tab separated format. For an example of output format see the [example out file]().

A full list of command line options and descriptions is listed here:

`-s, --sequence [SEQUENCE]` use pepsickle in single sequence mode. Takes a string sequence as input and returns predicted cleavage sites in standard format.

`-f, --fasta [FASTA]` use pepsickle in fasta mode. Takes a fasta file with protein ID's and corresponding sequences.

`-o, --out [OUT_FILE]` name and destination for prediction outputs in TSV format. If none is provided, the output will be printed directly to the screen.

`-v, --verbose` In fasta mode, prints progress during cleavage predictions for fasta files with multiple protein sequences.

`-m, --model-type [epitope (default) | in-vitro | in-vitro-2]` allows the use of models trained on alternative types of data. Defaults to epitope based model, with options for in-vitro based gradient boosted model (in-vitro)or an experimental neural network based in-vitro model (in-vitro-2)

`-p, "--proteasome-type [C | I]` allows predictions to be made based on constitutive proteasomal (C) or immunoproteasomal (I) cleavage profiles. Note that if predictions are made using the epitope-based model (default), predictions will be proteasome type agnostic.

`-t, --threshold [0-1 (default=0.5)]` probability threshold to be used for cleavage predictions

`--human-only` ses models trained on human data only. Note that human only data sets are substantially smaller and may produce less stable predictions.
