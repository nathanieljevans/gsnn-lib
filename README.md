# gsnn-lib
Library for reproducing the works presented in "Graph Structured Neural Networks for Perturbation Biology"

NOTE: this repo is currently in development. Please contact: evansna@ohsu.edu for questions. 

```
@article {Evans2024.02.28.582164,
	author = {Nathaniel J. Evans and Gordon B. Mills and Guanming Wu and Xubo Song and Shannon McWeeney},
	title = {Graph Structured Neural Networks for Perturbation Biology},
	elocation-id = {2024.02.28.582164},
	year = {2024},
	doi = {10.1101/2024.02.28.582164},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/02/29/2024.02.28.582164},
	eprint = {https://www.biorxiv.org/content/early/2024/02/29/2024.02.28.582164.full.pdf},
	journal = {bioRxiv}
}
```

The figures and analysis presented in the preprint can be run using the code available from this [release](https://github.com/nathanieljevans/GSNN/releases/tag/v1.0.0). The GSNN library can be found [here](https://github.com/nathanieljevans/GSNN). Note, the code in this repo expects the GSNN library to be installed prior to running these analysis, please follow the instructions [here](https://github.com/nathanieljevans/GSNN/README.md) to install GSNN.


## Getting Started

Create the `conda/mamba` python environment and install the GSNN package: 
```bash 
$ mamba env create -f environment.yml 
$ conda activate gsnn 
(gsnn) $ pip install -e .
```

Download the necessary raw data: 
```bash 
$ ./get_data.sh /path/to/download/dir/
```

Process and save data appropriate for modeling: 
```bash 
(gsnn) $ python make_data.py --data /path/to/download/dir/ --out /path/to/processed/dir/ --pathways R-HSA-9006934 --feature_space landmark best-inferred --targetome_targets
```

Train models: 
```bash 
(gsnn) $ python train_gsnn.py --data /path/to/processed/dir/ --fold /path/to/data/partitions/dir/ --out /path/to/output/ 

(gsnn) $ python train_gnn.py --data /path/to/processed/dir/ --fold /path/to/data/partitions/dir/ --out /path/to/output/ 

(gsnn) $ python train_nn.py --data --data /path/to/processed/dir/ --fold /path/to/data/partitions/dir/ --out /path/to/output/
```

> NOTE: use ```$ python <fn> --help``` to get optional command line arguments. 

## Manuscript Figures 

See the `jupyter notebooks` in `./notebooks/` for the scripts used to generate all figures in manuscript. For more information, contact: evansna@ohsu.edu. 