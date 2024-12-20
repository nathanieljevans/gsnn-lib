#!/bin/zsh 

# example use: 
###     $ ./exp1.sh

####### DATA PARAMS #########

DATA=../../data/
OUT=../output/sciplex3/
EXTDATA=../extdata/
DTI_SOURCES='clue targetome'
FILTER_DEPTH=10
N_GENES=1000
UNDIRECTED=False
SEED=0
VAL_PROP=0.15
TEST_PROP=0.15
DOSE_EPS_=1e-6

###############################

source ~/.zshrc
conda activate gsnn 

# make data 
python make_sciplex3_data.py --data $DATA \
							 --out $OUT \
							 --extdata $EXTDATA \
							 --dti_sources $DTI_SOURCES \
							 --filter_depth $FILTER_DEPTH \
							 --n_genes $N_GENES \
							 --undirected $UNDIRECTED \
							 --seed $SEED \
							 --val_prop $VAL_PROP \
							 --test_prop $TEST_PROP \
							 --dose_eps_ $DOSE_EPS_

../slurm_scripts/sciplex3_gsnn_grid_search.sh $DATA $OUT/models/gsnn/ $OUT

../slurm_scripts/sciplex3_nn_grid_search.sh $DATA $OUT/models/nn/ $OUT

../slurm_scripts/sciplex3_icnn_grid_search.sh $DATA $OUT/models/icnn/ $OUT