#!/bin/zsh 

# example use: 
###     $ ./exp1.sh


########## PARAMS #########

## Experiment details 
NAME=exp1
DATA=../../data/
OUT=../output/$NAME/
PROC=$OUT/proc/
EPOCHS=100
N_FOLDS=10

# Graph construction details

FEATURE_SPACE="landmark"		# options: landmark, best-inferred, inferred [e.g., "landmark best-inferred"]
DTI_SOURCES="clue targetome" 	# options: clue, targetome, stitch [e.g., "clue targetome stitch"]
DRUGS='none'					# broad ids (space separated); "none" will include all valid drugs
LINCS='none'					# uniprot lincs outputs (space separated); "none" will include all valid lincs
LINES='none'					# cell lines (LINCS `cell_iname`) (space separated); "none" will include all valid lines
OMICS='expr mut cnv methyl'		# which omics to include
OMICS_Q_FILTER=0.25				# omics with the quantile lower than this value will not be included in the graph
TIME=24							# LINCS measurement time (hours); [recommend: 24, 48, 72]
FILTER_DEPTH=10					# Primary criteria for molecular entity inclusion. 
MIN_OBS_PER_DRUG=100			# number of observations per drug for drug to be included in observations and graph
UNDIRECTED=''					# whether to make the function->function graph undirected [option: '', '--undirected']

# partition split details 
TEST_PROP=0.15
VAL_PROP=0.15
HOLD_OUT='cell-drug'			# how to create hold-out sets [options: 'cell-drug', 'cell']. 'cell' will hold out cell lines, 'cell-drug' will hold out cell-drug keys. 
MIN_NUM_DRUGS_PER_CELL_LINE=5
MIN_NUM_OBS_PER_CELL_LINE=5
MIN_NUM_CELL_LINES_PER_DRUG=3
MIN_NUM_OBS_PER_DRUG=3

# Hyper-parameter search budget 
N=10							# number of parameter configurations to test (randomly sampled); see `batched_xxx.sh` for details on which params to test

## SLURM request details ## 
MAKE_DATA_TIME=01:00:00
MAKE_DATA_CPUS=8
MAKE_DATA_MEM=32G

GSNN_TIME=2-00:00:00
GSNN_MEM=32G
GSNN_BATCH=25
GSNN_GRES=gpu:1

NN_TIME=12:00:00
NN_MEM=20G
NN_BATCH=256

GNN_TIME=24:00:00
GNN_MEM=20G
GNN_BATCH=25
GNN_GRES=gpu:1

###########################

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$MAKE_DATA_CPUS
#SBATCH --time=$MAKE_DATA_TIME
#SBATCH --mem=$MAKE_DATA_MEM
#SBATCH --output=./SLURM_OUT/$NAME-%j.out
#SBATCH --error=./SLURM_OUT/$NAME-%j.err

cd .. 
pwd
echo 'making data...' 
source ~/.zshrc
conda activate gsnn 

echo 'removing out dir and making proc dir...'
# NOTE: comment these out if you are running additional folds with the same data. 
rm -r $OUT
mkdir $OUT
mkdir $PROC

# create processed data directory 
python make_data.py --data $DATA \
					--out $PROC \
					--feature_space $FEATURE_SPACE \
					--dti_sources $DTI_SOURCES \
					--drugs $DRUGS \
					--lines $LINES \
					--lincs $LINCS \
					--omics $OMICS \
					--omics_q_filter $OMICS_Q_FILTER \
					--time $TIME \
					--filter_depth $FILTER_DEPTH \
					--min_obs_per_drug $MIN_OBS_PER_DRUG \
					$DIRECTED >> $PROC/make_data.out

if [ -e "$PROC/make_data_completed_successfully.flag" ]; then

	# NOTE: adjust "FOLD" value if adding additional folds to dataset to prevent overwriting 
	for (( FOLD=1; FOLD<=$N_FOLDS; FOLD++ )); do

		FOLD_DIR="$OUT/FOLD-\$FOLD"
		echo "FOLD DIR: \$FOLD_DIR"
		mkdir \$FOLD_DIR 

		python create_data_splits.py --data $DATA --proc $PROC --out $FOLD_DIR \
									 --test_prop $TEST_PROP \ 
									 --val_prop $VAL_PROP \ 
									 --hold_out $HOLD_OUT \ 
									 --min_num_drugs_per_cell_line $MIN_NUM_DRUGS_PER_CELL_LINE \ 
									 --min_num_obs_per_cell_line $MIN_NUM_OBS_PER_CELL_LINE \ 
									 --min_num_cell_lines_per_drug $MIN_NUM_CELL_LINES_PER_DRUG \ 
									 --min_num_obs_per_drug $MIN_NUM_OBS_PER_DRUG 

		echo 'submitting gsnn jobs...'
		mkdir \$FOLD_DIR/GSNN/
		#                                          HH:MM:SS MEM BTCH GRES        
		./batched_gsnn.sh $PROC \$FOLD_DIR/GSNN/ $EPOCHS $GSNN_TIME $GSNN_MEM $GSNN_BATCH $GSNN_GRES \$FOLD_DIR $N

		echo 'submitting nn jobs...'
		mkdir \$FOLD_DIR/NN/
		#                                      HH:MM:SS MEM BTCH
		./batched_nn.sh $PROC \$FOLD_DIR/NN/ $EPOCHS $NN_TIME $NN_MEM $NN_BATCH \$FOLD_DIR $N

		echo 'submitting gnn jobs...'
		mkdir \$FOLD_DIR/GNN/
		#                                        HH:MM:SS MEM   BTCH
		./batched_gnn.sh $PROC \$FOLD_DIR/GNN/ $EPOCHS $GNN_TIME $GNN_MEM $GNN_BATCH $GNN_GRES \$FOLD_DIR $N

	done 

else 
	echo "make_data.py did not complete successfully. no model batch scripts submitted."
fi
EOF

