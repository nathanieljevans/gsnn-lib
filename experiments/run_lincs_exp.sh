#!/bin/zsh 

# example use: 
###     $ ./lincs_exp1.sh ./configs/exp1.sh

# If the user provides a config file as the first argument:
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi

# Source the config
. $CONFIG_FILE

rm -r $OUT
mkdir -p $OUT
mkdir -p $PROC

source ~/.zshrc
conda activate gsnn-lib

python $PROC_DIR/make_lincs_data.py --data $DATA \
					--out $PROC \
					--extdata $EXTDATA\
					--feature_space $FEATURE_SPACE \
					--dti_sources $DTI_SOURCES \
					--drugs $DRUGS \
					--lines $CELL_LINES \
					--lincs $LINCS \
					--omics $OMICS \
					--omics_q_filter $OMICS_Q_FILTER \
					--time $TIME \
					--filter_depth $FILTER_DEPTH \
					--min_obs_per_drug $MIN_OBS_PER_DRUG \
					$EXCLUDE_BLOOD_LINES \
					$UNDIRECTED #>> $PROC/make_lincs_data.out

python $PROC_DIR/create_lincs_splits.py --data $DATA \
							 --proc $PROC \
							 --outer_k $N_FOLDS \
							 --val_prop $VAL_PROP \
							 --hold_out $HOLD_OUT #>> $PROC/create_data_splits.out

# perform MC hyper-parameter search

if [ "$RUN_GSNN" -eq 1 ]; then 
	$SCRIPT_DIR/batched_gsnn.sh $PROC $OUT $EPOCHS $GSNN_TIME $GSNN_MEM $N $SEARCHSPACE
fi

if [ "$RUN_NN" -eq 1 ]; then 
	$SCRIPT_DIR/batched_nn.sh $PROC $OUT $EPOCHS $NN_TIME $NN_MEM $N $SEARCHSPACE
fi 

if [ "$RUN_GNN" -eq 1 ]; then 
	$SCRIPT_DIR/batched_gnn.sh $PROC $OUT $EPOCHS $GNN_TIME $GNN_MEM $N $SEARCHSPACE
fi