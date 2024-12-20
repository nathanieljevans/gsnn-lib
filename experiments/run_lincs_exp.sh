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

conda activate gsnn-lib

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
					$UNDIRECTED >> $PROC/make_lincs_data.out

python create_data_splits.py --data $DATA \
							 --proc $PROC \
							 --outer_k $N_FOLDS \
							 --val_prop $VAL_PROP \
							 --hold_out $HOLD_OUT >> $PROC/create_data_splits.out

# perform MC hyper-parameter search
$SCRIPT_DIR/batched_gsnn.sh $PROC $OUT $EPOCHS $GSNN_TIME $GSNN_MEM $N ""
$SCRIPT_DIR/batched_gsnn.sh $PROC $OUT $EPOCHS $GSNN_TIME $GSNN_MEM $N --randomize

$SCRIPT_DIR/batched_nn.sh $PROC $OUT $EPOCHS $NN_TIME $NN_MEM $N

$SCRIPT_DIR/batched_gnn.sh $PROC $OUT $EPOCHS $GNN_TIME $GNN_MEM $N
