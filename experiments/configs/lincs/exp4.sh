#!/bin/zsh 

########## DESCRIPTION #########

# single output prediction task (TP53) 


########## PARAMS #########

SCRIPT_DIR=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/scripts/slurm/
PROC_DIR=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/scripts/data_proc/
EXTDATA=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/extdata/

# Experiment details 
NAME=exp4
DATA=/home/exacloud/gscratch/NGSdev/evans/data/
OUT=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/output/$NAME/
PROC=$OUT/proc/lincs/
EPOCHS=250

# Graph construction details
FEATURE_SPACE="landmark"		        # options: landmark, best-inferred, inferred [e.g., "landmark best-inferred"]
DTI_SOURCES=("clue" "targetome") 	    # options: clue, targetome, stitch [e.g., "clue targetome stitch"]
DRUGS='none'					        # broad ids (space separated); "none" will include all valid drugs
LINCS="P04637"					        # uniprot lincs outputs (space separated); "none" will include all valid lincs
CELL_LINES='none'					    # cell lines (LINCS `cell_iname`) (space separated); "none" will include all valid lines
OMICS=('expr' 'mut' 'cnv' 'methyl')		# which omics to include
OMICS_Q_FILTER=0.25				        # omics with the std in this quantile will not be included in the graph (remove low variance features)
TIME=24							        # LINCS measurement time (hours); [recommend: 24; options: 6, 24, 48, 72]
FILTER_DEPTH=10					        # Primary criteria for molecular entity inclusion. 
MIN_OBS_PER_DRUG=100			        # number of observations per drug for drug to be included in observations and graph
UNDIRECTED=''					        # whether to make the function->function graph undirected [option: '', '--undirected']

# partition split details 
HOLD_OUT='cell-drug'			        # how to create hold-out sets [options: 'cell-drug', 'cell']. 'cell' will hold out cell lines, 'cell-drug' will hold out cell-drug keys. 
VAL_PROP=0.1					        # proportion of data to hold out for validation
N_FOLDS=5						        # number of outer folds to create

# Hyper-parameter search budget 
N=100 							        # number of parameter configurations to test (randomly sampled); see `batched_xxx.sh` for details on which params to test
SEARCHSPACE="large"                     # options: 'large', 'small' (number of hyper-parameters to test)

# SLURM settings (node request parameters)
GSNN_TIME=24:00:00
GSNN_MEM=32G

NN_TIME=06:00:00
NN_MEM=20G

GNN_TIME=24:00:00
GNN_MEM=32G

RUN_GSNN=1
RUN_GSNN_RAND=0
RUN_NN=1
RUN_GNN=0

###########################
