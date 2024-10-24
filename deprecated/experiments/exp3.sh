#!/bin/zsh 

# example use: 
###     $ ./exp3.sh

# Signaling by ALK: R-HSA-201556
# 		PIP3 activates AKT signaling: R-HSA-1257604
# 				MTOR signalling: R-HSA-165159
# 						Energy dependent regulation of mTOR by LKB1-AMPK: R-HSA-380972					
# 				PTEN Regulation: R-HSA-6807070
# 				Intrinsic Pathway for Apoptosis: R-HSA-109606
#				Metabolism of nitric oxide: NOS3 activation and regulation: R-HSA-202131
#				Regulation of TP53 Expression and Degradation: R-HSA-6806003
# 							Regulation of TP53 Activity through Phosphorylation: R-HSA-6804756
# 							DNA Double Strand Break Response: R-HSA-5693606
# 		RAF/MAP kinase cascade: R-HSA-5673001 
# 		DAG and IP3 signaling: R-HSA-1489509

########## PARAMS #########

NAME=exp3
PATHWAY="R-HSA-201556 R-HSA-1257604 R-HSA-165159 R-HSA-165159 R-HSA-380972 R-HSA-6807070 R-HSA-109606 R-HSA-202131 R-HSA-6806003 R-HSA-6804756 R-HSA-5693606 R-HSA-5673001 R-HSA-1489509"
DATA=../../data/
OUT=../output_01_04_24/$NAME/
PROC=$OUT/proc/
EPOCHS=100
N_FOLDS=10

MAKE_DATA_TIME=08:00:00
MAKE_DATA_CPUS=8
MAKE_DATA_MEM=32G
FEATURE_SPACE="landmark"
TARGETOME="--targetome_targets"
STITCH=""
FULL_GRN=""

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
rm -r $OUT
mkdir $OUT
mkdir $PROC

# create processed data directory 
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space $FEATURE_SPACE $TARGETOME $STITCH $FULL_GRN >> $PROC/make_data.out

if [ -e "$PROC/make_data_completed_successfully.flag" ]; then

	for (( FOLD=1; FOLD<=$N_FOLDS; FOLD++ )); do

		FOLD_DIR="$OUT/FOLD-\$FOLD"
		echo "FOLD DIR: \$FOLD_DIR"
		mkdir \$FOLD_DIR 

		# comment this out to add new runs to the existing folds. 
		python create_data_splits.py --data $DATA --proc $PROC --out \$FOLD_DIR

		echo 'submitting gsnn jobs...'
		mkdir \$FOLD_DIR/GSNN/
		#                                          HH:MM:SS MEM BTCH GRES        
		./batched_gsnn.sh $PROC \$FOLD_DIR/GSNN/ $EPOCHS $GSNN_TIME $GSNN_MEM $GSNN_BATCH $GSNN_GRES \$FOLD_DIR 

		echo 'submitting nn jobs...'
		mkdir \$FOLD_DIR/NN/
		#                                      HH:MM:SS MEM BTCH
		./batched_nn.sh $PROC \$FOLD_DIR/NN/ $EPOCHS $NN_TIME $NN_MEM $NN_BATCH \$FOLD_DIR 

		echo 'submitting gnn jobs...'
		mkdir \$FOLD_DIR/GNN/
		#                                        HH:MM:SS MEM   BTCH
		./batched_gnn.sh $PROC \$FOLD_DIR/GNN/ $EPOCHS $GNN_TIME $GNN_MEM $GNN_BATCH $GNN_GRES \$FOLD_DIR 

	done 

else 
	echo "make_data.py did not complete successfully. no model batch scripts submitted."
fi
EOF
