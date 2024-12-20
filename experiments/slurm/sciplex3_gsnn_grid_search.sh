#!/bin/zsh

DATA=$1
OUT=$2
PROC=$3

CPUS=16
TIME=24:00:00
MEM=32G
GPU=gpu:1

ITERS=100
BATCH_SIZE=64
LAYERS_LIST=(10 15) 
CHANNELS_LIST=(5 10)
DROPOUT=0.
WEIGHT_DECAY=0.
LR_LIST=(1e-4 1e-3 1e-2)
BLUR_LIST=(0.1 0.05 0.01)
SCALING_LIST=(0.5 0.75 0.9)

# Request a node for each parameter configuration
ii=0
for LR in "${LR_LIST[@]}"; do
    for BLUR in "${BLUR_LIST[@]}"; do
        for SCALING in "${SCALING_LIST[@]}"; do
            for CHANNELS in "${CHANNELS_LIST[@]}"; do
                for LAYERS in "${LAYERS_LIST[@]}"; do

ii=$((ii+1))
NAME="sciplex3_gsnn_${ii}"
            
sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=$GPU
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=$CPUS
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --output=./SLURM_OUT/$NAME-%j.out
#SBATCH --error=./SLURM_OUT/$NAME-%j.err

cd .. 
pwd
echo 'making data...' 
source ~/.zshrc
conda activate gsnn 

python ../scripts/train_sciplex3.py --data $DATA \
                                    --out $OUT \
                                    --proc $PROC \
                                    --iters $ITERS \
                                    --batch_size $BATCH_SIZE \
                                    --T_lr $LR \
                                    --method shd \
                                    --T_arch gsnn \
                                    --blur $BLUR \
                                    --scaling $SCALING \
                                    --weight_decay $WEIGHT_DECAY \
                                    --dropout $DROPOUT \
                                    --channels $CHANNELS \
                                    --layers $LAYERS \
                                    --checkpoint

EOF

done
done