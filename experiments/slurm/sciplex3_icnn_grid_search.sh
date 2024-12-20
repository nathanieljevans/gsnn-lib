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
LAYERS_LIST=(4) 
CHANNELS_LIST=(124 256)
DROPOUT=0.
WEIGHT_DECAY=0.
F_LR_LIST=(1e-4 1e-3)
T_LR_LIST=(1e-4 1e-3)
REG_LIST=(0.1 0.01)
G_ITERS_LIST=(5 10)

# Request a node for each parameter configuration
ii=0
for F_LR in "${F_LR_LIST[@]}"; do
    for T_LR in "${T_LR_LIST[@]}"; do
        for REG in "${REG_LIST[@]}"; do
            for G_ITERS in "${G_ITERS_LIST[@]}"; do
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
                                    --method icnn \
                                    --T_arch nn \
                                    --weight_decay $WEIGHT_DECAY \
                                    --dropout $DROPOUT \
                                    --f_channels $CHANNELS \
                                    --f_layers $LAYERS \
                                    --f_lr $F_LR \
                                    --T_lr $T_LR \
                                    --iccn_reg $REG \
                                    --icnn_g_iters $G_ITERS

EOF

done
done