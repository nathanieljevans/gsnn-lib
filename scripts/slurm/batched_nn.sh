#!/bin/zsh
# example use:
# ./batched_nn.sh $PROC $OUT $EPOCHS $TIME $MEM 
# ./batched_nn.sh ../output/exp1-1/proc/ ../output/nn_test/ 10 00:10:00 8G
# NOTE: we run this on CPU to conserve priority (tractable on cpu)

ROOT=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/scripts/training/

#######################################################
#######################################################
###            HYPER-PARAMETER SEARCH SPACE         ###
#######################################################
#######################################################
lr_list=("1e-2" "1e-3" "1e-4")
do_list=("0" "0.1" "0.25")
c_list=("64" "124" "256")
lay_list=("1" "2" "4")
arch_list=('nn' 'ae')
batch_list=("256" "512" "1024")
wd_list=("0" "1e-6" "1e-8")
ldim_list=("64" "128" "256")
#######################################################
#######################################################
#######################################################

# COMMANDLINE INPUTS
PROC=$1         # path to processed data directory (str, e.g., ../../proc/lincs)
OUT=$2          # path to output directory (str, e.g., ../../output/NN/)
EPOCHS=$3       # number of training epochs to run (int)
TIME=$4         # amount of time to request for the slurm job (hours); (e.g., 01:00:00 -> 1 hour )
MEM=$5          # amount of memory to request for the slurm job (GB); should be in format xG (e.g., 16G -> 16 GB )
N=$6            # number of jobs for h param search to submit

echo "#######################################"
echo "batched_nn.sh commandline inputs:"
echo "PROC=$PROC"
echo "OUT=$OUT" 
echo "EPOCHS=$EPOCHS" 
echo "TIME=$TIME"
echo "MEM=$MEM" 
echo "N=$N"
echo "#######################################"

# MAKE output directory in case it doesn't exist
mkdir -p $OUT

OUT2=$OUT/SLURM_LOG_NN/
if [ -d "$OUT2" ]; then
        echo "slurm output log dir exists. Erasing contents..."
        rm -r "$OUT2"/*
else
        echo "slurm output log dir does not exist. Creating..."
        mkdir "$OUT2"
fi

jobid=0
# LIMITED HYPER-PARAMETER SEARCH 
for ((i=1; i<=N; i++)); do
        lr=$(echo "${lr_list[@]}" | tr ' ' '\n' | shuf -n 1)
        do=$(echo "${do_list[@]}" | tr ' ' '\n' | shuf -n 1)
        c=$(echo "${c_list[@]}" | tr ' ' '\n' | shuf -n 1)
        lay=$(echo "${lay_list[@]}" | tr ' ' '\n' | shuf -n 1)
        arch=$(echo "${arch_list[@]}" | tr ' ' '\n' | shuf -n 1)
        batch=$(echo "${batch_list[@]}" | tr ' ' '\n' | shuf -n 1)
        ldim=$(echo "${ldim_list[@]}" | tr ' ' '\n' | shuf -n 1)

        jobid=$((jobid+1))

        echo "submitting job: NN (lr=$lr, do=$do, c=$c, lay=$lay, arch=$arch, batch=$batch, ldim=$ldim)"

        # SUBMIT SBATCH JOB 

        sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=nn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=batch
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

conda activate gsnn-lib
cd $ROOT
python train_nn_lincs.py --data $PROC \
                  --out $OUT \
                  --layers $lay \
                  --dropout $do \
                  --channels $c \
                  --lr $lr \
                  --epochs $EPOCHS \
                  --batch $batch \
                  --latent_dim $ldim \
                  --arch $arch

EOF
done
