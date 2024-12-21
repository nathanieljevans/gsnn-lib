#!/bin/zsh
# example use:
# ./batched_gsnn.sh $PROC $OUT $EPOCHS $TIME $MEM $N $RAND
# ./batched_gsnn.sh ../output/exp1-1/proc/ ../output/gpu_test/ 10 00:10:00 8G 10 --randomize

ROOT=/home/exacloud/gscratch/NGSdev/evans/gsnn-lib/scripts/training/

#######################################################
#######################################################
###            HYPER-PARAMETER SEARCH SPACE         ###
#######################################################
#######################################################
lr_list=("1e-2" "1e-3" "1e-4")
do_list=("0" "0.1")
c_list=("3" "5" "10")
lay_list=("10" "15" "20")
ase_list=("" "--add_function_self_edges")
share_list=("" "--share_layers")
norm_list=("none" "batch" "layer" "softmax")
bias_list=("" "--no_bias")
wd_list=("0" "1e-6" "1e-8")
batch_list=("32" "64" "128")
#######################################################
#######################################################
#######################################################

# COMMANDLINE INPUTS
PROC=$1   # path to processed data directory (str, e.g., ../../proc/lincs) 
OUT=$2    # path to output directory (str, e.g., ../../output/GSNN/)
EPOCHS=$3 # number of training epochs to run (int)
TIME=$4   # amount of time to request for the slurm job (hours); (e.g., 01:00:00 -> 1 hour )
MEM=$5    # amount of memory to request for the slurm job (GB); should be in format xG (e.g., 16G -> 16 GB )
N=$6      # number of jobs for h param search to submit (int)
RAND=$7   # whether to randomize the graph structure (e.g., --randomize, or empty string) 

echo "#######################################"
echo "batched_gsnn.sh commandline inputs:"
echo "PROC=$PROC"
echo "OUT=$OUT" 
echo "EPOCHS=$EPOCHS" 
echo "TIME=$TIME"
echo "MEM=$MEM" 
echo "N=$N"
echo "RAND=$RAND"
echo "#######################################"

# MAKE output directory in case it doesn't exist
mkdir -p $OUT

# make slurm log dir 
OUT2=$OUT/SLURM_LOG_GSNN/
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
        ase=$(echo "${ase_list[@]}" | tr ' ' '\n' | shuf -n 1)
        share=$(echo "${share_list[@]}" | tr ' ' '\n' | shuf -n 1)
        norm=$(echo "${norm_list[@]}" | tr ' ' '\n' | shuf -n 1)
        bias=$(echo "${bias_list[@]}" | tr ' ' '\n' | shuf -n 1)
        wd=$(echo "${wd_list[@]}" | tr ' ' '\n' | shuf -n 1)
        batch=$(echo "${batch_list[@]}" | tr ' ' '\n' | shuf -n 1)

        jobid=$((jobid+1))

        echo "submitting job: GSNN (lr=$lr, do=$do, c=$c, lay=$lay, ase=$ase, share=$share, norm=$norm, bias=$bias, wd=$wd, batch=$batch)"

        # SUBMIT SBATCH JOB 

        sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=gsnn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --partition=gpu
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

conda activate gsnn-lib
cd $ROOT
python train_gsnn_lincs.py --data $PROC \
                     --out $OUT \
                     --dropout $do \
                     --channels $c \
                     --lr $lr \
                     --epochs $EPOCHS \
                     --batch $batch \
                     --layers $lay \
                     --norm $norm \
                     --wd $wd \
                     $share $ase $RAND $bias

EOF
done
