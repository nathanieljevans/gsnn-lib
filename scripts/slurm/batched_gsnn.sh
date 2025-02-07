#!/bin/zsh
# example use:
# ./batched_gsnn.sh $PROC $OUT $EPOCHS $TIME $MEM $N $SS
# ./batched_gsnn.sh ../output/exp1-1/proc/ ../output/gpu_test/ 10 00:10:00 8G 10

ROOT=/home/exacloud/gscratch/mcweeney_lab/evans/gsnn-lib/scripts/training/

#######################################################
#######################################################
###            HYPER-PARAMETER SEARCH SPACE         ###
#######################################################
#######################################################
SS=$7
if [[ "$SS" == "large" ]]
#2x3x2x2x6x2x2x2x3x2=6912
then 
        lr_list=("1e-2" "1e-3")
        do_list=("0" "0.25" "0.5")
        c_list=("5" "10")
        lay_list=("10" "12")
        ase_list=("--add_function_self_edges")
        share_list=("")
        norm_list=("none" "batch" "layer" "softmax" "groupbatch" "edgebatch")
        bias_list=("" "--bias")
        wd_list=("0")
        batch_list=("200" "400")
        optim_list=("adam" "adan")
        init_list=("kaiming" "xavier" "lecun")
        nonlin_list=("elu" "prelu")
        checkpoint_list=("--checkpoint")
        random_list=("none")

elif [[ "$SS" == "small" ]]
#2x3x3x2x2=72
then
        lr_list=("1e-2" "1e-3")
        do_list=("0")
        c_list=("8" "10" "12")
        lay_list=("10" "12" "14")
        ase_list=("--add_function_self_edges")
        share_list=("")
        norm_list=("layer")
        bias_list=("")
        wd_list=("0" "1e-8")
        batch_list=("64" "256")
        optim_list=("adan")
        init_list=("kaiming")
        nonlin_list=("elu")
        checkpoint_list=("--checkpoint")
        random_list=("none")

elif [[ "$SS" == "norm_ablation" ]]
# 2x2x3x6x4=288
then
        lr_list=("1e-2" "1e-3")
        do_list=("0")
        c_list=("3" "6" "9")
        lay_list=("10")
        ase_list=("--add_function_self_edges")
        share_list=("")
        norm_list=("none" "batch" "layer" "softmax" "edgebatch")
        bias_list=("")
        wd_list=("0")
        batch_list=("256" "512" "1024")
        optim_list=("adam")
        init_list=("kaiming")
        nonlin_list=("elu")
        checkpoint_list=("--checkpoint")
        random_list=("none")

elif [[ "$SS" == "init_ablation" ]]
then
        lr_list=("5e-3")
        do_list=("0")
        c_list=("3" "6")
        lay_list=("10")
        ase_list=("--add_function_self_edges")
        share_list=("")
        norm_list=("batch")
        bias_list=("")
        wd_list=("0")
        batch_list=("512")
        optim_list=("adam")
        init_list=("xavier" "kaiming" "lecun" "normal")
        nonlin_list=("elu" "gelu" "prelu" "mish")
        checkpoint_list=("--checkpoint")
        random_list=("none")

elif [[ "$SS" == "rand_ablation" ]]
then
        lr_list=("5e-3")
        do_list=("0")
        c_list=("3" "6")
        lay_list=("10")
        ase_list=("--add_function_self_edges")
        share_list=("")
        norm_list=("batch")
        bias_list=("")
        wd_list=("0")
        batch_list=("512")
        optim_list=("adam")
        init_list=("kaiming")
        nonlin_list=("elu")
        checkpoint_list=("--checkpoint")
        random_list=("none" "graph" "inputs:all" "inputs:drugs" "inputs:omics" "outputs")

fi

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
        optimm=$(echo "${optim_list[@]}" | tr ' ' '\n' | shuf -n 1)
        init=$(echo "${init_list[@]}" | tr ' ' '\n' | shuf -n 1)
        nonlin=$(echo "${nonlin_list[@]}" | tr ' ' '\n' | shuf -n 1)
        chkpt=$(echo "${checkpoint_list[@]}" | tr ' ' '\n' | shuf -n 1)
        random=$(echo "${random_list[@]}" | tr ' ' '\n' | shuf -n 1)

        jobid=$((jobid+1))

        echo "submitting job: GSNN (lr=$lr, do=$do, c=$c, lay=$lay, ase=$ase, share=$share, norm=$norm, bias=$bias, wd=$wd, batch=$batch, optim=$optimm, init=$init, nonlin=$nonlin, chkpt=$chkpt, random=$random)"

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

source ~/.zshrc
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
                     --optim $optimm \
                     --init $init \
                     --nonlin $nonlin \
                     --randomization $random \
                     $share $ase $bias $chkpt

EOF
done
