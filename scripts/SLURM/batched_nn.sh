#!/bin/zsh
# example use:
# ./batched_nn.sh $PROC $OUT $EPOCHS $TIME $MEM $BATCH 
# ./batched_nn.sh ../output/exp1-1/proc/ ../output/nn_test/ 10 00:10:00 8G 256

PROC=$1
OUT=$2 
EPOCHS=$3
TIME=$4 
MEM=$5
BATCH=$6
FOLD_DIR=$7
N=$8      # number of jobs for h param search to submit

# parameter search grid
lr_list=("0.01" "0.001")
do_list=("0" "0.1")
c_list=("10" "20")
lay_list=("10" "20")
ase_list=("" "--add_function_self_edges")

mkdir $OUT

OUT2=$OUT/slurm_logs__NN/
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

        jobid=$((jobid+1))

        echo "submitting job: GSNN (lr=$lr, do=$do, c=$c, lay=$lay, ase=$ase)"

        # SUBMIT SBATCH JOB 

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=nn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

source ~/.zshrc
conda activate gsnn 
cd /home/exacloud/gscratch/NGSdev/evans/GSNN/scripts/
python train_nn.py --data $PROC --fold $FOLD_DIR --out $OUT --layers $lay --dropout $do --channels $c --lr $lr --epochs $EPOCHS --batch $BATCH
python train_nn.py --data $PROC --fold $FOLD_DIR --out $OUT --layers $lay --dropout $do --channels $c --lr $lr --epochs $EPOCHS --batch $BATCH --cell_agnostic

EOF
done
done
done
done
