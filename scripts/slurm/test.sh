#!/bin/bash

# Number of random samples to draw
N=10

# Parameter lists
lr_list=("0.01" "0.001")
do_list=("0" "0.1")
c_list=("10" "20")
lay_list=("10" "20")
ase_list=("" "--add_function_self_edges")

echo "Testing random parameter selection:"

for ((i=1; i<=N; i++)); do
    lr=$(echo "${lr_list[@]}" | tr ' ' '\n' | shuf -n 1)
    do=$(echo "${do_list[@]}" | tr ' ' '\n' | shuf -n 1)
    c=$(echo "${c_list[@]}" | tr ' ' '\n' | shuf -n 1)
    lay=$(echo "${lay_list[@]}" | tr ' ' '\n' | shuf -n 1)
    ase=$(echo "${ase_list[@]}" | tr ' ' '\n' | shuf -n 1)

    echo "Sample $i: lr=$lr, dropout=$do, channels=$c, layers=$lay, ase=$ase"
done