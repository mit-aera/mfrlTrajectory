#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time 01:00:00

while getopts t:d:i:o: flag
do
    case "${flag}" in
        t) eval_type=${OPTARG};;
        d) tmp_dir=${OPTARG};;
        i) batch_idx=${OPTARG};;
        o) flag_online=${OPTARG};;
    esac
done

echo "Start slurm job eval_$eval_type"
echo "curr path: $curr_path"
echo "tmp dir: $tmp_dir"
echo "flag_online: $flag_online"
echo "num_batch: $num_batch"

if [[ $flag_online -eq 1 ]]
then
mpirun python3 $curr_path/reward_estimator_mpislurm_node.py -o -t $eval_type -d $tmp_dir -i $batch_idx -nb $num_batch
else
mpirun python3 $curr_path/reward_estimator_mpislurm_node.py -t $eval_type -d $tmp_dir -i $batch_idx -nb $num_batch
fi