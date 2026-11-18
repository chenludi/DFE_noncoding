#!/bin/bash
#$ -l h_rt=72:00:00,h_data=16G,highp
#$ -o <output_log_path_here>
#$ -e <error_log_path_here>
#$ -A klohmuel
#$ -N <name_here>
#$ -t 1-20:1


SLIM=<path_to_slim_binary>

# from np.random.seed(3389251); np.random.randint(0, 100_000, 20)
seeds=(32006 48038 85651 48289 65339 56950 46346 67108 62107 29258 95882 81483 46275 40508 99384 96680 87210 70137 25909 48372)

seed=${seeds[${SGE_TASK_ID}-1]}
version="v7"

out_path=<output_path_here>
sim_annots="/u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/sim_files/${version}/sim_${SGE_TASK_ID}_annots.csv"
sim_recomb="/u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/sim_files/${version}/sim_${SGE_TASK_ID}_recomb.csv"
slim_script="/u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/sim_for_chenlu_${version}.slim"

mkdir -p "${out_path}"

# note that model_number is a string
$SLIM -d seed=${seed} \
  -d out_path="'${out_path}'" \
  -d sim_annots="'${sim_annots}'" \
  -d sim_recomb="'${sim_recomb}'" \
  ${slim_script}
