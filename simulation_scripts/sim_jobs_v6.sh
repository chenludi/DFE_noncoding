#!/bin/bash
#$ -l h_rt=72:00:00,h_data=16G,highp
#$ -o <output_log_path_here>
#$ -e <error_log_path_here>
#$ -A klohmuel
#$ -N <name_here>
#$ -t 1-20:1


SLIM=<path_to_slim_binary>

# from np.random.seed(909012); np.random.randint(0, 100_000, 20)
seeds=(38904 21581 1974 89559 74771 93378 80849 97632 79452 96002 6130 5408 11347 87666 26784 14981 54027 5824 83353 66671)

seed=${seeds[${SGE_TASK_ID}-1]}
version="v6"

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
