#!/bin/bash
#$ -l h_rt=72:00:00,h_data=16G,highp
#$ -o /u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/output/logs/
#$ -e /u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/output/logs/
#$ -A klohmuel
#$ -N sim_for_chenlu
#$ -t 1-20:1


SLIM=/u/home/s/sweram/software/miniconda3/bin/slim

# from np.random.seed(122424); np.random.randint(0, 100_000, 20)
seeds=(57475 1475 81343 74959 53851 7391 19431 95199 81369 34310 7869 72250 74033 17263 13333 13460 30838 49647 76180 40622)

seed=${seeds[${SGE_TASK_ID}-1]}
version="v2"

out_path="/u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/output/${version}/sim_${SGE_TASK_ID}"
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
