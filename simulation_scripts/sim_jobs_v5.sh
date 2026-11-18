#!/bin/bash
#$ -l h_rt=72:00:00,h_data=16G,highp
#$ -o /u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/output/logs/
#$ -e /u/project/klohmuel/sweram/stuff_for_chenlu/bgnc_simulations_v2/output/logs/
#$ -A klohmuel
#$ -N sim_for_chenlu
#$ -t 1-20:1


SLIM=/u/home/s/sweram/software/miniconda3/bin/slim

# from np.random.seed(7171032); np.random.randint(0, 100_000, 20)
seeds=(69295 94593 58256 40216 20326 33807 96123 87355 89473 86195 96641 94282 38210 89608 686 49303 61578 25575 17807 53218)

seed=${seeds[${SGE_TASK_ID}-1]}
version="v5"

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
