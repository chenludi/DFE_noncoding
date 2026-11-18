import os
import pandas as pd
import numpy as np

data_path = "/u/home/s/sweram/project-klohmuel/stuff_for_chenlu/bgnc_simulations_v2"
outdir = os.path.join(data_path, "output")

#%%

def get_sfs(sfs_paths, out_path):
    for ss in [10, 100, 1000]:
        qui_sum_sfs, bg_sum_sfs, cnc_sum_sfs = np.zeros(ss), np.zeros(ss), np.zeros(ss)
        for i, path in enumerate(sfs_paths):
            # quiescent
            mut4 = np.loadtxt(os.path.join(path, f"sfs_m4_{ss}_ss.csv"))
            qui_sum_sfs += mut4
            # background
            mut3 = np.loadtxt(os.path.join(path, f"sfs_m3_{ss}_ss.csv"))
            mut5 = np.loadtxt(os.path.join(path, f"sfs_m5_{ss}_ss.csv"))
            bg_sum_sfs += mut3 + mut5
            # conserved non-coding
            mut6 = np.loadtxt(os.path.join(path, f"sfs_m6_{ss}_ss.csv"))
            cnc_sum_sfs += mut6

        np.savetxt(f"{out_path}/fs_qui_{ss}_ss.csv", qui_sum_sfs, fmt='%d')
        np.savetxt(f"{out_path}/fs_bg_{ss}_ss.csv", bg_sum_sfs, fmt='%d')
        np.savetxt(f"{out_path}/fs_cnc_{ss}_ss.csv", cnc_sum_sfs, fmt='%d')

#%%
# create table of details
def create_details_file(v):
    os.makedirs(os.path.join(outdir, "compiled_sfss", v), exist_ok=True)
    annots_path = os.path.join(data_path, "sim_files", v)
    info_df = pd.DataFrame()
    info_df.index = [f'sim_{i}' for i in range(1, 21)]
    info_df['chrom_type'] = 'real'
    info_df['sample_sizes'] = '[10, 100, 1000]'
    info_df['Na'] = 10000
    info_df['mu'] = 2e-8
    info_df[['L_qui', 'L_bg', 'L_cnc', 'L_exon']] = 0
    info_df['recomb_avg'] = 0.0

    for sim_num in range(1, 21):
        annots = pd.read_csv(os.path.join(annots_path, f"sim_{sim_num}_annots.csv"))
        annots['length'] = (annots['end'] + 1) - annots['start']  # slim is stupid
        sums = annots.groupby('type')['length'].sum()
        info_df.loc[f'sim_{sim_num}', ['L_qui', 'L_bg', 'L_cnc', 'L_exon']] = sums.loc[['qui', 'bkgd', 'cnc', 'exon']].values
        recomb = pd.read_csv(os.path.join(annots_path, f"sim_{sim_num}_recomb.csv"))
        info_df.loc[f'sim_{sim_num}', 'recomb_avg'] = recomb['rate'].mean()

    info_df['L_total'] = info_df['L_qui'] + info_df['L_bg'] + info_df['L_cnc'] + info_df['L_exon']
    info_df.to_csv(os.path.join(outdir, "compiled_sfss", v, f"{v}_simulation_details.csv"), sep='\t')

for v in ['v2', 'v3', 'v4', 'v5']:
    create_details_file(v)

for v in ['v6', 'v7']:
    create_details_file(v)

#%%
# compile sfss
sim_dirs = [os.path.join(data_path, "output", vers, f"sim_{sim_num}") for sim_num in range(1, 11)]
get_sfs(sim_dirs, os.path.join(outdir, "compiled_sfss", vers))
