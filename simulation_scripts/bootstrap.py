import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

out_path = "/u/home/s/sweram/project-klohmuel/stuff_for_chenlu/bgnc_simulations_v2/output"


#%%
# get sfs function
def get_sfs(sim_names, ss, sim_path):
    arr_dim = (len(sim_names), ss)
    qui_df, bg_df, cnc_df, exon_syn_df, exon_nsyn_df = np.zeros(arr_dim), np.zeros(arr_dim), np.zeros(arr_dim), \
                                                       np.zeros(arr_dim), np.zeros(arr_dim)
    for i, sim_name in enumerate(sim_names):
        # quiescent
        qui_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m4_{ss}_ss.csv"))
        # background
        mut3 = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m3_{ss}_ss.csv"))
        mut5 = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m5_{ss}_ss.csv"))
        bg_sfs = mut3 + mut5
        # conserved non-coding
        cnc_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m6_{ss}_ss.csv"))
        # exons
        exon_nsyn_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m1_{ss}_ss.csv"))
        exon_syn_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m2_{ss}_ss.csv"))

        qui_df[i] = qui_sfs
        bg_df[i] = bg_sfs
        cnc_df[i] = cnc_sfs
        exon_nsyn_df[i] = exon_nsyn_sfs
        exon_syn_df[i] = exon_syn_sfs

    return qui_df, bg_df, cnc_df, exon_nsyn_df, exon_syn_df


# v5 has different mutation types
def get_sfs_nc_neut(sim_names, ss, sim_path):
    arr_dim = (len(sim_names), ss)
    qui_df, bg_df, cnc_df, exon_syn_df, exon_nsyn_df = np.zeros(arr_dim), np.zeros(arr_dim), np.zeros(arr_dim), \
        np.zeros(arr_dim), np.zeros(arr_dim)
    for i, sim_name in enumerate(sim_names):
        # quiescent
        qui_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m4_{ss}_ss.csv"))
        # background
        bg_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m3_{ss}_ss.csv"))
        # conserved non-coding
        cnc_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m5_{ss}_ss.csv"))
        # exons
        exon_nsyn_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m1_{ss}_ss.csv"))
        exon_syn_sfs = np.loadtxt(os.path.join(sim_path, sim_name, f"sfs_m2_{ss}_ss.csv"))

        qui_df[i] = qui_sfs
        bg_df[i] = bg_sfs
        cnc_df[i] = cnc_sfs
        exon_nsyn_df[i] = exon_nsyn_sfs
        exon_syn_df[i] = exon_syn_sfs

    return qui_df, bg_df, cnc_df, exon_nsyn_df, exon_syn_df


# bootstrap function
def bootstrap(ss, sim_dfs, size, sim_details, reps=10):
    arr_dim = (reps, ss)
    qui_boot, bg_boot, cnc_boot, exon_syn_boot, exon_nsyn_boot = np.zeros(arr_dim), np.zeros(arr_dim), \
                                                                 np.zeros(arr_dim), np.zeros(arr_dim), np.zeros(arr_dim)
    boot_details_dict = {}
    for i in range(reps):
        if size <= sim_dfs[0].shape[0]:
            rand_rows = np.random.choice(sim_dfs[0].shape[0], size=size, replace=False)
        else:
            rand_rows = np.random.choice(sim_dfs[0].shape[0], size=size, replace=True)
        boot_details = bootstrap_seq_lens(rand_rows, sim_details)
        boot_details_dict[f'boot_{i + 1}_ss_{ss}_size_{size}'] = boot_details
        qui_boot[i] = sim_dfs[0][rand_rows].sum(axis=0)
        bg_boot[i] = sim_dfs[1][rand_rows].sum(axis=0)
        cnc_boot[i] = sim_dfs[2][rand_rows].sum(axis=0)
        exon_nsyn_boot[i] = sim_dfs[3][rand_rows].sum(axis=0)
        exon_syn_boot[i] = sim_dfs[4][rand_rows].sum(axis=0)

    boot_details_df = pd.DataFrame.from_dict(boot_details_dict, orient='index')
    boot_details_df['sample_sizes'] = ss
    boot_details_df[['chrom_type', 'Na', 'mu']] = sim_details.loc['sim_1', ['chrom_type', 'Na', 'mu']].to_list()
    boot_details_df = boot_details_df[sim_details.columns]
    return qui_boot, bg_boot, cnc_boot, exon_nsyn_boot, exon_syn_boot, boot_details_df


def bootstrap_seq_lens(rand_rows, sim_details):
    rand_rows_details = sim_details.iloc[rand_rows, :]
    boot_dict = {}
    boot_dict.update(rand_rows_details.sum(axis=0)[['L_qui', 'L_bg', 'L_cnc', 'L_exon', 'L_total']].to_dict())
    boot_dict['recomb_avg'] = rand_rows_details['recomb_avg'].mean()
    return boot_dict


def bootstrap_helper(v, s, sim_names, size=10, reps=10):
    sim_path = os.path.join(out_path, v)
    sim_details = pd.read_csv(os.path.join(out_path, "compiled_sfss", v, f"{v}_simulation_details.csv"), sep='\t',
                              index_col=0)
    if v in ['v5', 'v6']:
        dfs = get_sfs_nc_neut(sim_names, s, sim_path)
    else:
        dfs = get_sfs(sim_names, s, sim_path)
    boot = bootstrap(s, dfs, size, sim_details, reps=reps)
    np.savetxt(os.path.join(out_path, 'compiled_sfss', v, f'{v}_boot_qui_{s}_size{size}.csv'), boot[0], fmt='%d')
    np.savetxt(os.path.join(out_path, 'compiled_sfss', v, f'{v}_boot_bg_{s}_size{size}.csv'), boot[1], fmt='%d')
    np.savetxt(os.path.join(out_path, 'compiled_sfss', v, f'{v}_boot_cnc_{s}_size{size}.csv'), boot[2], fmt='%d')
    np.savetxt(os.path.join(out_path, 'compiled_sfss', v, f'{v}_boot_exon_nsyn_{s}_size{size}.csv'), boot[3], fmt='%d')
    np.savetxt(os.path.join(out_path, 'compiled_sfss', v, f'{v}_boot_exon_syn_{s}_size{size}.csv'), boot[4], fmt='%d')
    sim_details = pd.concat([sim_details, boot[5]])
    sim_details.to_csv(os.path.join(out_path, "compiled_sfss", v, f"{v}_simulation_details.csv"), sep='\t')


#%%
# make files for v3
np.random.seed(8312565)
ss = 100
names = [f"sim_{i}" for i in range(1, 21)]
for vers in ['v2', 'v3']:
    bootstrap_helper(vers, ss, names)

sss = [10, 1000]
names = [f"sim_{i}" for i in range(1, 21)]
for vers, ss in product(['v2', 'v3'], sss):
    bootstrap_helper(vers, ss, names)


#%%
# make files for v4
np.random.seed(948765165)
sss = [10, 100, 1000]
names = [f"sim_{i}" for i in range(1, 21)]
vers = 'v4'
for ss in sss:
    bootstrap_helper(vers, ss, names)


#%%
# make files for v5
np.random.seed(55274413)
sss = [10, 100, 1000]
names = [f"sim_{i}" for i in range(1, 21)]
vers = 'v5'
for ss in sss:
    bootstrap_helper(vers, ss, names)


#%%
# bootstrap 100 replicates
np.random.seed(29481837)
sss = [10, 100, 1000]
names = [f"sim_{i}" for i in range(1, 21)]
vers = ['v2', 'v3', 'v4', 'v5']
for v in vers:
    for ss in sss:
        bootstrap_helper(v, ss, names, size=100)


#%%
# make files for v6 and v7
np.random.seed(123094)
sss = [10, 100, 1000]
names = [f"sim_{i}" for i in range(1, 21)]
vers = ['v6', 'v7']
for v in vers:
    for ss in sss:
        bootstrap_helper(v, ss, names)
        bootstrap_helper(v, ss, names, size=100)

#%%

# plot sfs on top of each other
vers = 'v2'
sss = [10, 100, 1000]
sim_names = [f"sim_{i}" for i in range(1, 21)]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for ss in sss:
    qui_df, bg_df, cnc_df = get_sfs(sim_names, ss, os.path.join(out_path, vers))
    qui_sum, bg_sum, cnc_sum = qui_df.sum(axis=0), bg_df.sum(axis=0), cnc_df.sum(axis=0)
    x_vals = np.arange(1, ss + 1) / ss
    for i, count_sfss in enumerate([qui_sum, bg_sum, cnc_sum]):
        ax[0, i].plot(x_vals, count_sfss, label=f"ss={ss}")
        ax[1, i].plot(x_vals, count_sfss / count_sfss.sum(), label=f"ss={ss}")
for i, title in enumerate(["Quiescent", "Background", "Conserved non-coding"]):
    ax[0, i].set_title(title)
    ax[0, i].set_xlabel("Allele frequency")
    ax[0, i].set_ylabel("Count")
    ax[0, i].legend()
    ax[1, i].set_title(title)
    ax[1, i].set_xlabel("Allele frequency")
    ax[1, i].set_ylabel("Proportion")
    ax[1, i].legend()
fig.tight_layout()
fig.show()

#%%
# window thing