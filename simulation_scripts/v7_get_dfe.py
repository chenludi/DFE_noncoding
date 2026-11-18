import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sim_path = "/u/home/s/sweram/project-klohmuel/stuff_for_chenlu/bgnc_simulations_v2/output/v7"

#%%
"""Read in and process data"""
sim_nums = np.arange(1, 21)
mut_type_dict = {1: 'Nonsynonymous',
                 2: 'Synonymous',
                 3: 'Background non-coding',
                 4: 'Quiescent',
                 5: 'Background non-coding',
                 6: 'Conserved non-coding'}
mut_types = np.unique(list(mut_type_dict.values()))

all_muts = pd.DataFrame()
for n in sim_nums:
    mut_type = pd.read_csv(os.path.join(sim_path, f"sim_{n}", "mut_type.csv"), header=None, names=['type'])
    mut_sel = pd.read_csv(os.path.join(sim_path, f"sim_{n}", "mut_sel.csv"), header=None, names=['sel'])

    muts = pd.concat([mut_type, mut_sel], axis=1)
    all_muts = pd.concat([all_muts, muts], ignore_index=True, axis=0)

all_muts['type_name'] = all_muts['type'].map(mut_type_dict)
assert all_muts['type_name'].isna().sum() == 0
assert (all_muts['sel'] > 0.5).sum() == 0

# convert to dadi selection definition
all_muts['sel'] = np.abs(all_muts['sel'] / 2)

#%%
"""Calculate DFE"""

dfe_bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.5]  # nothing should be above 0.5
dfe_bin_names = [r'$0 \leq \mid s \mid < 10^{-5}$',
                 r'$10^{-5} \leq \mid s \mid < 10^{-4}$',
                 r'$10^{-4} \leq \mid s \mid < 10^{-3}$',
                 r'$10^{-3} \leq \mid s \mid < 10^{-2}$',
                 r'$10^{-2} \leq \mid s \mid < 0.5$']

plot_df = pd.DataFrame(index=dfe_bin_names)
for mut_type in mut_types:
    mut_df = all_muts[all_muts['type_name'] == mut_type]
    dfe = np.histogram(mut_df['sel'], bins=dfe_bins)[0]
    dfe_prop = dfe / dfe.sum()
    plot_df[f'{mut_type} (prop)'] = dfe_prop
    plot_df[f'{mut_type} (counts)'] = dfe
plot_df.to_csv(os.path.join(sim_path, "dfes.csv"))

#%%
"""Plot DFE"""
sub_plot_df = plot_df[['Background non-coding', 'Conserved non-coding', 'Nonsynonymous']]
pal = {'Nonsynonymous': 'mediumpurple', 'Background non-coding': 'royalblue', 'Conserved non-coding': 'limegreen'}
melt_df = pd.melt(sub_plot_df.reset_index(), id_vars='index', var_name='mut_type', value_name='proportion')
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=melt_df, x='index', y='proportion', hue='mut_type', palette=pal, ax=ax)
ax.legend().set_title('')
ax.set_ylim([0, 1])
ax.set_xlabel('')
ax.set_ylabel('Probability Mass')
fig.show()

#%%
"""Compile mutation type, selection coefficient, and generation tick for Chenlu"""
all_data = pd.DataFrame(columns=['sim_num', 'type', 'sel', 'gen'])
for n in sim_nums:
    mut_type = pd.read_csv(os.path.join(sim_path, f"sim_{n}", "mut_type.csv"), header=None, names=['type'])
    mut_sel = pd.read_csv(os.path.join(sim_path, f"sim_{n}", "mut_sel.csv"), header=None, names=['sel'])
    mut_tick = pd.read_csv(os.path.join(sim_path, f"sim_{n}", "mut_tick.csv"), header=None, names=['gen'])

    muts = pd.concat([mut_type, mut_sel, mut_tick], axis=1)
    muts['sim_num'] = n
    all_data = pd.concat([all_data, muts], ignore_index=True, axis=0)

all_data['type'] = all_data['type'].map(mut_type_dict)
assert all_data['type'].isna().sum() == 0
assert (all_data['sel'] > 0.5).sum() == 0

# convert to dadi selection definition
all_data['sel'] = np.abs(all_data['sel'] / 2)

all_data.to_csv(os.path.join(sim_path, "all_mutation_data.csv"), index=False)