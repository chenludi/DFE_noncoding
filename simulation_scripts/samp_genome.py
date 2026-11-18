#%%
#!/u/home/s/sweram/software/miniconda3/envs/stdpopsim/bin/python
import os
import stdpopsim
import numpy as np
import pandas as pd


def check_intervals_overlap(starts, ends):
    # intervals should be sorted already
    for i in range(len(starts) - 1):
        if starts[i+1] < ends[i]:
            print(i)
            raise ValueError("Overlapping intervals")


# bedtools subtract -a $statebed -b $statequi
# trivial and obvious (not)
# this function DOES NOT allow overlapping intervals in a
def sub_intervals(xs, ys):
    if len(xs) == 0:
        return []

    i = j = 0
    rs = []
    cur_lo = xs[0][0]

    while i < len(xs) and j < len(ys):
        x_lo, x_hi = xs[i]
        cur_lo = max(cur_lo, x_lo)
        y_lo, y_hi = ys[j]

        if cur_lo < y_lo:
            rs.append((cur_lo, min(y_lo, x_hi)))
            if x_hi < y_lo:
                i += 1
            else:
                cur_lo = y_lo
        else:
            if y_hi < x_hi:
                cur_lo = y_hi
                j += 1
            else:
                i += 1

    if i < len(xs):
        rs.append((max(cur_lo, xs[i][0]), xs[i][1]))
    rs.extend(xs[i+1:])
    return rs


def merge_intervals(intervals, sort=True):
    if len(intervals) == 0:
        return []

    if not sort:
        intervals = sorted(intervals, key=lambda x: x[0])

    rs = []
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            rs.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi

    rs.append((cur_lo, cur_hi))
    return rs


def fill_in_annots(annots, fill_name, fill_col):
    """
    >>> df = pd.DataFrame({'start': [1, 5, 10, 15], 'end': [3, 9, 15, 21], 'type': ['exon']*4})
    >>> fill_in_annots(df, 'fill', 'type')
       start  end   type
    0      1    3   exon
    1      3    5   fill
    2      5    9   exon
    3      9   10   fill
    4     10   15   exon
    6     15   21   exon
    """
    new_annots = []
    for i in range(len(annots) - 1):
        s, e, f = annots.iloc[i]
        new_annots.append((s, e, f))
        next_s = annots.iloc[i + 1, 0]
        if e < next_s:
            new_annots.append((e, next_s, fill_name))

    new_annots.append((annots.iloc[-1, 0], annots.iloc[-1, 1], annots.iloc[-1, 2]))
    return pd.DataFrame(new_annots, columns=['start', 'end', fill_col])


# run this once
def make_chrom_annots():
    if all([os.path.exists(f"./sim_files/{chrom}_annotations.csv") for chrom in chr_names]):
        print("Annotations already exist")
        return

    os.makedirs("./sim_files", exist_ok=True)

    # centromere and gap locations from ucsc genome browser
    centromeres = pd.read_csv('./other_stuff/centromeres.txt', sep='\t', header=None,
                              names=['bin', 'chrom', 'start', 'end', 'name'])
    gaps = pd.read_csv('./other_stuff/gap.txt', sep='\t', header=None,
                       names=['bin', 'chrom', 'start', 'end', 'ix', 'n', 'size', 'type', 'bridge'])

    # top 5% CNC regions from 470-way
    cnc = pd.read_csv("/u/project/klohmuel/cdi/noncdoingdfe/test_0118/data_bed/conserve_bins_nomiss/"
                      "phastCons470way/phastCons470way_0-5_nomiss.bed", sep='\t', header=None,
                      names=['chrom', 'start', 'end'], usecols=[0, 1, 2])

    # quiescent regions from ChromHMM
    qui = pd.read_csv("/u/project/klohmuel/cdi/noncdoingdfe/data/data_ernst/chromHMM_100_states/temp_Quies.bed",
                      sep='\t', header=None, names=['chrom', 'start', 'end'], usecols=[0, 1, 2])

    # CDS annots from stdpopsim
    cds = species.get_annotations('ensembl_havana_104_CDS')

    # for each chromosome, save all types of annotations to file
    for chrom in chr_names:
        # get annotations for each type for this chromosome
        cds_chrom = cds.get_chromosome_annotations(chrom)
        cnc_chrom = cnc.loc[cnc['chrom'] == chrom, ['start', 'end']]
        centromeres_chrom = centromeres.loc[centromeres['chrom'] == chrom, ['start', 'end']]
        gaps_chrom = gaps.loc[gaps['chrom'] == chrom, ['start', 'end']]
        qui_chrom = qui.loc[qui['chrom'] == chrom, ['start', 'end']]

        # quiescent is not sorted so sort here
        qui_chrom = qui_chrom.sort_values(by=['start', 'end'])

        # check for overlaps within each type of annotation
        check_intervals_overlap(cds_chrom[:, 0], cds_chrom[:, 1])
        check_intervals_overlap(cnc_chrom['start'].values, cnc_chrom['end'].values)
        check_intervals_overlap(centromeres_chrom['start'].values, centromeres_chrom['end'].values)
        check_intervals_overlap(gaps_chrom['start'].values, gaps_chrom['end'].values)
        check_intervals_overlap(qui_chrom['start'].values, qui_chrom['end'].values)

        # subtraction
        cds_chrom = sub_intervals(cds_chrom, centromeres_chrom.values)
        cds_chrom = sub_intervals(cds_chrom, gaps_chrom.values)

        cnc_chrom = sub_intervals(cnc_chrom.values, centromeres_chrom.values)
        cnc_chrom = sub_intervals(cnc_chrom, gaps_chrom.values)
        cnc_chrom = sub_intervals(cnc_chrom, cds_chrom)

        qui_chrom = sub_intervals(qui_chrom.values, centromeres_chrom.values)
        qui_chrom = sub_intervals(qui_chrom, gaps_chrom.values)
        qui_chrom = sub_intervals(qui_chrom, cds_chrom)
        qui_chrom = sub_intervals(qui_chrom, cnc_chrom)

        cds_chrom = pd.DataFrame(cds_chrom, columns=['start', 'end'])
        cnc_chrom = pd.DataFrame(cnc_chrom, columns=['start', 'end'])
        qui_chrom = pd.DataFrame(qui_chrom, columns=['start', 'end'])

        # add column to each dataframe for each type of annotation
        cds_chrom['type'] = 'exon'
        cnc_chrom['type'] = 'cnc'
        qui_chrom['type'] = 'qui'
        centromeres_chrom['type'] = 'exclude'
        gaps_chrom['type'] = 'exclude'

        # combine all annotations for this chromosome and sort by start and end
        chrom_annots = pd.concat([cds_chrom, cnc_chrom, qui_chrom, centromeres_chrom, gaps_chrom])
        chrom_annots = chrom_annots.sort_values(by=['start', 'end'])

        # check for overlaps
        if check_intervals_overlap(chrom_annots['start'].values, chrom_annots['end'].values):
            print(f"Overlapping intervals in {chrom}")
            continue

        # fill in gaps between annotations
        chrom_annots = fill_in_annots(chrom_annots, 'bkgd', 'type')

        # save to file
        chrom_annots.to_csv(f"./sim_files/{chrom}_annotations.csv", index=False)


def subset_annotations(chrom_annots, start, end):
    """
    >>> df = pd.DataFrame({'start': [1, 3, 5, 9, 10, 15], 'end': [3, 5, 9, 10, 15, 21]})
    >>> subset_annotations(df, 2, 9)
       start  end
    0      2    3
    1      3    5
    2      5    9
    >>> subset_annotations(df, 2, 11)
       start  end
    0      2    3
    1      3    5
    2      5    9
    3      9   10
    4     10   11
    >>> subset_annotations(df, 10, 21)
       start  end
    2     10   15
    3     15   21
    """
    # get the left bin index for start and end
    start_idx = np.digitize(start, chrom_annots['start'].values) - 1
    end_idx = np.digitize(end, chrom_annots['end'].values)
    if chrom_annots.iloc[end_idx - 1, 1] == end:
        end_idx -= 1
    subset_annots = chrom_annots.iloc[start_idx:end_idx + 1].copy()
    subset_annots.iloc[0, 0] = start
    subset_annots.iloc[-1, 1] = end
    return subset_annots


def sample_chrom_segment(chr_lengths, chr_names, chr_probs, length=20_000_000):
    chrom = np.random.choice(chr_names, p=chr_probs)
    chrom_length = chr_lengths[chrom]
    start = np.random.randint(1, chrom_length - length)
    end = start + length
    return chrom, start, end


#%%
os.chdir("/u/home/s/sweram/project-klohmuel/stuff_for_chenlu/bgnc_simulations_v2")

# chromosome details from stdpopsim
species = stdpopsim.get_species("HomSap")
chr_lengths = {f'chr{i.id}': i.length for i in species.genome.chromosomes if i.id not in ["X", "Y", "MT"]}
chr_probs = np.array(list(chr_lengths.values())) / sum(chr_lengths.values())
chr_names = list(chr_lengths.keys())

make_chrom_annots()

#%%
# sample 20 Mb segments for num_sims simulations
def make_annots(seed, out_path, num_sims):
    np.random.seed(seed)
    segment_details = []
    for i in range(1, num_sims + 1):
        while True:
            # sample chromosome segment
            c, s, e = sample_chrom_segment(chr_lengths, chr_names, chr_probs)

            # read annots of segment
            chrom_annots = pd.read_csv(f"./sim_files/{c}_annotations.csv")
            segment_annots = subset_annotations(chrom_annots, s, e)

            # make sure there are no "exclude"s in segment annots
            if "exclude" in segment_annots['type'].values:
                continue

            # get recomb rates
            segment = species.get_contig(chromosome=c, left=s, right=e, genetic_map="HapMapII_GRCh38")
            recomb_rate = segment.recombination_map.rate

            # make sure there are no nans in recomb rates
            if np.sum(np.isnan(recomb_rate)) == 0:
                break

        # make the annots start at 0 and end at length
        segment_annots['start'] = segment_annots['start'] - s
        segment_annots['end'] = segment_annots['end'] - s - 1  # end is inclusive in slim
        segment_annots.to_csv(f'{out_path}/sim_{i}_annots.csv', index=False)

        # save recomb rates to file
        recomb_pos = segment.recombination_map.position.astype(int)
        # slim only uses recomb rates at the end of each interval
        recomb_ends = recomb_pos[1:]
        recomb_df = pd.DataFrame({'end': recomb_ends, 'rate': recomb_rate})
        recomb_df.to_csv(f'{out_path}/sim_{i}_recomb.csv', index=False)

        # save segment details
        segment_details.append((c, s, e))

    # save segment details to file
    segment_details_df = pd.DataFrame(segment_details, columns=['chrom', 'start', 'end'])
    segment_details_df.to_csv(f'{out_path}/sim_segment_details.csv', index=False)


# os.makedirs("./sim_files/v2", exist_ok=True)
# make_annots(3079104, "./sim_files/v2", 20)

# os.makedirs("./sim_files/v5", exist_ok=True)
# make_annots(3914302, "./sim_files/v5", 20)

version = "v6"
os.makedirs(f"./sim_files/{version}", exist_ok=True)
make_annots(67458910, f"./sim_files/{version}", 20)

version = "v7"
os.makedirs(f"./sim_files/{version}", exist_ok=True)
make_annots(543589, f"./sim_files/{version}", 20)

#%%
# make distribution of exon and cnc percentages
import matplotlib.pyplot as plt


def plot_element_type_perc_dist(annots_path):
    exon_percs, cnc_percs, qui_percs, bkgd_percs = [], [], [], []
    total_len = 20_000_000
    for i in range(1, 11):
        annots = pd.read_csv(os.path.join(annots_path, f'sim_{i}_annots.csv'))
        exons, cnc = annots.loc[annots['type'] == 'exon'], annots.loc[annots['type'] == 'cnc']
        qui, bkgd = annots.loc[annots['type'] == 'qui'], annots.loc[annots['type'] == 'bkgd']
        exon_len = sum(exons['end'] - exons['start'])
        cnc_len = sum(cnc['end'] - cnc['start'])
        qui_len = sum(qui['end'] - qui['start'])
        bkgd_len = sum(bkgd['end'] - bkgd['start'])
        exon_percs.append(exon_len / total_len * 100)
        cnc_percs.append(cnc_len / total_len * 100)
        qui_percs.append(qui_len / total_len * 100)
        bkgd_percs.append(bkgd_len / total_len * 100)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharey=True)
    axes = axes.ravel()
    perc_lists = [exon_percs, cnc_percs, qui_percs, bkgd_percs]
    colors = ['orangered', 'gold', 'lightblue', 'lightgreen']
    names = ['CDS', 'CNC', 'Quiescent', 'Background']
    names_w_means = [f'{n} (mean: {np.mean(p):.2f})' for n, p in zip(names, perc_lists)]
    for ax, percs, color, percs_name in zip(axes, perc_lists, colors, names_w_means):
        ax.hist(percs, bins=20, color=color, edgecolor='black', linewidth=1.2)
        ax.axvline(x=np.mean(percs), color='black', linestyle='--')
        ax.set_title(percs_name)
        ax.set_xlabel("Percent")
        ax.set_ylabel("Count")
        ax.set_yticks(np.arange(int(ax.get_ylim()[1]) + 1))
    fig.tight_layout()
    fig.show()

