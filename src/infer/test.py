import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import random
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from training.utils import Train
from shared.GCN_model import GCNModel
from shared.graph_construction import PrepareData
random.seed(1)

def create_structs(ads_type):
    obj = PrepareData()
    obj.create_structs_testing(ads_type)

def create_graphs(ads_type):
    obj = PrepareData()
    obj.create_graphs_testing(ads_type)

def test_models(ads_type, batch_size=1000):
    obj = Train()
    testing_data = torch.load(f'./test_unseen_data/output_files/graphs_testing/all_graphs_{ads_type}.pkl')
    dataloader = DataLoader(testing_data, batch_size, shuffle = False)
    for model_num in range(1, 5):
        model = GCNModel(obj.input_size, obj.hidden_size, obj.output_size)
        model.load_state_dict(torch.load(f'training/output_files/model_{model_num}.pt'))
        model.eval()
        preds = []
        for counter, data in enumerate(dataloader):
            if counter % 100 == 0:
                print(counter)
            pred = model(data.x, data.edge_index, data.batch)
        preds.extend([round(float(arr[0]), 5) for arr in pred])
        print(len(preds))
        np.save(f'./test_unseen_data/output_files/preds_testing/preds_{ads_type}_model_{model_num}.npy', preds)

def calc_uncertainty(ads_type):
    preds = []
    for model_num in range(1, 5):
        pred_file = np.load(f'./test_unseen_data/output_files/preds_testing/preds_{ads_type}_model_{model_num}.npy')
        preds.append(list(pred_file))
    preds = np.array(preds)
    std_values = np.std(preds, axis=0)
    print(np.max(std_values), np.min(std_values))
    return std_values

def load_structs(ads_type):
    structs = np.load(f'./test_unseen_data/output_files/structs_testing/all_structs_{ads_type}.npy')
    return structs

def choose_structs_labeling(std_values, structs, threshold=0.1, num_labeling=30):
    thresh = threshold # higher than this threshold is considered inaccurate
    candidate_struct = [struct for std, struct in zip(std_values, structs) if std > thresh]
    candidate_std = [std for std, struct in zip(std_values, structs) if std > thresh]
    inacc = 100 * len(candidate_struct) / len(structs)
    print(f'Accurate structures: {100-inacc}%\nInaccurate structures: {inacc}%')
    samples = random.sample(candidate_struct, num_labeling)
    return samples

def save_symbols(samples, ads_type):
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt', 'Co', 'Ga', 'Ni', 'Zn']
    dict_rep = {
            'CO': [[36], [37, 38, 39, 41, 42, 43], [27, 33, 34]],
            'H_FCC': [[36, 37, 39], [40, 41, 43], [27, 28, 34]],
            'H_HCP': [[37, 39, 40], [36, 38, 42], [28]]
            }
    all_symbols = []
    for sample in samples:
        selec1 = [i*[metals[counter % 9]] if i!=0 else [] for counter, i in enumerate(sample)]
        selec2 = [item for sublist in selec1 for item in sublist if sublist]
        symbols = random.choices(metals, k=45) # 45 = 3 * 3 * 5
        counter = 0
        for indices in dict_rep[ads_type]:
            for ind in indices:
                symbols[ind] = selec2[counter]
                counter += 1
        all_symbols.append(symbols)
    return all_symbols

def main(show_structs=False):
    '''
    ads_type = 'H_HCP'
    create_structs(ads_type)
    '''
    lims1  = {'CO': (0.67, 0.8), 'H_FCC': (0.37, 0.5),  'H_HCP': (0.82, 0.95)}
    lims2  = {'CO': (-0.01, 0.23), 'H_FCC': (-0.01, 0.13), 'H_HCP': (-0.01, 0.33)}
    ticks1 = {'CO': (0.7, 0.81), 'H_FCC': (0.4, 0.51), 'H_HCP': (0.85, 0.96)}
    ticks2 = {'CO': (0, 0.21), 'H_FCC': (0, 0.11), 'H_HCP': (0, 0.31)}
    spacing = {'CO': 0.05, 'H_FCC': 0.05, 'H_HCP':0.1}
    bins = {'CO': 1500, 'H_FCC': 1500, 'H_HCP': 750}
    label_pos = {'CO': 0.55, 'H_FCC': 0.55, 'H_HCP': 0.55}
    ads_types = ['CO', 'H_FCC', 'H_HCP']
    alpha = 0.4
    dict_symbols = {}
    for ads_type in ads_types:
        print(ads_type)
        structs = load_structs(ads_type)
        std_values = calc_uncertainty(ads_type)
        #plt.figure(figsize=(50,10))
        lim1 = lims1[ads_type]
        lim2 = lims2[ads_type]
        tick1 = ticks1[ads_type]
        tick2 = ticks2[ads_type]
        rat = (lim1[1] - lim1[0]) / (lim2[1] - lim2[0])
        # Create two subplots with a gap
        fig, (ax2, ax1) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, rat]}, figsize=(15,5))
        fig.subplots_adjust(wspace=0.04)  # adjust space between axes
        ax1.hist(std_values, bins=bins[ads_type], density=True, alpha=alpha, color='blue')
        ax2.hist(std_values, bins=bins[ads_type], density=True, alpha=alpha, color='blue')
        # zoom-in / limit the view to different portions of the data
        ax1.set_xlim(lim1[0], lim1[1])  # outliers only
        ax2.set_xlim(lim2[0], lim2[1])  # most of the data
        ax1.spines.left.set_visible(False)
        ax1.tick_params(left=False)  # don't put tick labels at the top
        ax1.set_xticks(np.arange(tick1[0], tick1[1], spacing[ads_type]))
        ax1.tick_params(axis='x', labelsize=15)
        ax2.set_xticks(np.arange(tick2[0], tick2[1], spacing[ads_type]))
        ax2.tick_params(axis='x', labelsize=15)
        ax2.spines.right.set_visible(False)
        ax2.tick_params(left=False)  # don't put tick labels at the top
        ax2.tick_params(labelleft=False)
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 0], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([1, 1], [1, 0], transform=ax2.transAxes, **kwargs)
        #ax2.yaxis.tick_right()
        # Show the percentage of data within each range
        percentages = [0, 25, 50, 75, 100]
        percentile_values = np.percentile(std_values, percentages)
        for p, val in zip(percentages, percentile_values):
            if (val > lim1[0]) and (val < lim1[1]):
                ax1.axvline(val, color='red', linestyle='dashed', linewidth=2)
                ax1.text(val + 0.003, 1, f'{p}%', rotation=90, color='red', fontsize=20)
            if (val > lim2[0]) and (val < lim2[1]):
                ax2.axvline(val, color='red', linestyle='dashed', linewidth=2)
                ax2.text(val + 0.003, 1, f'{p}%', rotation=90, color='red', fontsize=20)
        # Set a common xlabel at the middle
        #middle_position = (ax1.get_position().x0 + ax2.get_position().x1) / 2
        fig.text(label_pos[ads_type]-0.015, 0.025, r'$\epsilon_m$', ha='center', va='center', fontsize=22)
        fig.text(label_pos[ads_type]+0.015, 0.025, r'$(\mathrm{eV})$', ha='center', va='center', fontsize=17)
        #r'$\epsilon_m (eV)$'
        #plt.xlabel(r'$\epsilon_m$ (eV)', fontsize=20)
        #plt.tight_layout()
        plt.savefig(f'./test_unseen_data/output_files/plots/{ads_type}_e_dist.png')
        plt.clf()
        #samples = choose_structs_labeling(std_values, structs)
        #dict_symbols[ads_type] = save_symbols(samples, ads_type)
    #with open('output_files/structs_labeling/labeling_symbols.pkl', 'wb') as pickle_file:
        #pickle.dump(dict_symbols, pickle_file)

    '''
        if show_structs:
            for counter, struct in enumerate(samples):
                print(f'Structure number: {counter}')
                for i in range(0, 9):
                    for _ in range(int(struct[i])):
                        print(metals[i % 9], end='\t')
                print('\n')
                for i in range(9, 18):
                    for _ in range(int(struct[i])):
                        print(metals[i % 9], end='\t')
                print('\n')
                for i in range(18, 27):
                    for _ in range(int(struct[i])):
                        print(metals[i % 9], end='\t')
                print('\n')
    '''

if __name__ == '__main__':
    main()
