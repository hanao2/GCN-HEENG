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
from shared.GCN_model import GCNModel
from shared.graph_construction import PrepareData
from training.utils import Train

def data_each_alloy():
    data = PrepareData()
    graphs, graphs_each = data.data_all_adsorbates()
    ind_pure = data.pure_indices()
    return graphs_each, ind_pure

def test_model(data, ads_type, batch_size=100):
    obj = Train()
    testing_data = data[ads_type]
    dataloader = DataLoader(testing_data, batch_size, shuffle = False)
    all_preds = []
    for model_num in range(1, 5):
        model = GCNModel(obj.input_size, obj.hidden_size, obj.output_size)
        model.load_state_dict(torch.load(f'./training/output_files/model_{model_num}.pt'))
        model.eval()
        preds = []
        labels = []
        for counter, data in enumerate(dataloader):
            #if counter % 100 == 0:
            #    print(counter)
            pred = model(data.x, data.edge_index, data.batch)
            lbl = data.y
            preds.extend([round(float(arr[0]), 5) for arr in pred])
            labels.extend(lbl)
        all_preds.append(preds)
    output = np.mean(all_preds, axis=0)
    gts = labels
    return output, gts
        #np.save(f'./output_files/preds_testing/preds_{ads_type}_model_{model_num}.npy', preds)

def main():
    #GCN versus DFT plots
    fontsize=15
    bounds = {'CO': (-2, 0.5), 'H_FCC': (-0.8, 0.8), 'H_HCP': (-0.8, 0.8)}
    colors = ['c', 'r']
    GCN_energies = {'CoCuGaNiZn': {}, 'AgAuCuPdPt': {}}
    GCN_energies_pure = {'CoCuGaNiZn': {}, 'AgAuCuPdPt': {}}
    DFT_energies = {'CoCuGaNiZn': {}, 'AgAuCuPdPt': {}}
    DFT_energies_pure = {'CoCuGaNiZn': {}, 'AgAuCuPdPt': {}}
    for ads_type in ['CO', 'H_FCC', 'H_HCP']:
        plt.figure(figsize=(6,6))
        for counter, key in enumerate(['CoCuGaNiZn', 'AgAuCuPdPt']):
            data, pure_ind = data_each_alloy()
            preds, labels = test_model(data[key], ads_type)
            GCN_energies[key][ads_type] = preds
            DFT_energies[key][ads_type] = labels
            plt.scatter(labels, preds, marker='o', linewidth=0.01, color=colors[counter], alpha=0.5, label=key)

            if len(pure_ind[key][ads_type]) <= 5:
                GCN_energies_pure[key][ads_type] = [preds[i] for i in pure_ind[key][ads_type]]
                DFT_energies_pure[key][ads_type] = [labels[i] for i in pure_ind[key][ads_type]]

            elif len(pure_ind[key][ads_type]) > 5 and ads_type == 'H_FCC':
                GCN_energies_pure[key][ads_type] = []
                GCN_energies_pure[key][ads_type].extend([preds[i] for i in pure_ind[key][ads_type][:2]])
                GCN_energies_pure[key][ads_type].append(0.5 * (preds[pure_ind[key][ads_type][2]] + preds[pure_ind[key][ads_type][3]])) 
                GCN_energies_pure[key][ads_type].extend([preds[i] for i in pure_ind[key][ads_type][4:]])
                DFT_energies_pure[key][ads_type] = []
                DFT_energies_pure[key][ads_type].extend([labels[i] for i in pure_ind[key][ads_type][:2]])
                DFT_energies_pure[key][ads_type].append(0.5 * (labels[pure_ind[key][ads_type][2]] + labels[pure_ind[key][ads_type][3]])) 
                DFT_energies_pure[key][ads_type].extend([labels[i] for i in pure_ind[key][ads_type][4:]])

            elif len(pure_ind[key][ads_type]) > 5 and ads_type == 'H_HCP':
                GCN_energies_pure[key][ads_type] = []
                GCN_energies_pure[key][ads_type].append(0.5 * (preds[pure_ind[key][ads_type][0]] + preds[pure_ind[key][ads_type][1]])) 
                GCN_energies_pure[key][ads_type].extend([preds[i] for i in pure_ind[key][ads_type][2:]])
                DFT_energies_pure[key][ads_type] = []
                DFT_energies_pure[key][ads_type].append(0.5 * (labels[pure_ind[key][ads_type][0]] + labels[pure_ind[key][ads_type][1]])) 
                DFT_energies_pure[key][ads_type].extend([labels[i] for i in pure_ind[key][ads_type][2:]])

        x = np.linspace(bounds[ads_type][0], bounds[ads_type][1], 100)
        plt.plot(x, x,'k-', alpha=.75)
        plt.plot(x, x+0.1,'k--', alpha=.75)
        plt.plot(x, x-0.1,'k--', alpha=.75)
        plt.ylabel('\u0394' + r'$E^{GCN}$' + ' (eV)', fontsize=fontsize)
        plt.xlabel('\u0394' + r'$E_{CO}^{DFT}$' + ' (eV)', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        plt.minorticks_on()
        plt.xlim(bounds[ads_type][0], bounds[ads_type][1], 10)
        plt.ylim(bounds[ads_type][0], bounds[ads_type][1], 10)
        plt.tight_layout()
        plt.savefig(f'./test_training_data/output_files/GCN_vs_DFT_{ads_type}.png', dpi=1000)
        plt.clf()

    #print(DFT_energies_pure)
    eng_all = GCN_energies_pure['AgAuCuPdPt']['H_HCP'] + GCN_energies_pure['CoCuGaNiZn']['H_HCP']
    float_list = [float(tensor) for tensor in eng_all]
    names = ['Ag', 'Au', 'Cu', 'Pd', 'Pt', 'Co', 'Cu', 'Ga', 'Ni', 'Zn']
    # Combine the two lists into pairs
    combined_list = list(zip(float_list, names))
    # Sort the pairs based on the numeric values
    sorted_list = sorted(combined_list, key=lambda x: x[0])
    # Extract the sorted names
    sorted_names = [pair[1] for pair in sorted_list]
    print(f'CO data: {np.sort(float_list)}')
    print(f'CO data: {sorted_names}')
    #print(eng_all)
    #print(GCN_energies_pure)

    #GCN and DFT histogram plots
    fontsize=12
    bins= {'CO': np.linspace(-2, 0.5, 125), 'H_FCC': np.linspace(-0.8, 0.8, 80) , 'H_HCP': np.linspace(-0.8, 0.8, 80)}
    xlim = {'CO': (-2, .5), 'H_FCC': (-0.8, 0.81), 'H_HCP': (-0.8, 0.81)}
    y_th = 100
    n_points = 10
    linewidth = 1

    for ads_type in ['CO', 'H_FCC', 'H_HCP']:
        plt.figure(figsize=(6,4))
        plt.subplot(2,1,1)
        plt.hist(list(GCN_energies['CoCuGaNiZn'][ads_type]) + list(GCN_energies['AgAuCuPdPt'][ads_type]), bins=bins[ads_type], color='gray', alpha=0.5)
        plt.hist(GCN_energies['CoCuGaNiZn'][ads_type], bins=bins[ads_type], color='c', alpha=0.5)
        plt.hist(GCN_energies['AgAuCuPdPt'][ads_type], bins=bins[ads_type], color='r', alpha=0.5)
        for key in GCN_energies.keys():
            for counter, energy in enumerate(GCN_energies_pure[key][ads_type]):
                metal = key[2*counter:2*(counter+1)]
                if metal == 'Cu':
                    linestyle = '--'
                    color = 'k'
                    eng = 0.5 * (GCN_energies_pure['AgAuCuPdPt'][ads_type][2] + GCN_energies_pure['CoCuGaNiZn'][ads_type][1])
                    plt.plot(n_points * [eng], np.linspace(0, y_th, n_points), color=color, linestyle=linestyle, linewidth=linewidth)
                else:
                    linestyle = '--'
                    color = 'k'
                    plt.plot(n_points * [energy], np.linspace(0, y_th, n_points), color=color, linestyle=linestyle, linewidth=linewidth)
        plt.xlim(xlim[ads_type])
        plt.ylim((0,y_th))
        plt.xticks([])
        plt.yticks(np.arange(0,y_th+1, 50))
        plt.minorticks_on()

        plt.subplot(2,1,2)
        plt.hist(list(DFT_energies['CoCuGaNiZn'][ads_type]) + list(DFT_energies['AgAuCuPdPt'][ads_type]), bins=bins[ads_type], color='gray', alpha=0.5)
        plt.hist(DFT_energies['CoCuGaNiZn'][ads_type], bins=bins[ads_type], color='c', alpha=0.5)
        plt.hist(DFT_energies['AgAuCuPdPt'][ads_type], bins=bins[ads_type], color='r', alpha=0.5)
        for key in DFT_energies.keys():
            for counter, energy in enumerate(DFT_energies_pure[key][ads_type]):
                metal = key[2*counter:2*(counter+1)]
                if metal == 'Cu':
                    linestyle = '--'
                    color = 'k'
                    eng = 0.5 * (DFT_energies_pure['AgAuCuPdPt'][ads_type][2] + DFT_energies_pure['CoCuGaNiZn'][ads_type][1])
                    plt.plot(n_points * [eng], np.linspace(0, y_th, n_points), color=color, linestyle=linestyle, linewidth=linewidth)
                else:
                    linestyle = '--'
                    color = 'k'
                    plt.plot(n_points * [energy], np.linspace(0, y_th, n_points), color=color, linestyle=linestyle, linewidth=linewidth)

        plt.xlim(xlim[ads_type])
        plt.ylim((0,y_th))
        plt.yticks(np.arange(0,y_th+1, 50))
        plt.minorticks_on()
        plt.xticks([])
        plt.xlabel('\u0394' + r'$E_{CO}^{DFT}$' + ' (eV)', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'./test_training_data/output_files/hist_{ads_type}.png', dpi=1000)
        plt.clf()

if __name__ == '__main__':
    main()
