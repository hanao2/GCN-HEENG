import os
import itertools
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/input/HEA_properties.csv')

class PrepareData:
    def __init__(self):
        '''
        Initializes the PrepareData class with properties and computes statistical parameters.
        '''
        self.num_metals = 9
        self.df_property = pd.read_csv(data_path)
        self.atom_ind = {'AgAuCuPdPt': [0, 1, 2, 3, 4], 'CoCuGaNiZn': [5, 2, 6, 7, 8]}
        self.desired_prop = [1, 2, 3, 4, 5, 6, 7]
        self.dict_prop = {key:[] for key in self.df_property.columns[self.desired_prop]}
        self.num_feat = len(self.desired_prop)
        self.metal_mean = np.array(self.df_property.iloc[:, self.desired_prop].mean(axis=0))
        self.metal_std   = np.array(self.df_property.iloc[:, self.desired_prop].std(axis=0))
        self.edge_index = {}
        self.structs = {}
        self.ads_types = {'CO', 'H_FCC', 'H_HCP'}
        self.natoms_list = {'CO': [1, 6, 3], 'H_FCC': [3, 3, 3], 'H_HCP': [3, 3, 1]}
        self.create_edge_indices()
        self.fill_inds = {
                self.ads_type: (
                    [0] * self.natoms_list[ads_type][0] +
                    [1] * self.natoms_list[ads_type][1] +
                    [2] * self.natoms_list[ads_type][2]
                    )
                for ads_type in self.ads_types.keys()
                }
        self.inds_combined = {
                'CO': [0, 0, 0] + [1, 0, 0],
                'H_FCC': [0, 0, 0] + [0, 1, 0],
                'H_HCP': [0, 0, 0] + [0, 0, 1],
                }

    def prepare_property(self, atom_indices: list) -> dict:
        for dict_key, df_col in zip(self.dict_prop.keys(), self.df_property.columns[self.desired_prop]):
            self.dict_prop[dict_key] = [self.df_property[df_col][i] for i in atom_indices]
        return self.dict_prop

    def first_layer(self, idx: int) -> list:
        metals_Co = ['Co', 'Cu', 'Ga', 'Ni', 'Zn'] # ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
        combs = [''.join(comb) for comb in itertools.combinations_with_replacement(metals_CO, 3)]
        metal_str = combs[idx]
        metal_1, metal_2, metal_3 = metal_str[:2], metal_str[2:4], metal_str[4:]
        list_metal_idx = [metals_Co.index(metal_1), metals_Co.index(metal_2), metals_Co.index(metal_3)]
        return list_metal_idx

    def prepare_features(self, ads_type, dict_df, atom_indices):
        alloy_prop = self.prepare_property(atom_indices)
        features = {}
        if ads_type == 'CO':
            for key in dict_df:
                df = dict_df[key]
                all_feats = []
                for index, row in df.iterrows():
                    feats = []
                    for counter, i in enumerate(row[:5]):
                        for _ in range(int(i)):
                            feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    for counter, i in enumerate(row[5:10]):
                        for _ in range(int(i)):
                            feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    for counter, i in enumerate(row[10:15]):
                        for _ in range(int(i)):
                            feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    feats.append(row[15])
                    all_feats.append(feats)
                features[key] = pd.DataFrame(all_feats)

        elif ads_type in ['H_FCC', 'H_HCP']:
            for key in dict_df:
                df = dict_df[key]
                all_feats = []
                for index, row in df.iterrows():
                    idx = row[:35].index[row[:35] == 1].tolist()
                    metals_green = self.first_layer(idx[0]) 
                    feats = []
                    for counter in metals_green:
                        feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    for counter, i in enumerate(row[35:40]):
                        for _ in range(int(i)):
                            feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    for counter, i in enumerate(row[40:45]):
                        for _ in range(int(i)):
                            feats.extend([alloy_prop[j][counter] for j in alloy_prop.keys()])
                    feats.append(row[45])
                    all_feats.append(feats)
                features[key] = pd.DataFrame(all_feats)

        return features
 
    def create_each_graph_training(self, ads_type, data):
        features = []
        for counter, fill_ind in enumerate(self.fill_inds[ads_type]):
            inds = self.inds_combined[ads_type]
            inds[fill_ind] = 1
            features.append(np.append(data[counter * self.num_feat : (counter + 1) * self.num_feat], inds))
        features_each = torch.FloatTensor(features)
        y_each = torch.FloatTensor([data[(counter + 1) * self.num_feat]])
        
        return features_each, y_each

    def create_edge_indices(self):
        #connectivity: all the nodes connected together
        for ads_type in self.ads_types.keys():
            edges = [(i, j) for i in range(torch.sum(self.natoms_list[ads_type])) for j in range(self.natoms[ads_type]) if i != j]
            self.edge_index[ads_type] = torch.tensor(edges, dtype=torch.long).T

    def data_normalization_training(self, data): #standardscaling
        for counter, i in enumerate(data.columns[:-1]):
            data[i] = (data[i] - self.metal_mean[counter % self.num_feat]) / self.metal_std[counter % self.num_feat]
        return data

    def create_graphs_training(self, ads_type, data):
        data = self.data_normalization_training(data)
        graph_list = []
        
        for index, row in data.iterrows():
            features_each, y_each = self.create_each_graph_training(ads_type, row.values)
            graph_data = Data(x=features_each, y=y_each, edge_index=self.edge_index[ads_type])
            graph_list.append(graph_data)
            
        return graph_list

    def prepare_data_training(self, ads_type):
        if ads_type == 'CO':
            ads = 'CO'
        elif ads_type == 'H_FCC':
            ads = 'H_fcc'
        elif ads_type == 'H_HCP':
            ads = 'H_hcp'

        data_dict = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        data_dict_org = {}  # original
        data_dict_feat = {} # with features
        data_dict_without_feat = {}
        data_dict_with_feat = {}
        for key in data_dict.keys():
            for size in ['2x2', '3x3']:
                data_dict[key][size] = pd.read_csv(f'../data/input/{key}_{size}_{ads}.csv', header=None)
        for key in data_dict.keys():
            data_dict_without_feat[key] = pd.concat([data_dict[key]['2x2'], data_dict[key]['3x3']])
        for key in data_dict.keys():
            data_dict_with_feat[key] = self.prepare_features(ads_type, data_dict[key], self.atom_ind[key])
        for key in data_dict.keys():
            data_dict_feat[key] = pd.concat([data_dict_with_feat[key]['2x2'], data_dict_with_feat[key]['3x3']])
        return data_dict_feat, data_dict_without_feat

    def pure_indices(self):
        indices_each_alloy = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        DFT_energies = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        for ads_type in ['CO', 'H_FCC', 'H_HCP']:
            data_dict_feat, data_dict_no_feat = self.prepare_data_training(ads_type)
            for key in indices_each_alloy.keys():
                if ads_type == 'CO':
                    ind_check = [(0+i, 5+i, 10+i) for i in range(5)]
                    a, b, c = [1, 6, 3]
                    e_index = 15
                elif ads_type == 'H_FCC':
                    elems = [0, 15, 25, 31, 34]
                    ind_check = [(elems[i], 35+i, 40+i) for i in range(5)]
                    a, b, c = [1, 3, 3]
                    e_index = 45
                else:
                    elems = [0, 15, 25, 31, 34]
                    ind_check = [(elems[i], 35+i, 40+i) for i in range(5)]
                    a, b, c = [1, 3, 1]
                    e_index = 45
                df = data_dict_no_feat[key]
                indices_each_alloy[key][ads_type] = []
                for j in ind_check:
                    indices_each_alloy[key][ads_type].extend(df.index[(df[j[0]] > a-1) & (df[j[1]] > b-1) & (df[j[2]] > c-1)].tolist())
        return indices_each_alloy

    def data_all_adsorbates(self):
        graphs_all = {}
        graphs_each_alloy = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        for ads_type in ['CO', 'H_FCC', 'H_HCP']:
            data_dict_feat, data_dict_no_feat = self.prepare_data_training(ads_type)
            for key in graphs_each_alloy.keys():
                graphs_each_alloy[key][ads_type] = self.create_graphs_training(ads_type, data_dict_feat[key])
            graphs_all[ads_type] = graphs_each_alloy['AgAuCuPdPt'][ads_type] + graphs_each_alloy['CoCuGaNiZn'][ads_type]
        return graphs_all, graphs_each_alloy

    #from this point on is related to the exploration part
    def partitions(self, n, b):
        mask = np.identity(b, dtype=int)
        for c in combinations_with_replacement(mask, n):
            yield sum(c)

    def layer_combinations(self, ads_type):
        nums = self.natoms_list[ads_type]

        layer1_combs = list(self.partitions(nums[0], self.num_metals))
        layer2_combs = list(self.partitions(nums[1], self.num_metals))
        layer3_combs = list(self.partitions(nums[2], self.num_metals))
        structs = []
        for i in layer1_combs:
            for j in layer2_combs:
                for k in layer3_combs:
                    structs.append(list(i) + list(j) + list(k))
        return structs
 
    def create_each_graph_testing(self, ads_type, data):
        features = []
        if ads_type == 'CO':

        for counter, fill_ind in enumerate(self.fill_inds[ads_type]):
            inds = self.inds_combined[ads_type]
            inds[fill_ind] = 1
            features.append(np.append(data[counter, :], inds))
        return features

    def create_graphs_testing(self, ads_type):
        structs = self.layer_combinations(ads_type)
        graphs = []
        for counter, struct in enumerate(structs):
                print(counter)
                feats = []
                for i in range(0, len(struct)):
                    for _ in range(struct[i]):
                        feats.extend([self.dict_prop[j][i % self.num_metals] for j in self.dict_prop.keys()])

                data = torch.FloatTensor(feats)
                data = (data - self.metal_mean) / self.metal_std
                features = self.create_each_graph_testing(ads_type, data)
                features_each = torch.FloatTensor(features)
                graph = Data(x=features_each, edge_index=self.edge_index[ads_type])
                all_graphs.append(graph)
        torch.save(all_graphs, f'./output_files/all_graphs_{ads_type}.pkl')
