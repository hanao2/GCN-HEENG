import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

class PrepareData:
    def __init__(self):
        self.num_metals = 9
        self.df_property = pd.read_csv('../data/input/HEA_properties.csv')
        self.atom_ind = {'AgAuCuPdPt': [0, 1, 2, 3, 4], 'CoCuGaNiZn': [5, 2, 6, 7, 8]}
        self.desired_prop = [1, 2, 3, 4, 5, 6, 7]
        self.dict_prop = {key:[] for key in self.df_property.columns[self.desired_prop]}
        self.num_feat = len(self.desired_prop)
        self.metal_mean = np.array(self.df_property.iloc[:, self.desired_prop].mean(axis=0))
        self.metal_std   = np.array(self.df_property.iloc[:, self.desired_prop].std(axis=0))
        self.edge_index = {}
        self.structs = {}
        self.create_edge_indices()

    def prepare_property(self, atom_indices):
        for dict_key, df_col in zip(self.dict_prop.keys(), self.df_property.columns[self.desired_prop]):
            self.dict_prop[dict_key] = [self.df_property[df_col][i] for i in atom_indices]
        return self.dict_prop

    def first_layer(self, idx):
        metals_Co = ['Co', 'Cu', 'Ga', 'Ni', 'Zn'] # ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
        combs = ['CoCoCo', 'CoCoCu', 'CoCoGa', 'CoCoNi', 'CoCoZn',
               'CoCuCu', 'CoCuGa', 'CoCuNi', 'CoCuZn', 'CoGaGa',
               'CoGaNi', 'CoGaZn', 'CoNiNi', 'CoNiZn', 'CoZnZn',
               'CuCuCu', 'CuCuGa', 'CuCuNi', 'CuCuZn', 'CuGaGa',
               'CuGaNi', 'CuGaZn', 'CuNiNi', 'CuNiZn', 'CuZnZn',
               'GaGaGa', 'GaGaNi', 'GaGaZn', 'GaNiNi', 'GaNiZn',
               'GaZnZn', 'NiNiNi', 'NiNiZn', 'NiZnZn', 'ZnZnZn']
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
        if ads_type == 'CO':
            features = []
            features.append(np.append(data[0*self.num_feat:1*self.num_feat], [1, 0, 0, 1, 0, 0]))
            features.append(np.append(data[1*self.num_feat:2*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[2*self.num_feat:3*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[3*self.num_feat:4*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[4*self.num_feat:5*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[5*self.num_feat:6*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[6*self.num_feat:7*self.num_feat], [0, 1, 0, 1, 0, 0]))
            features.append(np.append(data[7*self.num_feat:8*self.num_feat], [0, 0, 1, 1, 0, 0]))
            features.append(np.append(data[8*self.num_feat:9*self.num_feat], [0, 0, 1, 1, 0, 0]))
            features.append(np.append(data[9*self.num_feat:10*self.num_feat],[0, 0, 1, 1, 0, 0]))
            features_each = torch.FloatTensor(features)
            y_each = torch.FloatTensor([data[10*self.num_feat]])

        elif ads_type == 'H_FCC':
            features = []
            features.append(np.append(data[0*self.num_feat:1*self.num_feat], [1, 0, 0, 0, 1, 0]))
            features.append(np.append(data[1*self.num_feat:2*self.num_feat], [1, 0, 0, 0, 1, 0]))
            features.append(np.append(data[2*self.num_feat:3*self.num_feat], [1, 0, 0, 0, 1, 0]))
            features.append(np.append(data[3*self.num_feat:4*self.num_feat], [0, 1, 0, 0, 1, 0]))
            features.append(np.append(data[4*self.num_feat:5*self.num_feat], [0, 1, 0, 0, 1, 0]))
            features.append(np.append(data[5*self.num_feat:6*self.num_feat], [0, 1, 0, 0, 1, 0]))
            features.append(np.append(data[6*self.num_feat:7*self.num_feat], [0, 0, 1, 0, 1, 0]))
            features.append(np.append(data[7*self.num_feat:8*self.num_feat], [0, 0, 1, 0, 1, 0]))
            features.append(np.append(data[8*self.num_feat:9*self.num_feat], [0, 0, 1, 0, 1, 0]))
            features_each = torch.FloatTensor(features)
            y_each = torch.FloatTensor([data[9*self.num_feat]])

        elif ads_type == 'H_HCP':
            features = []
            features.append(np.append(data[0*self.num_feat:1*self.num_feat], [1, 0, 0, 0, 0, 1]))
            features.append(np.append(data[1*self.num_feat:2*self.num_feat], [1, 0, 0, 0, 0, 1]))
            features.append(np.append(data[2*self.num_feat:3*self.num_feat], [1, 0, 0, 0, 0, 1]))
            features.append(np.append(data[3*self.num_feat:4*self.num_feat], [0, 1, 0, 0, 0, 1]))
            features.append(np.append(data[4*self.num_feat:5*self.num_feat], [0, 1, 0, 0, 0, 1]))
            features.append(np.append(data[5*self.num_feat:6*self.num_feat], [0, 1, 0, 0, 0, 1]))
            features.append(np.append(data[6*self.num_feat:7*self.num_feat], [0, 0, 1, 0, 0, 1]))
            features_each = torch.FloatTensor(features)
            y_each = torch.FloatTensor([data[7*self.num_feat]])
        
        return features_each, y_each

    def create_edge_indices(self):
        #connectivity: all the nodes connected together
        self.edge_index['CO'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 8, 8, 9, 9],
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 2, 3, 4, 5, 6, 1, 3, 4, 5, 6, 1, 2, 4, 5, 6, 1, 2, 3, 5, 6, 1, 2, 3, 4, 6, 1, 2, 3, 4, 5, 8, 9, 7, 9, 7, 8]], dtype=torch.long)
        
        self.edge_index['H_FCC'] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8],
                                        [1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 4, 5, 6, 7, 8, 0, 1, 3, 4, 5, 6, 7, 8, 0, 1, 2, 4, 5, 6, 7, 8, 0, 1, 2, 3, 5, 6, 7, 8, 0, 1, 2, 3, 4, 6, 7, 8, 0, 1, 2, 3, 4, 5, 7, 8, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        
        self.edge_index['H_HCP'] = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
                                        [1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6, 0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5]], dtype=torch.long)

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
        #self.data_AgAuCuPdPt = pd.concat([data_dict_with_features['AgAuCuPdPt']['2x2'], data_dict_with_features['AgAuCuPdPt']['3x3'])
        #self.data_CoCuGaNiZn = pd.concat([data_dict_with_features['CoCuGaNiZn']['2x2'], data_dict_with_features['CoCuGaNiZn']['3x3'])
        #Combine data
        #original
        #for size in ['2x2', '3x3']:
        #    data_dict_org[size] = pd.concat([data_dict['AgAuCuPdPt'][size], data_dict['CoCuGaNiZn'][size]])
        #with intrinsic feature
        #for size in ['2x2', '3x3']:
        #    data_dict_feat[size] = pd.concat([data_dict_with_features['AgAuCuPdPt'][size], data_dict_with_features['CoCuGaNiZn'][size]])
        #data = pd.concat([data_dict_feat['2x2'], data_dict_feat['3x3']])
        for key in data_dict.keys():
            data_dict_feat[key] = pd.concat([data_dict_with_feat[key]['2x2'], data_dict_with_feat[key]['3x3']])
        #data = pd.concat([data_dict_feat['2x2'], data_dict_feat['3x3']])
        #self.data_graphs_training[ads_type] = self.create_graphs_training(ads_type, data)
        return data_dict_feat, data_dict_without_feat

    def pure_indices(self):
        indices_each_alloy = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        DFT_energies = {'AgAuCuPdPt': {}, 'CoCuGaNiZn': {}}
        for ads_type in ['CO', 'H_FCC', 'H_HCP']:
            data_dict_feat, data_dict_no_feat = self.prepare_data_training(ads_type)
            for key in indices_each_alloy.keys():
                #print(data_dict_no_feat[key])
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
                #indices_each_alloy[key][ads_type] = [index for index, row in data_dict_no_feat[key].iterrows() for j in ind_check if int([j[0]])==a and int(row[j[1]])==b and int(row[j[2]])==c]
                #DFT_energies[key][ads_type] = [((data_dict_no_feat[key]).iloc[index])[e_index] for index in indices_each_alloy[key][ads_type]]
                #print(indices_each_alloy[key][ads_type])
        return indices_each_alloy
                #indices_each_alloy[key][ads_type] = [for i in range(9) if ]
            #graphs_all[ads_type] = graphs_each_alloy['AgAuCuPdPt'][ads_type] + graphs_each_alloy['CoCuGaNiZn'][ads_type]
        #return graphs_all, graphs_each_alloy

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
        if ads_type == 'CO':
            nums = [1, 6, 3]
        elif ads_type == 'H_FCC':
            nums = [3, 3, 3]
        elif ads_type == 'H_HCP':
            nums = [3, 3, 1]

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
                features.append(np.append(data[0, :], [1, 0, 0, 1, 0, 0]))
                features.append(np.append(data[1, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[2, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[3, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[4, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[5, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[6, :], [0, 1, 0, 1, 0, 0]))
                features.append(np.append(data[7, :], [0, 0, 1, 1, 0, 0]))
                features.append(np.append(data[8, :], [0, 0, 1, 1, 0, 0]))
                features.append(np.append(data[9, :], [0, 0, 1, 1, 0, 0]))

        if ads_type == 'H_FCC':
                features.append(np.append(data[0, :], [1, 0, 0, 0, 1, 0]))
                features.append(np.append(data[1, :], [1, 0, 0, 0, 1, 0]))
                features.append(np.append(data[2, :], [1, 0, 0, 0, 1, 0]))
                features.append(np.append(data[3, :], [0, 1, 0, 0, 1, 0]))
                features.append(np.append(data[4, :], [0, 1, 0, 0, 1, 0]))
                features.append(np.append(data[5, :], [0, 1, 0, 0, 1, 0]))
                features.append(np.append(data[6, :], [0, 0, 1, 0, 1, 0]))
                features.append(np.append(data[7, :], [0, 0, 1, 0, 1, 0]))
                features.append(np.append(data[8, :], [0, 0, 1, 0, 1, 0]))

        if ads_type == 'H_HCP':
                features.append(np.append(data[0, :], [1, 0, 0, 0, 0, 1]))
                features.append(np.append(data[1, :], [1, 0, 0, 0, 0, 1]))
                features.append(np.append(data[2, :], [1, 0, 0, 0, 0, 1]))
                features.append(np.append(data[3, :], [0, 1, 0, 0, 0, 1]))
                features.append(np.append(data[4, :], [0, 1, 0, 0, 0, 1]))
                features.append(np.append(data[5, :], [0, 1, 0, 0, 0, 1]))
                features.append(np.append(data[6, :], [0, 0, 1, 0, 0, 1]))

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
