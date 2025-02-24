import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer, GraphMaskExplainer
from shared.GCN_model import GCNModel
from training.utils import Train

class GCNExplanation:
    def __init__(self, algo):
        self.algo = algo # 'GNNExplainer', 'CaptumExplainer', 'GraphMaskExplainer'
        self.labels = ['Atomic number', 'Mass', 'Atomic radius', 'Covalent radius', 'Electronegativity', 'Electron_affinity', '1st ionization energy', '1st layer', '2nd layer', '3rd layer', 'CO adsorbate', 'H_FCC adsorbate', 'H_HCP adsorbate']
        self.read_data()

    def read_data(self):
        obj = Train()
        self.loader = {}
        self.loader['CO'] = obj.loader_CO
        self.loader['H_FCC'] = obj.loader_H_FCC
        self.loader['H_HCP'] = obj.loader_H_HCP
        self.train_loader = obj.trn_loader 
        self.input_size = obj.input_size
        self.hidden_size = obj.hidden_size
        self.output_size = obj.output_size

    def define_algorithm(self, algo):
        if algo == 'GNNExplainer':
            algorithm = GNNExplainer(epochs=300)
        elif algo == 'CaptumExplainer':
            algorithm = CaptumExplainer('IntegratedGradients') #'IntegratedGradients', 'Saliency', 'Deconvolution' and 'GuidedBackprop'results weren't good. 
        elif algo == 'GraphMaskExplainer':
            algorithm = GraphMaskExplainer(2, epochs=5)
        return algorithm

    def perform_explanation(self, model_num):
        model = GCNModel(self.input_size, self.hidden_size, self.output_size)
        model.load_state_dict(torch.load(f'./training/output_files/model_{model_num}.pt'))
        model.eval()
        model_explainer = Explainer(
                model=model,
                algorithm=self.define_algorithm(self.algo),
                explanation_type='model', 
                node_mask_type='attributes', 
                model_config=dict(
                    mode='regression',
                    task_level='graph',
                    return_type='raw'
                    ),
                )

        for ads in ['CO', 'H_FCC', 'H_HCP']:
            for counter, data_loader in enumerate(self.loader[ads]):
                data = data_loader
                if counter >= 1:
                    break

            explan = model_explainer(x=data.x, edge_index=data.edge_index, batch=data.batch) 
            path_features = f"./explanation/output_files/feature_importance_{ads}_{self.algo}_model_{model_num}.png"
            explan.visualize_feature_importance(path_features, feat_labels=self.labels)
