import random
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torch.nn import L1Loss
from shared.graph_construction import PrepareData
from shared.GCN_model import GCNModel

class Train():
    def __init__(self):
        self.input_size = PrepareData().num_feat + 6
        self.hidden_size = 256
        self.output_size = 1
        self.trn_val_split = 0.8
        self.batch_size = 32
        self.epochs = 500
        self.learning_rate = 0.01
        self.gamma = 0.98
        self.weight_decay = 5e-8
        self.es_threshold = 20  #Early stopping epoch threshold (How many epochs to go further if validation loss goes up)
        self.es_epoch_start = 20 #Start checking for early stopping from this epoch on
        self.input_data()

    def input_data(self):
        random.seed(1)
        data = PrepareData()
        graphs, graphs_each = data.data_all_adsorbates()
        data_graphs = graphs['CO'] + graphs['H_FCC'] + graphs['H_HCP']
        self.num_graphs = len(data_graphs)
        data_graphs_s = data_graphs
        random.shuffle(data_graphs_s)
        trn_graphs = data_graphs_s[:int(self.trn_val_split * self.num_graphs)]
        val_graphs = data_graphs_s[int(self.trn_val_split * self.num_graphs):]
        self.trn_loader = DataLoader(trn_graphs, batch_size=self.batch_size, shuffle = False)
        self.val_loader = DataLoader(val_graphs, batch_size=self.batch_size, shuffle = False)
        self.loader_CO = DataLoader(graphs['CO'], batch_size=len(graphs['CO']), shuffle = False)
        self.loader_H_FCC = DataLoader(graphs['H_FCC'], batch_size=len(graphs['H_FCC']), shuffle = False)
        self.loader_H_HCP = DataLoader(graphs['H_HCP'], batch_size=len(graphs['H_HCP']), shuffle = False)

    def train(self):
        for counter, data in enumerate(self.trn_loader):
            self.optimizer.zero_grad()
            pred = self.model(*(data.x, data.edge_index, data.batch))
            loss = self.criterion(pred, torch.reshape(data.y,(-1,1)))
            loss.backward()
            self.optimizer.step()

    def test(self, dataloader):
        self.model.eval()
        loss = 0
        for counter, data in enumerate(dataloader):
            pred = self.model(data.x, data.edge_index, data.batch)
            loss +=  self.criterion(pred, torch.reshape(data.y,(-1,1)))
        return loss.item() / (counter + 1)

    def train_each_model(self):
        best_loss_val = 1
        best_loss_trn = 0

        file_path = f'./training/output_files/lcurve_{self.model_num}.out'
        headers = "Epoch Training_Loss(eV) Validation_Loss(eV)"
        with open(file_path, "w") as filee:
            filee.write(headers + "\n")
            for epoch in range(0, self.epochs):
                print(f'epoch: {epoch}')
                print('Training..........................................')
                self.train()
                print('Testing...........................................')
                trn_l1 = self.test(self.trn_loader)
                val_l1 = self.test(self.val_loader)
                print(f'Train loss : {trn_l1} \nValidation loss: {val_l1}')
                filee.write(f'{epoch+1}\t{trn_l1:.4f}\t{val_l1:.4f}\n')
                if epoch > self.es_epoch_start:
                    if val_l1 < best_loss_val:
                        best_loss_val = val_l1
                        best_loss_trn = trn_l1
                        es = 0
                    else:
                        es += 1
                        if es > (self.es_threshold - 1):
                            print(f'Early stopping with best_loss_val: {best_loss_val} \
                                  and best_loss_trn {best_loss_trn}')
                            break
                self.scheduler.step()

    def train_ensemble(self, model_num):
        self.model_num = model_num
        random.seed(1000 * self.model_num)
        torch.manual_seed(1000 * self.model_num)
        self.model = GCNModel(self.input_size, self.hidden_size, self.output_size)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)
        self.criterion = L1Loss()
        self.train_each_model()
        torch.save(self.model.state_dict(), f'./training/output_files/model_{self.model_num}.pt')
