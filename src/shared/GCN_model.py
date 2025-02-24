from torch.nn import Module
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCNModel(Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(GCNModel, self).__init__()
        self.lin1  = Linear(input_size, hidden_layer_size)
        self.lin2  = Linear(int(0.5*hidden_layer_size), int(0.25*hidden_layer_size))
        self.lin3  = Linear(int(0.25*hidden_layer_size), 1)
        self.conv1 = GCNConv(hidden_layer_size, int(0.5*hidden_layer_size))
        self.conv2 = GCNConv(int(0.5*hidden_layer_size), int(0.5*hidden_layer_size))

    def forward(self, x, edge_index, batch):
        x = self.lin1(x).relu()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)  # [batch_size, hidden_layer_size]
        x = F.dropout(x, p=0.5)
        x = self.lin2(x).relu()
        x = self.lin3(x)
        return x
