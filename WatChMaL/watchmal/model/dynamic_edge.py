from typing import List

import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm, Dropout, Linear, ReLU
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool, global_max_pool
from torch_geometric.nn.models import MLP

class DynamicEdgeConv_v1(torch.nn.Module):
    r'''
    Dynamic Edge Convolutional Graph Network
    The Graph Neural Network from the 
    “Dynamic Graph CNN for Learning on Point Clouds” paper, 
    where the graph is dynamically constructed using nearest 
    neighbors in the feature space.
    '''
    def __init__(
        self,
        in_channels: int,
        conv_in_channels: List[int], #number of nodes for the edges neural network
        linear_out_features: List[int],
        dropout: float = 0.2,
        k: int = 8,  # Number of nearest neighbors for DynamicEdgeConv
        aggr: str = 'add'  # Aggregation method for DynamicEdgeConv
    ) -> None:
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        # Define the convolutional and normalization layers using DynamicEdgeConv
        for i in range(len(conv_in_channels)):
            if i == 0:
                in_features = 2 * in_channels
            else:
                in_features = 2 * conv_in_channels[i - 1]
            out_channels = conv_in_channels[i]
            mlp = MLP([in_features, out_channels, out_channels]) #input, hidden 1, output, nb of hidden layer
            self.conv_layers.append(DynamicEdgeConv(mlp, k, aggr))
            self.cl_norms.append(BatchNorm(out_channels))
    
        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        # Define the hidden and normalization layers
        first_linear_in_features = 2 * conv_in_channels[-1]  # Times 2 because of the torch.cat on gap and gsp

        for j in range(len(linear_out_features) - 1):
            in_features = first_linear_in_features if j == 0 else linear_out_features[j - 1]
            out_features = linear_out_features[j]
            self.hidden_layers.append(Linear(in_features, out_features))
            self.hl_norms.append(BatchNorm(out_features))

        # Last layer doesn't have batch_norm
        self.last_layer = Linear(linear_out_features[-2], linear_out_features[-1])

        self.drop = Dropout(p=dropout)
        self.activation_cl = ReLU()
        self.activation_hl = ReLU()

        self.output_activation = None

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x,  batch = data.x, data.batch
        # x = torch.cat([x, pos], dim=-1)

        # Apply convolutional layers
        for conv_layer, norm_layer in zip(self.conv_layers, self.cl_norms):
            x = conv_layer(x, batch)
            x = self.activation_cl(x)
            x = norm_layer(x)
            x = self.drop(x)

        # Global pooling
        xs_gap = global_mean_pool(x, batch)
        xs_gsp = global_max_pool(x, batch)

        out = torch.cat((xs_gap, xs_gsp), dim=1)  # out.shape = (batch_size, 2 * conv_in_channels[-1])

        # Apply hidden layers
        for hidden_layer, norm_layer in zip(self.hidden_layers, self.hl_norms):
            out = hidden_layer(out)
            out = self.activation_hl(out)
            out = norm_layer(out)

        out = self.last_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return F.log_softmax(out, dim=1)
