import numpy as np
import matplotlib.pyplot as plt
import sys
import uproot
from typing import Tuple, Dict, List
import os
import torch
import torch_geometric
from datetime import datetime
import time


import omegaconf
from omegaconf import OmegaConf, ListConfig
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ResGatedGraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm, InstanceNorm, GraphNorm
from torch.nn import Dropout, Linear, Sigmoid, LogSoftmax, ReLU, LeakyReLU, SiLU

sys.path.append('/sps/t2k/cehrhardt/analysis_tools/tools')

# Import your modules
from tools.dataset_from_processed import DatasetFromProcessed



####################### Classes #########################

class Normalize(torch.nn.Module):
    """Normalize a torch_geometric Data object with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(
            self, 
            feat_norm,
            label_norm=None, 
            eps=1e-12, 
            inplace=False        
    ):
        
        super().__init__()
        
        self.feat_norm  = feat_norm
        self.label_norm = label_norm
        self.eps        = eps
        self.inplace    = inplace
        
        # For hydra compatibility
        if isinstance(self.feat_norm, omegaconf.listconfig.ListConfig):
            self.feat_norm = OmegaConf.to_container(self.feat_norm)
            if self.label_norm is not None:
                self.label_norm = OmegaConf.to_container(self.label_norm)

        # Need to convert list to torch tensor to perform addition & subtraction
        self.feat_norm = torch.tensor(self.feat_norm)
        if self.label_norm is not None:
            self.label_norm = torch.tensor(self.label_norm)


    def forward(self, data):
        """
        self.feat_norm and self.label_norm must contain Tensor object
        """
        
        data.x = (data.x - self.feat_norm[1]) / (self.feat_norm[0] - self.feat_norm[1] + self.eps)
        
        if self.label_norm is not None:
            data.y = (data.y - self.label_norm[1]) / (self.label_norm[0] - self.label_norm[1] + self.eps)

        return data

class ResGateConv_activation2(torch.nn.Module):
    r""""
    Args :
        in_channels : Number of features per node in the graph
        conv_kernels : The number of filters for each layer starting from the 2nd. The len defines 
            the number of layers.
        linear_out_features : The number of neurones at the end of each Linear layer. 
            The len defines the number of layers. 
            The last element of the list gives the number of classes the model will predict
    """
    def __init__(
        self,
        in_channels: int,
        conv_activation,
        linear_activation,
        conv_in_channels: List[int],
        linear_out_features: List[int],
        dropout: float = 0.2,
    ) -> None:

        super().__init__() # If more heritage than torch.nn.Module is added, modify this according to the changes


        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        # Define the convolutional and normalization layers
        for i in range(len(conv_in_channels)):
            in_channels = in_channels if i == 0 else conv_in_channels[i - 1] # Erwan : Sure to be a good idea ? Maybe enforce some check from the user (just use in_conv_feat)
            out_channels = conv_in_channels[i]
            self.conv_layers.append(ResGatedGraphConv(in_channels, out_channels))
            self.cl_norms.append(BatchNorm(out_channels))

        self.gap_norm = BatchNorm(conv_in_channels[-1])
        self.gsp_norm = BatchNorm(conv_in_channels[-1])

        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        # Define the hidden and normalization layers
        first_linear_in_features = 2 * conv_in_channels[-1] # Times 2 because of the torch.cat on gap and gsp

        for j in range(len(linear_out_features) - 1):
            in_features  = first_linear_in_features if j == 0 else linear_out_features[j - 1]
            out_features = linear_out_features[j]
            self.hidden_layers.append(Linear(in_features, out_features))
            self.hl_norms.append(BatchNorm(out_features))

        # Last layer doesn't have batch_norm
        self.last_layer = Linear(linear_out_features[-2], linear_out_features[-1])

        self.drop = Dropout(p=dropout)

        # self.activation_cl = conv_activation
        self.activation_hl = linear_activation

        
        # DO NOT add a sigmoid layer for the output (it is handled by the loss)
        # if it's not a sigmoid then define the output activation here
        # And then use BCELoss as a loss function (and not BCEwithlogitsloss)
        self.output_activation = None # ReLU()..


    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Define intermediate storage
        intermediate_values = {}

        intermediate_values['input'] = x.clone().detach()
        
        # # Apply convolutional layers
        # for conv_layer, norm_layer in zip(self.conv_layers, self.cl_norms):
        #     x = conv_layer(x, edge_index)
        #     x = self.activation_cl(x)
        #     x = norm_layer(x)
        #     x = self.drop(x)

        for i, (conv_layer, norm_layer) in enumerate(zip(self.conv_layers, self.cl_norms)):
            x = conv_layer(x, edge_index)
            key_prefix = f'conv_output{i + 1}'
            intermediate_values[key_prefix] = x.clone().detach()

            # x = self.activation_cl(x)
            # key_prefix = f'activation_conv_output{i + 1}'
            # intermediate_values[key_prefix] = x.clone().detach()

            x = norm_layer(x)
            key_prefix = f'norm_conv_output{i + 1}'
            intermediate_values[key_prefix] = x.clone().detach()

            x = self.drop(x)
            key_prefix = f'dropout_conv_output{i + 1}'
            intermediate_values[key_prefix] = x.clone().detach()

        # Global pooling 
        # Terminologie : gap, très moyen vu que global_add_pool existe aussi..
        xs_gap = global_mean_pool(x, batch)
        xs_gsp = global_max_pool(x, batch)

        xs_gap = self.gap_norm(xs_gap)       
        xs_gsp = self.gsp_norm(xs_gsp)

        out = torch.cat((xs_gap, xs_gsp), dim=1) # out.shape = (batch_size, 2 * conv_in_channels[-1])
        
        intermediate_values['pooling'] = out.clone().detach()

        # # Apply hidden layers
        # for _, (hidden_layer, norm_layer) in enumerate(zip(self.hidden_layers, self.hl_norms)):
        #     out = hidden_layer(out)
        #     out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
        #     out = norm_layer(out)
            
        for i, (hidden_layer, norm_layer) in enumerate(zip(self.hidden_layers, self.hl_norms)):
            out = hidden_layer(out)
            intermediate_values[f'linear_output{i + 1}'] = out.clone().detach()
            out = (out - torch.min(out))/(torch.max(out)- torch.min(out))
            intermediate_values[f'shifting_output{i + 1}'] = out.clone().detach()
            out = self.activation_hl(out)
            intermediate_values[f'activation_linear_output{i + 1}'] = out.clone().detach()
            out = norm_layer(out)
            intermediate_values[f'norm_linear_output{i + 1}'] = out.clone().detach()

            # if i < len(self.hidden_layers) - 1: # The last block won't have an activation and a norm layer
            #     out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
            #     out = norm_layer(out)

        out =  self.last_layer(out)
        intermediate_values[f'last_layer'] = out.clone().detach()
        
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out, intermediate_values

class ResGateConv_normalization(torch.nn.Module):
    r""""
    Args :
        in_channels : Number of features per node in the graph
        conv_kernels : The number of filters for each layer starting from the 2nd. The len defines 
            the number of layers.
        linear_out_features : The number of neurones at the end of each Linear layer. 
            The len defines the number of layers. 
            The last element of the list gives the number of classes the model will predict
    """
    def __init__(
        self,
        in_channels: int,
        conv_activation,
        linear_activation,
        norm_method, 
        conv_in_channels: List[int],
        linear_out_features: List[int],
        dropout: float = 0.2,
    ) -> None:

        super().__init__() # If more heritage than torch.nn.Module is added, modify this according to the changes

        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        if norm_method == 'BatchNorm':
            self.norm_method = BatchNorm
        if norm_method == 'InstanceNorm':
            self.norm_method = InstanceNorm
        if norm_method == 'GraphNorm':
            self.norm_method = GraphNorm

        # Define the convolutional and normalization layers
        for i in range(len(conv_in_channels)):
            print(i)
            in_channels = in_channels if i == 0 else conv_in_channels[i - 1] # Erwan : Sure to be a good idea ? Maybe enforce some check from the user (just use in_conv_feat)
            out_channels = conv_in_channels[i]
            self.conv_layers.append(ResGatedGraphConv(in_channels, out_channels))
            self.cl_norms.append(self.norm_method(out_channels))

        self.gap_norm = self.norm_method(in_channels=conv_in_channels[-1])
        self.gsp_norm = self.norm_method(in_channels=conv_in_channels[-1])

        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        # Define the hidden and normalization layers
        first_linear_in_features = 2 * conv_in_channels[-1] # Times 2 because of the torch.cat on gap and gsp

        for j in range(len(linear_out_features) - 1):
            in_features  = first_linear_in_features if j == 0 else linear_out_features[j - 1]
            out_features = linear_out_features[j]
            self.hidden_layers.append(Linear(in_features, out_features))
            self.hl_norms.append(self.norm_method(in_channels=out_features))

        # Last layer doesn't have batch_norm
        self.last_layer = Linear(linear_out_features[-2], linear_out_features[-1])

        self.drop = Dropout(p=dropout)

        self.activation_cl = conv_activation
        self.activation_hl = linear_activation

        
        # DO NOT add a sigmoid layer for the output (it is handled by the loss)
        # if it's not a sigmoid then define the output activation here
        # And then use BCELoss as a loss function (and not BCEwithlogitsloss)
        self.output_activation = None # ReLU()..


    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply convolutional layers
        for conv_layer, norm_layer in zip(self.conv_layers, self.cl_norms):
            x = conv_layer(x, edge_index)
            x = self.activation_cl(x)
            x = norm_layer(x)
            x = self.drop(x)

        # Global pooling 
        # Terminologie : gap, très moyen vu que global_add_pool existe aussi..
        xs_gap = global_mean_pool(x, batch)
        xs_gsp = global_max_pool(x, batch)

        xs_gap = self.gap_norm(xs_gap)       
        xs_gsp = self.gsp_norm(xs_gsp)

        out = torch.cat((xs_gap, xs_gsp), dim=1) # out.shape = (batch_size, 2 * conv_in_channels[-1])

        # Apply hidden layers
        for _, (hidden_layer, norm_layer) in enumerate(zip(self.hidden_layers, self.hl_norms)):
            out = hidden_layer(out)
            out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
            out = norm_layer(out)
            
            # if i < len(self.hidden_layers) - 1: # The last block won't have an activation and a norm layer
            #     out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
            #     out = norm_layer(out)

        out =  self.last_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

class ResGateConv_activation(torch.nn.Module):
    r""""
    Args :
        in_channels : Number of features per node in the graph
        conv_kernels : The number of filters for each layer starting from the 2nd. The len defines 
            the number of layers.
        linear_out_features : The number of neurones at the end of each Linear layer. 
            The len defines the number of layers. 
            The last element of the list gives the number of classes the model will predict
    """
    def __init__(
        self,
        in_channels: int,
        conv_activation,
        linear_activation,
        conv_in_channels: List[int],
        linear_out_features: List[int],
        dropout: float = 0.2,
    ) -> None:

        super().__init__() # If more heritage than torch.nn.Module is added, modify this according to the changes


        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        # Define the convolutional and normalization layers
        for i in range(len(conv_in_channels)):
            in_channels = in_channels if i == 0 else conv_in_channels[i - 1] # Erwan : Sure to be a good idea ? Maybe enforce some check from the user (just use in_conv_feat)
            out_channels = conv_in_channels[i]
            self.conv_layers.append(ResGatedGraphConv(in_channels, out_channels))
            self.cl_norms.append(BatchNorm(out_channels))

        self.gap_norm = BatchNorm(conv_in_channels[-1])
        self.gsp_norm = BatchNorm(conv_in_channels[-1])

        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        # Define the hidden and normalization layers
        first_linear_in_features = 2 * conv_in_channels[-1] # Times 2 because of the torch.cat on gap and gsp

        for j in range(len(linear_out_features) - 1):
            in_features  = first_linear_in_features if j == 0 else linear_out_features[j - 1]
            out_features = linear_out_features[j]
            self.hidden_layers.append(Linear(in_features, out_features))
            self.hl_norms.append(BatchNorm(out_features))

        # Last layer doesn't have batch_norm
        self.last_layer = Linear(linear_out_features[-2], linear_out_features[-1])

        self.drop = Dropout(p=dropout)

        self.activation_cl = conv_activation
        self.activation_hl = linear_activation

        
        # DO NOT add a sigmoid layer for the output (it is handled by the loss)
        # if it's not a sigmoid then define the output activation here
        # And then use BCELoss as a loss function (and not BCEwithlogitsloss)
        self.output_activation = None # ReLU()..


    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply convolutional layers
        for conv_layer, norm_layer in zip(self.conv_layers, self.cl_norms):
            x = conv_layer(x, edge_index)
            x = self.activation_cl(x)
            x = norm_layer(x)
            x = self.drop(x)

        # Global pooling 
        # Terminologie : gap, très moyen vu que global_add_pool existe aussi..
        xs_gap = global_mean_pool(x, batch)
        xs_gsp = global_max_pool(x, batch)

        xs_gap = self.gap_norm(xs_gap)       
        xs_gsp = self.gsp_norm(xs_gsp)

        out = torch.cat((xs_gap, xs_gsp), dim=1) # out.shape = (batch_size, 2 * conv_in_channels[-1])

        # Apply hidden layers
        for _, (hidden_layer, norm_layer) in enumerate(zip(self.hidden_layers, self.hl_norms)):
            out = hidden_layer(out)
            out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
            out = norm_layer(out)
            
            # if i < len(self.hidden_layers) - 1: # The last block won't have an activation and a norm layer
            #     out = self.activation_hl(out)   # Best thing would probably to add the last layer apart in __init__
            #     out = norm_layer(out)

        out =  self.last_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out

################# Load data ######################

feat_max = [3250, 3250, 3300, 500, 420]
feat_min = [-3250, -3250,  -3300, -600, 0]

label_max = [3250, 3250, 3300, 250]
label_min = [-3250, -3250,  -3300, -1200]

transform = torch_geometric.transforms.Compose([Normalize(feat_norm=[feat_max,feat_min], label_norm=[label_max, label_min])])

dataset = DatasetFromProcessed(graph_folder_path='/sps/t2k/cehrhardt/dataset/graph_1.5M_hitxyztc_t_xyzt_r30', graph_file_names = ['data.pt'],  verbose = 1, transform=transform)

################# index split ######################

index_list = np.load('/sps/t2k/cehrhardt/Caverns/index_lists/UnifVtx_electron_HK_10MeV_train_val_test_1.5M.npz')

# print("Arrays in the .npz file:", index_list.files)

# for array_name in index_list.files:
#     print(f"Array '{array_name}':")
#     print(index_list[array_name])

sampled_data = dataset[index_list['test_idxs']]

# predict_loader = DataLoader(sampled_train_data, batch_size=1, shuffle=True)

# sampled = dataset[index_list['test_idxs']]

loader = DataLoader(sampled_data, batch_size=5000, shuffle=False)


################# Load model ######################

best = "/sps/t2k/cehrhardt/Caverns/WatChMaL/outputs/2024-07-26/07-16-48/RegressionEngine_ResGateConv_activation_BEST.pth"
# best = "/sps/t2k/cehrhardt/Caverns/WatChMaL/outputs/2024-07-14/09-29-28/RegressionEngine_ResGateConv_activation_BEST.pth"
# Load data from the .pth file
checkpoint = torch.load(best, map_location=torch.device('cuda'))

print(checkpoint.keys())


nb_label=4
# best_model = ResGateConv_normalization(
#     5,  # in_channels
#     LeakyReLU(negative_slope=0.1),  # conv_activation
#     SiLU(),  # linear_activation
#     'BatchNorm',
#     [32, 64, 64],  # conv_in_channels
#     [128, 32, 4],  # linear_out_features
#     0  # dropout
# )
best_model = ResGateConv_activation(
    5,  # in_channels
    ReLU(),  # conv_activation
    SiLU(),  # linear_activation
    [32, 64, 128],  # conv_in_channels
    [128, 64, 32, 4],  # linear_out_features
    0  # dropout
)

# untrained_model = ResGateConv_activation_mygraphvar(5, [32, 64], [128, 32, 1], 0)
# best_model = ResGateConv_activation_mygraphvar(5, [32, 64], [128, 32, 1], 0)
# best_model = ResGateConv_activation(5, LeakyReLU(0.1), SiLU(), [32, 64], [128, 32, nb_label], 0)

best_model.load_state_dict(checkpoint['state_dict'])
best_model.eval()


################# Output investigation ######################

predictions_last = []
# initial_x = []
# predictions_first = []
times_per_event = []
# Iterate through the data loader
for data in loader: 
    start_time = time.time()
    # Forward pass through best_model
    out_last = best_model(data)
    end_time = time.time()
    time_per_event = (end_time - start_time) / len(data)
    times_per_event.append(time_per_event)
    predictions_last.append(out_last.detach().cpu().numpy())
    # initial_x.append(ini_x.detach().cpu().numpy())
    break
    # Forward pass through untrained_model
    # out_first, intermediate_values_untrained = untrained_model(data)
    # predictions_first.append(out_first.detach().cpu().numpy())

print(f"Average time per event: {times_per_event[0]:.6f} seconds")

# Concatenate predictions into arrays
predictions_last = np.concatenate(predictions_last, axis=0)
# initial_x = np.concatenate(initial_x, axis=0)
# predictions_first = np.concatenate(predictions_first, axis=0)

# Reshape predictions to match the desired shape
predictions_last = predictions_last.reshape(-1, 4)  # Assuming 4 is the number of output dimensions
# initial_x = initial_x.reshape(-1, 5)
# predictions_first = predictions_first.reshape(-1, 4)  # Assuming 4 is the number of output dimensions

# Perform label scaling back to original range
predictions_last = predictions_last * (np.array(label_max) - np.array(label_min)) + np.array(label_min)
# predictions_first = predictions_first * (np.array(label_max) - np.array(label_min)) + np.array(label_min)

# Get the current date in the desired format
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Generate the directory path
dir_path = f"/sps/t2k/cehrhardt/analysis_tools/distributions/{current_date}"

# Create the directory if it doesn't exist
os.makedirs(dir_path, exist_ok=True)

# Generate the file path
file_path = os.path.join(dir_path, "prediction.npz")

np.savez(file_path, trained = predictions_last)

# # distributions = np.array([intermediate_values_trained, intermediate_values_untrained])
# distributions = np.array([intermediate_values_trained])

# # Save predictions and intermediate values as dictionaries
# np.savez(f'/sps/t2k/cehrhardt/analysis_tools/layers_distributions/{current_date}/intermediate_values.npz', distributions = distributions)#intermediate_values_trained=intermediate_values_trained, intermediate_values_untrained=intermediate_values_untrained)
