
import numpy as np
import torch

import omegaconf
from omegaconf import OmegaConf

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

# watchmal imports
from watchmal.dataset.data_utils import match_type
from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)


"""
This file should contain all the callables (i.e python functions or class with a __call__ attribut) 
used for creating graph datasets.

"""

# À FAIRE : 
# 30/01 :  - Ajouter une erreur si les tailles de feat/label_norm et du nombre de features dans data.x/y ne correspondent pas
#          - Ajouter de la doc sur les appels [0] et [1]
# 14/02 :  - Mettre à jour la doc de cette fonction

class ExtremaModif(torch.nn.Module):

    def __init__(
            self, 
            feat_extrem,
            label_extrem=None, 
            inplace=False        
    ):
        
        super().__init__()
        
        self.feat_extrem  = feat_extrem
        self.label_extrem = label_extrem
        self.inplace    = inplace
        
        # For hydra compatibility
        if isinstance(self.feat_extrem, omegaconf.listconfig.ListConfig):
            self.feat_extrem = OmegaConf.to_container(self.feat_extrem)
            if self.label_extrem is not None:
                self.label_extrem = OmegaConf.to_container(self.label_extrem)

        # Convert lists to torch tensors
        self.feat_max = torch.tensor(self.feat_extrem[0])
        self.feat_min = torch.tensor(self.feat_extrem[1])
        if self.label_extrem is not None:
            self.label_max = torch.tensor(self.label_extrem[0])
            self.label_min = torch.tensor(self.label_extrem[1])


    def forward(self, data):
        """
        self.feat_norm and self.label_norm must contain Tensor object
        """
        
        # Apply maximum and minimum value filter for features
        for i in range(len(self.feat_max)):
            if self.feat_max[i]:
                # print(self.feat_max[i])
                data.x[:, i] = torch.clamp(data.x[:, i], max=self.feat_max[i])
            if self.feat_min[i]:
                data.x[:, i] = torch.clamp(data.x[:, i], min=self.feat_min[i])
        
        if self.label_extrem is not None:
            for i in range(len(self.label_max)):
                if self.label_max[i]:
                    data.y[:, i] = torch.clamp(data.y[:, i], max=self.label_max[i])
                if self.label_min[i]:
                    data.y[:, i] = torch.clamp(data.y[:, i], min=self.label_min[i])

        return data

class LogModif(torch.nn.Module):

    def __init__(
            self, 
            feat_log,
            label_log=None, 
            inplace=False        
    ):
        
        super().__init__()
        
        self.feat_log  = feat_log
        self.label_log = label_log
        self.inplace    = inplace
        
        # For hydra compatibility
        if isinstance(self.feat_log, omegaconf.listconfig.ListConfig):
            self.feat_log = OmegaConf.to_container(self.feat_log)
            if self.label_log is not None:
                self.label_log = OmegaConf.to_container(self.label_log)

        # Convert lists to torch tensors
        self.feat_log = torch.tensor(self.feat_log, dtype=torch.bool).squeeze()
        if self.label_log is not None:
            self.label_log = torch.tensor(self.label_log, dtype=torch.bool).squeeze()


    def forward(self, data):
        """
        self.feat_norm and self.label_norm must contain Tensor object
        """

        for i in range(len(self.feat_log)):
            if self.feat_log[i]:
                data.x[:, i] = torch.log(data.x[:, i] + 1) # log(x + 1) to avoid log(0)
        
        if self.label_log is not None:
            for i in range(len(self.label_log)):
                if self.label_log[i]:
                    data.y[:, i] = torch.log(data.y[:, i] + 1)

        return data

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
            eps=1e-8, 
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


class NormalizeGraphVar(torch.nn.Module):
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
            feat_nodes_norm,
            feat_graph_norm,
            label_norm=None, 
            eps=1e-8, 
            inplace=False        
    ):
        
        super().__init__()
        
        self.feat_nodes_norm  = feat_nodes_norm
        self.feat_graph_norm  = feat_graph_norm
        self.label_norm = label_norm
        self.eps        = eps
        self.inplace    = inplace
        
        # For hydra compatibility
        if isinstance(self.feat_nodes_norm, omegaconf.listconfig.ListConfig):
            self.feat_nodes_norm = OmegaConf.to_container(self.feat_nodes_norm)
            self.feat_graph_norm = OmegaConf.to_container(self.feat_graph_norm)
            if self.label_norm is not None:
                self.label_norm = OmegaConf.to_container(self.label_norm)

        # Need to convert list to torch tensor to perform addition & subtraction
        self.feat_nodes_norm = torch.tensor(self.feat_nodes_norm)
        self.feat_graph_norm = torch.tensor(self.feat_graph_norm)
        if self.label_norm is not None:
            self.label_norm = torch.tensor(self.label_norm)


    def forward(self, data):
        """
        self.feat_norm and self.label_norm must contain Tensor object
        """
        data.x = (data.x - self.feat_nodes_norm[1]) / (self.feat_nodes_norm[0] - self.feat_nodes_norm[1] + self.eps)
        
        data.x_graph = (data.x_graph - self.feat_graph_norm[1]) / (self.feat_graph_norm[0] - self.feat_graph_norm[1] + self.eps)

        if self.label_norm is not None:
            data.y = (data.y - self.label_norm[1]) / (self.label_norm[0] - self.label_norm[1] + self.eps)

        return data

# class DataToWatchmalDict(torch.nn.Module):
#     """
#     Arguments:
#         target_to_type (string) : type to convert the target to 

#     Forward method : 
#         .item() required in watchmal for the collate_fn to create a (b_size, ) instead of (b_size, 1)
        
#     Note :
#         - "idx" is only used when looking at the output folder (for evaluation only in watchmal)
#         - The class Negative Log Likelyhood of torch requires int as labels
#     """

#     def __init__(self, target_to_type):
#         super().__init__()

#         match target_to_type:
#             case 'int16':
#                 to_type = torch.int16
#             case 'int32':
#                 to_type = torch.int32
#             case 'int64':
#                 to_type = torch.int64
#             case 'float16':
#                 to_type = torch.float16
#             case 'float32':
#                 to_type = torch.float32
#             case _:
#                 log.info(f"DataToWatchmalDict : Value Error, target_to_type {target_to_type} is not supported")
#                 log.info("Add the data type into the transform or change the new target type\n\n")
#                 raise ValueError
            
#         self.target_to_type = to_type

#     def forward(self, data):

#         if isinstance(data, dict): # If the data has already been wrapped into a dict, no need to do it again
#             return data
#         # Note : See in_memory_dataset code "get" method for more details about 
#         # why becomes a dict after on pass (copy vs deepcopy)


#         watchmal_dict = {
#             'data': data,
#             'target': data.y.to(self.target_to_type),
#             'indices': data.idx # wtf le s n'a pas de sens ptn; à modifier dans engine.evaluate()
#         }

#         return watchmal_dict 
    

class ConvertAndToDict(torch.nn.Module):
    """
    Arguments:
        feature_to_type (string)      : type to convert each feature to
        target_to_type (string)       : type to convert the target to 
        map_labels (list of integers) : which label to assign the PID to. 
            For example map_labels=[11, 13, 111] will convert 11 -> 0, 13 -> 1 and 111 -> 2. 
        
    Note :
        - "idx" is only used when looking at the output folder (for evaluation only in watchmal)
        - The class Negative Log Likelyhood of torch requires int as labels
    """

    def __init__(self, feature_to_type: str, target_to_type: str, map_labels: list=None):
        super().__init__()

        # We consider homogeneous graphs for now (the 'Data' obj). 
        # So all features are of the same type
        # Hence the should be converted to the same type also (no need to make a list for feat..type and target..type)
        self.feature_to_type = match_type(feature_to_type) 
        self.target_to_type  = match_type(target_to_type)
        self.map_labels      = map_labels

    def forward(self, data):

        # If the data has already been wrapped into a dict, no need to do it again
        # See the in_memory_dataset "get" method for more details about 
        # why it becomes a dict after on pass (copy vs deepcopy)
        if isinstance(data, dict): 
            return data

        data.x = data.x.to(self.feature_to_type)

        if self.map_labels is not None:
            data.y = torch.tensor(self.map_labels.index(data.y))
        data.y = data.y.to(self.target_to_type)


        data_dict = {
            'data': data,
            'target': data.y,
            'indice': data.idx # might a better for for this. We duplicate the idx information (data + dict)
        }

        return data_dict 
    