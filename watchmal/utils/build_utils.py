"""
Functions to build / configure classes
that need to be done (any reason below)
- outside the engine
- outside the run(..) function in case of mulitprocessing
"""
# Import for doc
from omegaconf import DictConfig

# torch import
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# watchmal impport
from watchmal.dataset.data_utils import get_dataset

# hydra and log import
import logging
from hydra.utils import instantiate

#### Code

log = logging.getLogger(__name__)

def build_dataset(config: DictConfig):
    log.info(f"Loading the dataset..")

    dataset_config = config.data # data contains .dataset and .transforms

    dataset = get_dataset(
        dataset_config.dataset.parameters, 
        dataset_config.transforms
    )

    log.info('Finished loading')
    log.info(f"Length of the dataset : {len(dataset)}")
    log.info(f"First graph of the dataset : {dataset.get(0)}")
    print("")
    return dataset


def build_model(model_config, device, use_dpp=False):
    """
    Device arg was kept in the case of gpu_id > 0 but only one gpu used
    """
    model = instantiate(model_config)
    if (device == 'cpu') or (use_dpp and device =='cuda:0'):
        print(f"\nNumber of parameters in the model : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    model.to(device)

    if use_dpp:
        # Convert model batch norms to synchbatchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap the model with DistributedDataParallel mode
        model = DDP(model, device_ids=[device])

    return model