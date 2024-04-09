"""
Main file used for running the code
"""

# hydra imports
import hydra
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

from omegaconf import OmegaConf, open_dict

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp

# generic imports
import logging
import os

# Watchmal import
from watchmal.utils.logging_utils import get_git_version

log = logging.getLogger(__name__)


from hydra.utils import get_original_cwd


@hydra.main(config_path='config/', config_name='resnet_train', version_base="1.1")
def main(config):
    """
    Run model using given config, spawn worker subprocesses as necessary

    Args:
        config  ... hydra config specified in the @hydra.main annotation
    """
    #log.info(f"Using the following git version of WatChMaL repository: {get_git_version(os.path.dirname(to_absolute_path(__file__)))}")
    log.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")


    # Display purpose (if needed for debug)
    # original_cwd = get_original_cwd()
    # print(f"Original working directory: {original_cwd}")
    
    # Get the Hydra output directory for the current run
    hydra_output_dir = os.getcwd()
    #print(f"Hydra output directory: {hydra_output_dir}")
    
    ngpus = len(config.gpu_list)
    # is_distributed = ngpus > 1

    # initialize seed
    if config.seed is None:
        config.seed = torch.seed()


    # create run directory
    if not os.path.exists(config.dump_path):
        log.info(f"Creating directory for run output at : {config.dump_path}")
        os.makedirs(config.dump_path)
    log.info(f"Global output directory: {config.dump_path}")
    log.info(f"Specific output directory (the one use by the engine) for this run :\n\n     {hydra_output_dir}\n")


    # Initialize process group env variables
    if ngpus == 0:
        log.info("gpu_list empty, running on cpu.")
        main_worker_function(0, ngpus, config)
    
    elif ngpus == 1:
        log.info("Only one gpu found, not using multiprocessing.")
        main_worker_function(0, ngpus, config)

    else: # meaning ngpus > 1, so we are going to use multi-processing
        os.environ['MASTER_ADDR'] = 'localhost'

        if 'MASTER_PORT' in config:
            master_port = config.MASTER_PORT
        else:
            master_port = 12355
            
        # Automatically select port based on base gpu
        master_port += config.gpu_list[0]
        os.environ['MASTER_PORT'] = str(master_port)

        devids = [f"cuda:{x}" for x in config.gpu_list]
        log.info(devids)
        log.info("Using multiprocessing. Let's go for world size")
        log.info(f"Using DistributedDataParallel on these devices: {devids}")
        # mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, config, HydraConfig.get()))
        mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, config, HydraConfig.get()))


    # old and ugly code
    # if is_distributed:
    #     devids = [f"cuda:{x}" for x in config.gpu_list]
    #     log.info(devids)
    #     log.info("Using multiprocessing...")
    #     log.info(f"Using DistributedDataParallel on these devices: {devids}")
    #     mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, is_distributed, config, HydraConfig.get()))
    
    # elif ngpus == 1:
    #     log.info("Only one gpu found, not using multiprocessing...")
    #     main_worker_function(0, ngpus, is_distributed, config)
    
    # else : 
    #     log.info("gpu_list empty, running on cpu...")
    #     main_worker_function(0, ngpus, is_distributed, config)


def main_worker_function(rank, ngpus_per_node, config, hydra_config=None):
    """
    Instantiate model on a particular GPU, and perform train/evaluation tasks as specified

    Args:
        rank            ... rank of process among all spawned processes (in multiprocessing mode)
        ngpus_per_node  ... number of gpus being used (in multiprocessing mode)
        is_distributed  ... boolean indicating if running in multiprocessing mode
        config          ... hydra config specified in the @hydra.main annotation
        hydra_config    ... HydraConfig object for logging in multiprocessing
    """

    devids = [f"cuda:{x}" for x in config.gpu_list]

    if ngpus_per_node == 0:
        device = torch.device("cpu")
    else: 
        device = devids[rank] # when only one master process rank is 0
        torch.cuda.set_device(device)

        if ngpus_per_node > 1:
            # Spawned process needs to configure the job logging configuration
            configure_log(hydra_config.job_logging, hydra_config.verbose)
        
            # Set up pytorch distributed processing
            torch.distributed.init_process_group('nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
    
    log.info(f"Running main worker function rank {rank} on device: {device}")

    # if is_distributed:
    #     # Spawned process needs to configure the job logging configuration
    #     configure_log(hydra_config.job_logging, hydra_config.verbose)
    
    #     # Set up pytorch distributed processing
    #     torch.distributed.init_process_group('nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
    
    # if ngpus_per_node == 0:
    #     device = torch.device("cpu")
    # else:
    #     # Infer rank from gpu and ngpus, rank is position in gpu list

    #     #device = config.gpu_list[rank]
    #     device = devids[rank]
    #     print(f"\~Device : {device}\n")

    #     torch.cuda.set_device(device)
    


    # Instantiate the model
    model = instantiate(config.model)
    print(f"\nNumber of parameters in the model : {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
    model.to(device)
    # Configure the device to be used for model training and inference
    
    if ngpus_per_node > 1:
        # Convert model batch norms to synchbatchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])

    
    # Instantiate the engine
    # Erwan 25/02/2024 : Why don't we put the seed and is_distributed in the engine ?
    hydra_output_dir = os.getcwd()
    engine = instantiate(config.engine, model=model, rank=rank, device=device, dump_path=hydra_output_dir + "/")
    
    # Create dataset (only for gnn, for cnn it's done in the for loop below)
    if config.kind =='gnn':
        engine.configure_dataset(config.data)
        print(f"\nLength of the dataset : {len(engine.dataset)}\n")

    # keys to update in each dataloaders confic dictionnary           
    for task, task_config in config.tasks.items():

        with open_dict(task_config):

            # Configure data loaders
            if 'data_loaders' in task_config:
                match config.kind:
                    case 'cnn':
                        engine.configure_data_loaders(
                            config.data, 
                            task_config.pop("data_loaders"),
                        )
                    case 'gnn':                                                   
                        engine.configure_data_loaders_v2(
                            task_config.pop("data_loaders"), 
                        )
                    case _:
                        print(f"The kind parameter {config.kind} is unknown. Set it to 'cnn' or 'gnn'")
                        raise ValueError                    

            # Configure optimizers
            if 'optimizers' in task_config:
                engine.configure_optimizers(task_config.pop("optimizers"))
            
            # Configure scheduler
            if 'scheduler' in task_config:
                engine.configure_scheduler(task_config.pop("scheduler"))
            
            # Configure loss
            if 'loss' in task_config:
                engine.configure_loss(task_config.pop("loss"))

    # Perform tasks
    for task, task_config in config.tasks.items():
        getattr(engine, task)(**task_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
