"""
Main file used for running the code
"""

# hydra imports
import hydra
from hydra.utils import instantiate, get_original_cwd, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

import omegaconf
from omegaconf import OmegaConf, open_dict, DictConfig

#wandb import 
import wandb

# torch imports
import torch
import torch.multiprocessing as mp

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# generic imports
import logging
import os
import pprint
import yaml

# Watchmal import
from watchmal.utils.logging_utils import get_git_version
from watchmal.utils.build_utils import build_dataset, build_model


log = logging.getLogger(__name__)



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(
        backend="nccl", init_method='env://', rank=rank, world_size=world_size
    )

def wandb_setup_run(args):
    if args.log_all:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group="DDP",
        )
    else:
        if args.local_rank == 0:
            run = wandb.init(
                entity=args.entity,
                project=args.project,
            )
        else:
            run = None

    return run

def merge_config(hydra_config, wandb_config):
    """
    Update Hydra configuration with values from W&B configuration if the keys match.
    
    Args:
    - hydra_config (omegaconf.DictConfig): The Hydra configuration object.
    - wandb_config (dict): The dictionary containing W&B configuration.
    
    Returns:
    - hydra_config (omegaconf.DictConfig): The updated Hydra configuration object.
    """
    # Iterate over all items in wandb_config
    for key, value in wandb_config.items():
        # Check if the key exists in the hydra_config and is not a nested structure
        if omegaconf.OmegaConf.is_config(hydra_config) and key in hydra_config:
            # Update the value in hydra_config with the value from wandb_config
            print(f'hydra key : {key}')
            hydra_config[key] = value
        else:
            # If the key is deeper in the structure, let's try finding and updating it
            for sub_config in hydra_config:
                print(f"Sub config : {sub_config}")
                if omegaconf.OmegaConf.is_config(hydra_config[sub_config]) and key in hydra_config[sub_config]:
                    print(f"Sub key : {key}")
                    hydra_config[sub_config][key] = value
                    break
    
    return hydra_config


# A decorator changes the way a function work
@hydra.main(config_path='config/', config_name='resnet_train', version_base="1.1")
def hydra_main(config):
    """
    Run model using given config, spawn worker subprocesses as necessary

    Args:
        config  ... hydra config specified in the @hydra.main annotation
    """
    # Display the config of the run
    y = OmegaConf.to_yaml(config)
    log.info(f"Running with the following config:\n {y}\n")

    # Get the current and Hydra output directories for the run
    original_cwd = get_original_cwd()
    hydra_output_dir = os.getcwd()

    # Display folders for the run
    log.info(f"Original working directory: {original_cwd}")
    log.info(f"Hydra output directory: {hydra_output_dir}")
    log.info(f"Global output directory for this run : {config.dump_path}")
    log.info(f"Specific output directory (the one use by the engine to save & load ) for this run :\n\n     {hydra_output_dir}\n")

    # Get the global config
    global_hydra_config = HydraConfig.get()


    # Lauch the run
    print('\nHERE WE GO FOR THE MAIIIIIIIN\n')
    main(hydra_config=config, global_hydra_config=global_hydra_config)


def main(hydra_config, global_hydra_config):

    gpu_list = hydra_config.gpu_list
    dump_path = hydra_config.dump_path

    # Create the output folder for all the runs only once
    if not os.path.exists(dump_path):
        log.info(f"Creating directory for run output at : {dump_path}")
        os.makedirs(dump_path)

    with wandb.init() as wandb_run:
        wandb_config = wandb.config
        pprint.pprint(wandb_config)

        project = wandb_run.project
        entity = wandb_run.entity
        sweep_id = wandb_run.sweep_id

        print("Project:", project)
        print("Entity:", entity)
        print("Sweep ID:", sweep_id)

        hydra_config = merge_config(hydra_config, wandb_config)
        print(f"Now running with the following config : \n {OmegaConf.to_yaml(hydra_config)}")
        # Create or get the dataset (only for gnn, for cnn see run(..))
        # It's only when the dataset has to be processed that 
        # this part needs to be outside the run(..) function.
        # In the end we will need to make .root -> graph.pt outside of watchmal
        if hydra_config.kind == 'gnn':
            dataset = build_dataset(hydra_config)        
        else : # When using kind='cnn'
            dataset = None
        
        # Parse gpu argument to set the type of run (cpu/gpu/gpus)
        if len(gpu_list) == 0:
            log.info("The gpu list is empty. Run will be done on cpu.\n")
            start_run(rank=0, gpu_list=gpu_list, dataset=dataset, wandb_run=wandb_run, hydra_config=hydra_config)
        
        elif len(gpu_list) == 1:
            log.info("One gpu in the gpu list.\n")
            start_run(rank=0, gpu_list=gpu_list, dataset=dataset, wandb_run=wandb_run, hydra_config=hydra_config)

        else: 
            devids = [f"cuda:{x}" for x in gpu_list]
            log.info(f"Multiple gpus in the gpu_list. Running on distributed mode")
            log.info(f"List of accessible devices : {devids}\n")

            # Configure torch and hydra log for multi processing
            configure_log(global_hydra_config.job_logging, global_hydra_config.verbose)
            mp.spawn(
                start_run, 
                nprocs=len(gpu_list), # In our case we always consider n_processes=n_gpus=len(gpu_list)
                args=(gpu_list, dataset, wandb_run, hydra_config)
            )
 


def start_run(rank, gpu_list, dataset, wandb_run, hydra_config):

    
    device = 'cpu' if len(gpu_list) == 0 else rank
    wandb_run =  wandb_run if rank ==0 else None

    log.info(f"Running worker {rank} on device : {device}")

    if len(gpu_list) > 1:
        ddp_setup(rank, world_size=len(gpu_list))
    
    # Instantiate the model (for each process if many) 
    model = build_model(
        model_config=hydra_config.model, 
        device=device, 
        use_dpp=(len(gpu_list) > 1)
    )

    # Instantiate the engine (for each process if many) --- Let's do the model in the engine ?
    hydra_output_dir = os.getcwd()
    engine = instantiate(
        config=hydra_config.engine,
        wandb_run=wandb_run,
        dump_path=hydra_output_dir + "/",
        model=model, 
        rank=rank, 
        device=device
    )

    if hydra_config.kind == 'gnn':
            engine.set_dataset(dataset)
    
    # keys to update in each dataloaders confic dictionnary           
    for task, task_config in hydra_config.tasks.items():

        with open_dict(task_config):

            # Configure data loaders
            if 'data_loaders' in task_config:
                match hydra_config.kind:
                    case 'cnn':
                        engine.configure_data_loaders(
                            hydra_config.data, 
                            task_config.pop("data_loaders"),
                        )
                    case 'gnn':                                                   
                        engine.configure_data_loaders_v2(
                            task_config.pop("data_loaders"), 
                        )
                    case _:
                        print(f"The kind parameter {hydra_config.kind} is unknown. Set it to 'cnn' or 'gnn'")
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
    for task, task_config in hydra_config.tasks.items():
        getattr(engine, task)(**task_config)

    
    if len(gpu_list) > 1:
        destroy_process_group()
    


if __name__ == '__main__':
    hydra_main()



