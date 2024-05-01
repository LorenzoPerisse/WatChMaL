"""
Class for training a fully supervised classifier
"""

# generic imports
import numpy as np
from datetime import timedelta
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import pprint

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel

# pyg imports
from torch_geometric.nn import summary

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader, get_data_loader_v2, get_dataset
from watchmal.utils.logging_utils import CSVLog

log = logging.getLogger(__name__)


class ReconstructionEngine(ABC):
    def __init__(
            self, 
            truth_key, 
            model, 
            rank, 
            device, 
            dump_path,
            dataset=None
        ):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target values in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        """
        # create the directory for saving the log and dump files
        self.dump_path = dump_path

        # variables for the model
        self.rank = rank
        self.device = torch.device(device) 
        self.model = model

        # variables to monitor training pipelines
        self.epoch = 0
        self.step = 0
        self.iteration = 0
        self.best_validation_loss = 1.0e10

        # variables for the dataset
        self.dataset      = None if dataset is None else dataset
        self.split_path   = ""
        self.truth_key    = truth_key

        # Variables for the dataloaders
        self.data_loaders = {}

        # Set up the parameters to save given the model type
        if isinstance(self.model, DistributedDataParallel): # 25/05/2024 - Erwan : Best way to check ddp mode ?
            self.is_distributed = True

            self.module = self.model.module
            self.n_gpus = torch.distributed.get_world_size() # Returns the number of processes in the group. Not all the gpu availables
        else:
            self.is_distributed = False
            self.module = self.model

        # define the placeholder attributes
        self.data   = None
        self.target = None
        self.loss   = None
        self.outputs_epoch_history = []  

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # logging attributes
        self.train_log = CSVLog(self.dump_path + f"log_train_{self.rank}.csv")
        #self.val_log = CSVLog(self.dump_path + f"log_val_{self.rank}.csv")

        if self.rank == 0:
            self.val_log = CSVLog(self.dump_path + "log_val.csv") # Only rank 0 will save its performances at validation in a .csv file



    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config)

    def configure_optimizers(self, optimizer_config):
        """Instantiate an optimizer from a hydra config."""
        self.optimizer = instantiate(optimizer_config, params=self.module.parameters())

    def configure_scheduler(self, scheduler_config):
        """Instantiate a scheduler from a hydra config."""
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)


    def configure_dataset(self, data_config):
        """
        Set up the dataset according to the data_config and the transform_config
        There is only dataset for training and testing. The split train/test is made by the sampler 
        initialized with the dataloader.
        """
        #print(f"Dataset config : {data_config.dataset}")
        self.split_path = data_config.dataset.split_path # Initialized for data_loaders
        self.dataset = get_dataset(data_config.dataset.parameters, data_config.transforms)
    
    def set_dataset(self, dataset):
        
        if self.dataset is not None:
            print(f'Error : Dataset is already set in the engine of the process : {self.rank}')
            raise ValueError

        self.dataset = dataset
        self.split_path = dataset.split_path

    def configure_data_loaders_v2(self, loaders_config):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        loaders_config
            Dictionnary specifying a list of dataloader configurations.
        """
        #print(loaders_config)
        for name, loader_config in loaders_config.items():
           
            self.data_loaders[name] = get_data_loader_v2(
                dataset=self.dataset,
                split_path=self.split_path,
                device=self.device,
                is_distributed=self.is_distributed,
                **loader_config
            )
    
        # Instead, note the self.datatets
        # for name, loader_config in loaders_config.items():
        #    self.data_loaders[name] = get_data_loader(self.datasets, **loader_config, is_distributed=is_distributed, seed=seed)

    def configure_data_loaders(self, data_config, loaders_config):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        data_config
            Hydra config specifying dataset.
        loaders_config
            Hydra config specifying a list of dataloaders.
        is_distributed : bool
            Whether running in multiprocessing mode.
        seed : int
            Random seed to use to initialize dataloaders.
        """

        for name, loader_config in loaders_config.items():
            self.data_loaders[name] = get_data_loader(
                **data_config.h5_dataset, 
                **loader_config, 
                is_distributed=self.is_distributed, 
                device=self.device
            )

    def get_synchronized_outputs(self, output_dict):
        """
        Gathers results from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        output_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_output_dict : dict of torch.Tensor
            Dictionary containing concatenated tensor values gathered from all processes
        """
        global_output_dict = {}
        for name, tensor in output_dict.items():
            if self.is_distributed:
                if self.rank == 0:
                    tensor_list = [torch.zeros_like(tensor, device=self.device) for _ in range(self.n_gpus)]
                    torch.distributed.gather(tensor, tensor_list)
                    global_output_dict[name] = torch.cat(tensor_list).detach().cpu().numpy()
                else:
                    torch.distributed.gather(tensor, dst=0)
            else:
                global_output_dict[name] = tensor.detach().cpu().numpy()
        return global_output_dict

    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch 
        distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_metric_dict : dict
            Dictionary containing mean of tensor values gathered from all processes
        """
        global_metric_dict = {}
        
        for name, tensor in zip(metric_dict.keys(), metric_dict.values()): # .items() ?
            if self.is_distributed:

                # Aggregate the tensors from all the processes (of the group created by init_process_group(..))
                torch.distributed.reduce(tensor, 0) # default operation is adding all the 
                
                if self.rank == 0:
                    global_metric_dict[name] = tensor.item() / self.n_gpus
           
            else:
                global_metric_dict[name] = tensor.item()
        
        return global_metric_dict

    def get_synchronized(self, outputs):
        """
        Gathers metrics from multiple processes using pytorch 
        distributed operations for DistributedDataParallel

        Note :  Only modifies rank O's outputs dictionnary
                Tensors are kept. (Nothing is converted via .item())
        Parameters
        ==========
        outputs : dict
            Dictionary containing values that are tensor outputs of a single process.

        Returns
        =======
        global_metric_dict : dict
            Dictionary containing mean of tensor values gathered from all processes
        """
        new_outputs = {}
        
        for name, tensor in outputs.items():
            
            # Reduce must be called from all the processes
            # note that only the tensor on rank 0 is modified
            # the others remain the same.
            torch.distributed.reduce(tensor, 0, op=torch.distributed.ReduceOp.SUM) 

            if self.rank == 0:
                # The reduce operation being a sum over all the processes, 
                # we need to divide by n_gpus to get the average value 
                new_outputs[name] = tensor / self.n_gpus 

            #new_outputs[name] = tensor.item() # to detach and convert the tensor for each of the processes.
        
        return new_outputs
    
    @abstractmethod
    def forward(self, forward_type='train'):
        pass 

    def backward(self):
        """Backward pass using the loss computed for a mini-batch"""
        self.optimizer.zero_grad()  # reset gradients from last step
        self.loss.backward()        # compute the new gradients for this iteration
        self.optimizer.step()       # perform gradient descent on all the parameters of the model


    def train(self, epochs=0, val_interval=20, num_val_batches=4, checkpointing=False, save_interval=None):
        """
        Train the model on the training set. The best state is always saved during training.

        Parameters
        ==========
        epochs: int
            Number of epochs to train, default 1
        val_interval: int
            Number of iterations between each validation, default 20
        num_val_batches: int
            Number of mini-batches in each validation, default 4
        checkpointing: bool
            Whether to save state every validation, default False
        save_interval: int
            Number of epochs between each state save, by default don't save
        """
        
        start_run_time = datetime.now()

        log.info(f"Engine : {self.rank} | Dataloaders : {self.data_loaders}")
        if self.rank == 0:
            #log.info(f"Training {epochs} epochs with {num_val_batches}-batch validation each {val_interval} iterations\n\n")
            log.info(f"Starting training for {epochs} epochs\n\n")
        
        # initialize epoch and iteration counters
        epoch                = 0 # (used by nick)  counter of epoch
        self.iteration       = 1 # (used by erwan) counter of the steps of all epochs
        self.step            = 0 # (used by nick)  counter of the steps of one epoch
        
        self.best_validation_loss = np.inf # used keep track of the validation loss


        train_loader = self.data_loaders["train"]
        val_loader   = self.data_loaders["validation"]

        # global loop for multiple epochs        
        for epoch in range(epochs):
            
            # variables that will be used outside the train function
            self.epoch = epoch

            # ---- Starting the training epoch ---- #
            epoch_start_time = datetime.now()
            if ( self.rank == 0 ):
                log.info(f"\n\nTraining epoch {self.epoch + 1}/{epochs} starting at {epoch_start_time}")
            

            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)
          
            outputs_epoch_history = self.sub_train(train_loader, val_interval) # one train epoch 

            epoch_end_time = datetime.now()

            # --- Display global info about the train epoch --- #
            if self.rank == 0:
                log.info(f"(Train) Epoch : {epoch + 1} completed in {(epoch_end_time - epoch_start_time)} | Iteration : {self.iteration} ")
                log.info(f"Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info(f"Metrics over the (train) epoch {', '.join(f'{k}: {v:.5g}' for k, v in outputs_epoch_history.items())}")

            
            # ---- Starting the validation epoch ---- #
            epoch_start_time = datetime.now()
            if ( self.rank == 0 ):
                log.info("")
                log.info(f" -- Validation epoch starting at {epoch_start_time}")

            if self.is_distributed:
                val_loader.sampler.set_epoch(val_loader.sampler.epoch) # Previously +1, why?
                       
            outputs_epoch_history = self.validate_v2(val_loader) # one validation epoch

            epoch_end_time = datetime.now()

            # --- Display global info about the validation epoch --- #
            if self.rank == 0:
                log.info(f" -- Validation epoch completed in {epoch_end_time - epoch_start_time} | Iteration : {self.iteration}")
                log.info(f" -- Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info(f" -- Metrics over the (val) epoch {', '.join(f'{k}: {v:.5g}' for k, v in outputs_epoch_history.items())}")
                                
                # --- Logs ---- #
                log_entries = {
                    "iteration": self.iteration, 
                    "epoch": self.epoch, 
                    **outputs_epoch_history, 
                    "saved_best": False
                }

                # Save if this is the best model so far
                if outputs_epoch_history["loss"] < self.best_validation_loss:
                    log.info(" ... Best validation loss so far!")
                    self.best_validation_loss = outputs_epoch_history["loss"]
                    log_entries["saved_best"] = True

                    self.save_state(suffix="_BEST")

                elif checkpointing:
                    # if checkpointing = True the model is saved at the end of each validation epoch
                    self.save_state()
                
                self.val_log.log(log_entries)
                

        self.train_log.close() # Closing the .csv train file
        if self.rank == 0:
            self.val_log.close() # Closing the .csv val file



    def sub_train(self, loader, val_interval):

        self.module.train() # Set model to training mode
        outputs_epoch_history = {'loss': 0, 'accuracy': 0}
        
        for step, train_data in enumerate(loader):
            
            # Mount the batch of data to the device
            self.data = train_data['data'].to(self.device)
            self.target = train_data[self.truth_key].to(self.device)                
            
            # Call forward: make a prediction & measure the average error using data = self.data
            outputs = self.forward(forward_type='train')
            
            # Call backward: back-propagate error and update weights using loss = self.loss
            self.loss = outputs['loss']
            self.backward()
            
            # run scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # If not detaching now ( with .item() ) all the data of the epoch will be load into GPU memory
            # v.item() converts torch.tensors to python floats (and detachs + moves to cpu)
            outputs = {k: v.item() for k, v in outputs.items()} 
            outputs_epoch_history['loss']     += outputs['loss']
            outputs_epoch_history['accuracy'] += outputs['accuracy']
            
            #  --- Logs --- #
            log_entries = {
                "iteration": (self.iteration + step), 
                "epoch": self.epoch, 
                **outputs
            }
            self.train_log.log(log_entries) # add logs in the .csv file (each process has its csv file)
            

            # --- Display --- #
            if ( step  % val_interval == 0 ):
                log.info(f"GPU : {self.device} | Steps : {step + 1}/{len(loader)} | Iteration : {self.iteration + step} | Batch Size : {loader.batch_size}")
                #print(f"GPU : {self.device} | Steps : {step} | Iteration : {self.iteration + step} | Batch Size : {loader.batch_size}")
                
                if ( self.rank == 0 ) :
                    pass
                    #log.info(f" Iteration {self.iteration + step}, train loss : {outputs['loss']:.5f}, accuracy : {outputs['accuracy']:.5f}")
                    
        
        self.iteration += ( step + 1 )

        # Take the mean over the epoch
        outputs_epoch_history['loss'] /= (step + 1)
        outputs_epoch_history['accuracy'] /= (step +1)
 
        return outputs_epoch_history

    def validate_v2(self, loader):

        self.module.eval()
        outputs_epoch_history = {'loss': 0., 'accuracy': 0.}

        with torch.no_grad():
            

            for step, val_batch in enumerate(loader):
        
                # Mount the batch of data to the device
                self.data = val_batch['data'].to(self.device)
                self.target = val_batch[self.truth_key].to(self.device)
                
                # evaluate the network
                outputs = self.forward(forward_type='val') # output is a dictionnary with torch.tensors
                
                # In case of ddp we reduce outputs to get the global performance
                # Note : It is currently done at each step to optimize gpu memory usage
                # But this could also be perform at the end of the validation epoch
                if self.is_distributed:
                    outputs = self.get_synchronized(outputs)

                # Detaching outputs tensors ( with .item() )
                # otherwise all the data of the epoch will be load into GPU memory
                # v.item() converts torch.tensors to python floats (and detachs + moves to cpu)
                outputs = {k: v.item() for k, v in outputs.items()} 

            
                if self.rank == 0: 
                    # --- Storing performances --- #
                    outputs_epoch_history['loss']     += outputs['loss']
                    outputs_epoch_history['accuracy'] += outputs['accuracy']

            # Take the mean over the epoch
            outputs_epoch_history['loss'] /= (step + 1)
            outputs_epoch_history['accuracy'] /= (step +1)
            
            return outputs_epoch_history

    #  --- Step Logs --- # For now I see no reason to log at each step of the validation
    # But I keep the code here if needed
    # log_entries = {
    #     "iteration": (self.iteration + step), 
    #     "epoch": self.epoch, 
    #     **outputs
    # }

    # self.val_log.log(log_entries) # add logs in the .csv file (each process has its csv file)


    def validate(self, val_iter, num_val_batches, checkpointing):
        """
        Perform validation with the current state, on a number of batches of the validation set.

        Parameters
        ----------
        val_iter : iter
            Iterator of the validation dataset.
        num_val_batches : int
            Number of validation batches to iterate over.
        checkpointing : bool
            Whether to save the current state to disk.
        """
        # set model to eval mode
        self.model.eval()
        val_metrics = None

        for val_batch in range(num_val_batches): # num_val_batches is defined in grant_gnn_train et pas basé sur le val_dataset ??
            
            # get validation data mini-batch
            try:
                val_data = next(val_iter)
            
            except StopIteration:
                del val_iter
                if self.is_distributed:
                    self.data_loaders["validation"].sampler.set_epoch(self.data_loaders["validation"].sampler.epoch+1)
                
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)
            
            # extract the event data and target from the input data dict
            self.data = val_data['data'].to(self.device)
            self.target = val_data[self.truth_key].to(self.device)
            
            # evaluate the network
            outputs, metrics = self.forward(False)
            if val_metrics is None:
                val_metrics = metrics
            else:
                for k, v in metrics.items():
                    val_metrics[k] += v

        # record the validation stats to the csv
        val_metrics = {k: v/num_val_batches for k, v in val_metrics.items()}
        val_metrics = self.get_synchronized_metrics(val_metrics)

        if self.rank == 0:
            log_entries = {"iteration": self.iteration, "epoch": self.epoch, **val_metrics, "saved_best": False}
            
            # Save if this is the best model so far
            print(f"  Validation {', '.join(f'{k}: {v:.5g}' for k, v in val_metrics.items())}", end="\n")
            
            if val_metrics["loss"] < self.best_validation_loss:
                print(" ... Best validation loss so far!\n")
                self.best_validation_loss = val_metrics["loss"]
                self.save_state(suffix="_BEST")
                log_entries["saved_best"] = True
            else:
                print("")
           
            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()
            
            self.val_log.log(log_entries)
        

    def evaluate(self, report_interval=20):
        """Evaluate the performance of the trained model on the test set."""
        
        if self.rank == 0:
            log.info(f"\n\nEnd of the run. Test epoch starting.\nOutput directory: {self.dump_path}")
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # evaluation loop
            start_time = datetime.now()
            step_time = start_time
            steps_per_epoch = len(self.data_loaders["test"])
            
            for self.step, eval_data in enumerate(self.data_loaders["test"]):
                
                self.data = eval_data['data'].to(self.device)
                self.target = eval_data[self.truth_key].to(self.device)

                metrics = self.forward(forward_type='test') # will ouput loss + accuracy + softmax of the preds
                outputs = {'softmax': metrics.pop('softmax')} 

                # Add the local result to the final result
                if self.step == 0:
                    indices = eval_data['indices']
                    targets = self.target
                    eval_outputs = outputs
                    eval_metrics = metrics
                else:
                    indices = torch.cat((indices, eval_data['indices']))
                    targets = torch.cat((targets, self.target))
                    
                    for k in eval_outputs.keys():
                        eval_outputs[k] = torch.cat((eval_outputs[k], outputs[k]))

                    for k in eval_metrics.keys():
                        eval_metrics[k] += metrics[k]
               
                # print the metrics at given intervals
                if self.rank == 0 and self.step % report_interval == 0:
                    previous_step_time = step_time
                    step_time = datetime.now()
                    average_step_time = (step_time - previous_step_time)/report_interval
                    print(f"Step {self.step}/{steps_per_epoch}"
                          f" Evaluation {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())},"
                          f" Step time {average_step_time},"
                          f" Total time {step_time-start_time}")
       
        for k in eval_metrics.keys():
            eval_metrics[k] /= self.step+1
       
        # if self.is_distributed:
        #     indices = self.data_loaders['test'].sampler.sampler.indices
        # else :
        #     indices = self.data_loaders['test'].sampler.indices 

        eval_outputs["indices"] = indices.to(self.device)
        eval_outputs[self.truth_key] = targets
       
        # Gather results from all processes
        eval_metrics = self.get_synchronized_metrics(eval_metrics)
        eval_outputs = self.get_synchronized_outputs(eval_outputs)
        
        if self.rank == 0:
            # Save overall evaluation results
            log.info("Saving Data...")
            for k, v in eval_outputs.items():
                np.save(self.dump_path + k + ".npy", v)
            # Compute overall evaluation metrics
            for k, v in eval_metrics.items():
                log.info(f"Average evaluation {k}: {v:.4f}")


    def new_evaluate(self, report_interval=20):
        """
        Evaluate the performance of the trained model on the test set.
        Multi-processes is supported
        
        Pour le futur  :
        
            Even if there more than one gpu, the test will be done 
            on the master (rank 0) one.
            (Nécessite de changer l'instantiation du test_sampler)
            To Do : compute the acceleration with multiple gpu : (Is n-gpus really usefull at test time ?)
        """

        self.module.eval()        
        start_time = datetime.now()
        loader = self.data_loaders['test']

        indices = None

        if self.rank == 0:
            log.info(f"\n\nEnd of the training. Evaluation starting. \nOutput directory: {self.dump_path}")

        for step, test_data in enumerate(loader):

            self.data = test_data['data'].to(self.device)
            self.target = test_data[self.truth_key].to(self.device)

            metrics = self.forward(forward_type='test') # will ouput loss + accuracy + softmax of the preds
            outputs = {'softmax': metrics.pop('softmax')} 



    def save_state(self, suffix="", name=None):
        """
        Save model weights and other training state information to a file.

        Parameters
        ==========
        suffix : string
            The suffix for the filename. Should be "_BEST" for saving the best validation state.
        name : string
            The name for the filename. By default, use the engine class name followed by model class name.

        Returns
        =======
        filename : string
            Filename where the saved state is saved.
        """
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
       
        filename = f"{self.dump_path}{name}{suffix}.pth"
        
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.module.state_dict()
        
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        
        log.info(f"Saved state as: {filename}")
        
        return filename

    def restore_state(self, weight_file):
        """Restore model and training state from a given filename."""
        
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            log.info(f"Restoring state from {weight_file}\n")
            # prevent loading while DDP operations are happening
            if self.is_distributed:
                torch.distributed.barrier()
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f, map_location=self.device)
            # load network weights
            self.module.load_state_dict(checkpoint['state_dict'])
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']

    def restore_best_state(self, name=None):
        """Restore model using best model found in current directory."""
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        self.restore_state(f"{self.dump_path}{name}_BEST.pth")

