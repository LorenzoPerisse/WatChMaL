import torch


from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)

class ClassifierEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a classification network."""
    def __init__(
            self, 
            truth_key, 
            model, 
            rank, 
            device, 
            dump_path,
            wandb_run=None, 
            dataset=None,
            flatten_model_output=False, 
            prediction_threshold=None,
            label_set=None
        ):
        """
        Parameters
        ==========
        truth_key : string
            Name of the key for the target labels in the dictionary returned by the dataloader
        model
            `nn.module` object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N).
        """
        # create the directory for saving the log and dump files
        super().__init__(
            truth_key, 
            model, 
            rank, 
            device, 
            dump_path,
            wandb_run=wandb_run,
            dataset=dataset
        )
        
        self.flatten_model_output = flatten_model_output
        self.prediction_threshold = prediction_threshold
        self.label_set = label_set

        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

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
        super().configure_data_loaders(data_config, loaders_config)

        if self.label_set is not None:
            for name in loaders_config.keys():
                self.data_loaders[name].dataset.map_labels(self.label_set)

    def forward(self, forward_type='train'):
        """
        Compute predictions and metrics for a batch of data.

        Parameters
        ==========
        forward_type : (str) either 'train', 'val' or 'test'
            Whether in training mode, requiring computing gradients for backpropagation
            For 'test' also returns the softmax value in outputs
        Returns
        =======
        dict
            Dictionary containing loss, predicted labels, softmax, accuracy, and raw model outputs
        """
        metrics = {}
        outputs = {}
        grad_enabled = True if forward_type == 'train' else False

        with torch.set_grad_enabled(grad_enabled):
    
            model_out = self.model(self.data) # even in ddp, the forward is done with self.model and not self.module
            
            # Compute the loss
            if self.flatten_model_output:
                model_out = torch.flatten(model_out)

            self.target = self.target.reshape(-1)
            loss = self.criterion(model_out, self.target)

            # Apply softmax to model_out
            if self.flatten_model_output:
                softmax = self.sigmoid(model_out)
            else: 
                softmax = self.softmax(model_out)

            # Compute accuracy based on the softmax values
            if self.flatten_model_output:
                preds = ( softmax >= self.prediction_threshold )
            else :
                preds = torch.argmax(model_out, dim=-1)
            
            accuracy = (preds == self.target).sum() / len(self.target)

            # Add the metrics to the output dictionnary
            metrics['loss']     = loss
            metrics['accuracy'] = accuracy

            # Note : this softmax saving will be modified. Even maybe deleted
            if forward_type == 'test': # In testing mode we also save the softmax values
                outputs['pred'] = model_out

        # metrics and potentially outputs contains tensors linked to the gradient graph (and on gpu is any) 
        return metrics, outputs


        # if not train:
            # print(f"\nModel out : {model_out.shape}, {model_out}\n")
            # print(f"\nTarget out : {self.target.shape}, {self.target}\n")
            # print(f"\n Softmax : {softmax}\n")
            # print(f"\nPredicted_labels : {predicted_labels}\n")
        
        # if needed one day : predicted_labels.nelement() see https://pytorch.org/docs/stable/generated/torch.numel.html#torch.numel (nelement is an alias for .numel())
