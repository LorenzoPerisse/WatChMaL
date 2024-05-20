import torch

from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.utils.logging_utils import setup_logging


log = setup_logging(__name__)


class RegressionEngine(ReconstructionEngine):
    """Engine for performing training or evaluation for a regression network."""
    def __init__(
        self, 
        truth_key, 
        model, 
        rank, 
        device, 
        dump_path, 
        wandb_run=None,
        dataset=None,
        output_center=0, 
        output_scale=1
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
        output_center : float
            Value to subtract from target values
        output_scale : float
            Value to divide target values by
        """
        # create the directory for saving the log and dump files
        super().__init__(
            truth_key, 
            model, 
            rank, 
            device, 
            dump_path, 
            wandb_run=wandb_run,
            dataset=dataset)
        
        self.output_center = output_center # define for the cnn. No idea when it is used
        self.output_scale = output_scale   # neither why to do scaling this way

    def forward(self, forward_type='train'):
        """
        Compute predictions and metrics for a batch of data

        Parameters
        ==========
        train : bool
            Whether in training mode, requiring computing gradients for backpropagation

        Returns
        =======
        dict
            Dictionary containing loss and predicted values
        """

        outputs = {}
        grad_enabled = True if forward_type == 'train' else False

        with torch.set_grad_enabled(grad_enabled):
            # Previous version
            # model_out = self.model(self.data).reshape(self.target.shape)
            
            # scaled_target = self.scale_values(self.target)
            # scaled_model_out = self.scale_values(model_out)
            # self.loss = self.criterion(scaled_model_out, scaled_target)
             
            # New version version
            model_out = self.model(self.data)
            loss = self.criterion(model_out, self.target)

            #outputs = {"predicted_" + self.truth_key: model_out}
            #metrics = {'loss': self.loss}
            outputs['accuracy'] = torch.mean(model_out)
            outputs['loss']     = loss
        

        return outputs

    def scale_values(self, data):
        scaled = (data - self.output_center) / self.output_scale
        return scaled
