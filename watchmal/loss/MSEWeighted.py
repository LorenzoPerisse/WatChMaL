import torch.nn

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weights): # Only if weights are supposed to be fixed
        super().__init__()

        self.weights = torch.tensor([weights])
        self.mse     = torch.nn.MSELoss(reduction='none')

    def forward(self, input, targets):
        
        assert len(input) % len(self.weights) == 0, f"input should be divisable by the number of weights in weight. Received : {len(self.weights)} vs {len(input)}"
        squared_error = self.mse(input, targets)
        
        batch_weights = self.weights.repeat(len(input), 1) # Pq pas, tu ne peux pas faire plus compréhensible ?
        
        weighted_squared_error = squared_error * batch_weights
        loss = torch.mean(weighted_squared_error)

        return loss