import torch
from snntorch import spikegen

class RateCoder(torch.nn.Module):
    def __init__(self, window_size, n_channels, n_spikes_per_timestep):        
        super().__init__()

        self.output_size = window_size * n_channels * n_spikes_per_timestep
        self.n_spikes_per_timestep = n_spikes_per_timestep

    def forward(self, x):
        # reshape
        x = x.view(x.size(0), -1)
        spikes = spikegen.rate(x, num_steps=self.n_spikes_per_timestep)     
        spikes = spikes.permute(1, 2, 0)
        spikes = spikes.reshape(spikes.size(0), -1)  
        # spikes, Z1, Z2
        return spikes, None, None
    
    def print_learnable_params(self):
        print("Total Learnable Parameters (encoder):", 0)

    def update_drop_p(self, drop_p):
        pass