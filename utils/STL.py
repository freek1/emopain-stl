import torch

class RepeatLayer(torch.nn.Module):
    def __init__(self, source_size, target_size):
        super(RepeatLayer, self).__init__()
        assert target_size > source_size, f"target_size must be greater than source_size: source={source_size}, target={target_size}."
        assert target_size % source_size == 0, f"target_size must be a multiple of source_size: source={source_size}, target={target_size}."
        
        self.repeat_param = target_size // source_size
        print(self.repeat_param)

    def forward(self, x):
        return x.repeat_interleave(self.repeat_param, dim=1)

class SpikeThresholdLearning(torch.nn.Module):
    def __init__(self, 
                 n_timesteps:int, 
                 n_spikes_per_timestep:int,
                 n_channels:int, 
                 l1_sz:int,
                 l2_sz:int,
                 drop_p:float):        
        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_spikes_per_timestep = n_spikes_per_timestep
        self.n_channels = n_channels
        self.l1_sz = l1_sz
        self.l2_sz = l2_sz  
        self.drop_p = drop_p

        input_size = n_timesteps * n_channels
        output_size = input_size * n_spikes_per_timestep

        self.output_size = output_size
        
        if self.l1_sz == 0:
            self.l2_sz = 0 # assumption we enforce
            
            if input_size == output_size:
                # Match, so no trouble
                self.match_out = torch.nn.Identity()
                pass
            elif input_size > output_size:
                # AvgPool to fit the input size
                if input_size % output_size != 0:
                    raise ValueError(f"input must be a multiple of output: X={input_size}, output={output_size}.")
                
                pool_size = output_size // input_size
                self.match_out = torch.nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
            elif input_size < output_size:
                # Repeat to fit the input size
                if output_size % input_size != 0:
                    raise ValueError(f"input must be a divisor of output: X={input_size}, output={output_size}.")
                
                self.match_out = RepeatLayer(input_size, output_size)
        
        if self.l1_sz > 0 and self.l2_sz == 0:
            self.lin1 = torch.nn.Linear(input_size, self.l1_sz)
            self.relu1 = torch.nn.ReLU()
            self.drop1 = torch.nn.Dropout(drop_p)
            self.bn1 = torch.nn.BatchNorm1d(self.l1_sz)
            
            if self.l1_sz == input_size:
                # Match, so no trouble
                self.match_Z1 = torch.nn.Identity()
                pass
            elif self.l1_sz > input_size:
                # MaxPool to fit the input size
                if self.l1_sz % input_size != 0:
                    raise ValueError(f"l1_sz must be a multiple of input_size: X={input_size}, l1_sz={self.l1_sz}.")
                
                pool_size = self.l1_sz // input_size
                self.match_Z1 = torch.nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            elif self.l1_sz < input_size:
                # Repeat to fit the input size
                if input_size % self.l1_sz != 0:
                    raise ValueError(f"l1_sz must be a divisor of input_size: X={input_size}, l1_sz={self.l1_sz}.")
                
                self.match_Z1 = RepeatLayer(self.l1_sz, input_size)
                
            self.match_out = RepeatLayer(l1_sz, output_size)

        
        if self.l1_sz > 0 and self.l2_sz > 0:
            self.lin1 = torch.nn.Linear(input_size, self.l1_sz)
            self.relu1 = torch.nn.ReLU()
            self.drop1 = torch.nn.Dropout(drop_p)
            self.bn1 = torch.nn.BatchNorm1d(self.l1_sz)
            
            if self.l1_sz == input_size:
                # Match, so no trouble
                self.match_Z1 = torch.nn.Identity()
                pass
            elif self.l1_sz > input_size:
                # MaxPool to fit the input size
                if self.l1_sz % input_size != 0:
                    raise ValueError(f"l1_sz must be a multiple of input_size: X={input_size}, l1_sz={self.l1_sz}.")
                
                pool_size = self.l1_sz // input_size
                self.match_Z1 = torch.nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
            elif self.l1_sz < input_size:
                # Repeat to fit the input size
                if input_size % self.l1_sz != 0:
                    raise ValueError(f"l1_sz must be a divisor of input_size: X={input_size}, l1_sz={self.l1_sz}.")
                
                self.match_Z1 = RepeatLayer(self.l1_sz, input_size)
            
            self.lin2 = torch.nn.Linear(self.l1_sz, self.l2_sz)
            self.relu2 = torch.nn.ReLU()
            self.drop2 = torch.nn.Dropout(drop_p)
            self.bn2 = torch.nn.BatchNorm1d(self.l2_sz)
            
            if self.l2_sz == input_size:
                # Match, so no trouble
                self.match_Z2 = torch.nn.Identity()
                pass
            elif self.l2_sz > input_size:
                # MaxPool to fit the input size
                if self.l2_sz % input_size != 0:
                    raise ValueError(f"l2_sz must be a multiple of input_size: X={input_size}, l2_sz={self.l2_sz}.")
                
                pool_size = self.l2_sz // input_size
                self.match_Z2 = torch.nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
            elif self.l2_sz < input_size:
                # Repeat to fit the input size
                if input_size % self.l2_sz != 0:
                    raise ValueError(f"l2_sz must be a divisor of input_size: X={input_size}, l2_sz={self.l2_sz}.")
                
                self.match_Z2 = RepeatLayer(self.l2_sz, input_size)
            
            self.match_out = RepeatLayer(self.l2_sz, output_size)
            
        self.threshold_adder = torch.nn.Parameter(torch.Tensor(output_size).uniform_(0.4, 0.6), requires_grad=True) # NOTE: write down because part: (0.4, 0.6) to not have to have many weight updates to go from e.g. random 0.9 initialization to 0.1
        
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1) # flatten the input
        
        Z1 = None
        Z2 = None
        
        if self.l1_sz == 0:
            x = self.match_out(x)
        
        if self.l1_sz > 0 and self.l2_sz == 0:
            x = self.lin1(x)
            x = self.drop1(x)
            if batch_size > 1:
                x = self.bn1(x)
            x = self.relu1(x)
                
            Z1 = self.match_Z1(x)
            
            x = self.match_out(x)
        
        if self.l1_sz > 0 and self.l2_sz > 0:
            x = self.lin1(x)
            x = self.drop1(x)
            if batch_size > 1:
                x = self.bn1(x)
            x = self.relu1(x)
            
            Z1 = self.match_Z1(x)
            
            x = self.lin2(x)
            x = self.drop2(x)
            if batch_size > 1:
                x = self.bn2(x)
            x = self.relu2(x)
                
            Z2 = self.match_Z2(x)

            x = self.match_out(x)

        extracted_feats = x

        # NOTE: thesholds break the gradient propagation, so they cannot be used, 
        # therefore we use a surrogate: sigmoid w/ slope=25
        alpha = 25.0
        thresholded_feats = torch.sigmoid(alpha * (extracted_feats - self.threshold_adder.unsqueeze(0)))
        
        # NOTE: clamp the thresholds to avoid numerical instability
        with torch.no_grad():
            self.threshold_adder.clamp_(0.001, 1.0)

        return thresholded_feats, Z1, Z2
    
    def update_drop_p(self, new_drop_p):
        self.drop1.p = new_drop_p
        self.drop2.p = new_drop_p

    def print_learnable_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Learnable Parameters (encoder):", total_params)
        