import torch
import numpy as np

class EncoderLoss(torch.nn.Module):
    def __init__(self):
        super(EncoderLoss, self).__init__()
        self.__name__ = "EncoderLoss"
        print("EncoderLoss initialized.")

    def forward(self, W, X, Z1, Z2):
        """
        W: torch.Tensor, encoded 'spiketrain', not binarized yet
        X: torch.Tensor, input signal
        Z1: torch.Tensor, representation of Layer 1
        Z2: torch.Tensor, representation of Layer 2
        """
        assert isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor), "X and W must be torch tensors."
        
        if Z1 is not None:
            assert X.shape == Z1.shape, f"X and Z1 must have the same shape: {X.shape} != {Z1.shape}."
        if Z2 is not None:
            assert X.shape == Z2.shape, f"X and Z2 must have the same shape: {X.shape} != {Z2.shape}."
        
        n_samples = X.shape[0]
        n_timesteps = X.shape[1]
        W_unflat = W.reshape(n_samples, n_timesteps, -1)
        n_spikes_per_timestep = W_unflat.shape[2]
        W_sumspikes = torch.sum(W_unflat, dim=2) 
        # NOTE: Sum instead of mean, since mean will be around 0.5 or slightly lower or higher, because we have extremes [0, 1]
        # sum gives a semi-count of how many times we have near-1 values
        
        mi = compute_mutual_information(X, W_sumspikes)
        
        mi_Z1 = 0 
        mi_Z2 = 0
        if Z1 is not None:
            mi_Z1 = compute_mutual_information(X, Z1)
        if Z2 is not None:            
            mi_Z2 = compute_mutual_information(X, Z2)
        
        # Instead of punishing many spikes, i.e. every spike (L1 norm),
        # we can try punishing the difference in spikes from the area of the input signal
        
        area_X = torch.sum(X, dim=1)
        n_spikes = torch.sum(W, dim=1)
        
        # E.g., we want to have 525 spikes if the area of X is 525,
        # so we add a loss of the L1 or L2 distance:  
        L1 = torch.abs(n_spikes - (area_X * n_spikes_per_timestep))        
       
        MI = mi
        cnt = 1
        if Z1 is not None and Z2 is None:
            MI += mi_Z1
            cnt += 1
        if Z2 is not None:
            MI += mi_Z2
            cnt += 2
        loss = -(MI/cnt) + torch.mean(L1) 

        return loss
    
def compute_mutual_information(X, Z):
    """ 
    Computes the mutual information between two random variables X and Z.
    All computations are done using torch operations, to keep the gradient flow.
    """
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and W must have the same number of samples.")
    
    if X.ndim == 3:
        if X.size(2) == 1:
            X = X.squeeze(2)
    
    eps = torch.tensor(1e-12, dtype=torch.float32)
    joint_prob = torch.mean(torch.multiply(X, Z))
    px = torch.mean(X) # Marginal probability of X
    # px = px if px > 0 else eps
    assert px >= 0, f"Marginal prob. of X is smaller than 0, px: {px}"
    pz = torch.mean(Z) # Marginal probability of Z
    assert pz >= 0, f"Marginal prob. of Z is smaller than 0, pz: {pz}"
    
    if joint_prob == 0 or px == 0 or pz == 0:
        joint_prob = torch.max(joint_prob, eps)
        px = torch.max(px, eps)
        pz = torch.max(pz, eps)
        
    mutual_information = joint_prob * torch.log2((joint_prob) / (px * pz))
    # mutual_information = joint_prob * torch.log2((joint_prob) / px * pz)
    return mutual_information
