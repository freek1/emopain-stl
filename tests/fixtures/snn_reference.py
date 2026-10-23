# Exact selected definitions from emopain-snn bd50395d (2024-06-08).
# Used only for numerical parity tests; no experimental entry point is imported.
import torch
from torch import nn
import numpy as np
import snntorch as snn
import time


def normalize_data2(inputs):
    """
    Normalize the data between [0, 1].
    """
    start = time.time()

    # Mask NaNs for min and max calculations
    nan_mask = torch.isnan(inputs)
    inputs_masked = torch.where(nan_mask, torch.tensor(float('inf')).to(inputs.device), inputs)
    min_vals = torch.min(inputs_masked, dim=1, keepdim=True)[0]

    inputs_masked = torch.where(nan_mask, torch.tensor(float('-inf')).to(inputs.device), inputs)
    max_vals = torch.max(inputs_masked, dim=1, keepdim=True)[0]

    # Avoid division by zero by setting zero max values to 1
    max_vals[max_vals == 0] = 1.0

    # Normalize the data
    inputs = (inputs - min_vals) / max_vals

    # Replace NaNs with 0
    inputs = torch.nan_to_num(inputs, nan=0.0)

    # print("normalize vect: ", time.time() - start)
    return inputs

def compute_mutual_information(X, W):
    if X.shape[0] != W.shape[0]:
        raise ValueError("X and W must have the same number of samples.")

    if X.ndim == 3:
        if X.size(2) == 1:
            X = X.squeeze(2)

    eps = 1e-12
    joint_prob = torch.mean(torch.multiply(X, W))  # Joint probability
    px = torch.mean(X)  # Marginal probability of X
    # px = px if px > 0 else eps
    assert px >= 0, f"Marginal prob. of X is smaller than 0, px: {px}"
    pw = torch.mean(W)  # Marginal probability of W
    # pw = pw if pw > 0 else eps
    assert pw >= 0, f"Marginal prob. of W is smaller than 0, pw: {pw}"
    # Eps needed when there are no spikes, mean(W) = 0
    if joint_prob == 0:
        joint_prob = eps
    if px == 0:
        px = eps
    if pw == 0:
        pw = eps
    mutual_information = joint_prob * torch.log2((joint_prob) / px * pw)

    return mutual_information

class EncoderLoss(nn.Module):
    def __init__(self):
        super(EncoderLoss, self).__init__()
        # self.mi_score = MutualInfoScore()
        self.__name__ = "EncoderLoss"
        print("EncoderLoss initialized.")

    def forward(self, W, X, Z1, Z2):
        """
        W: torch.Tensor, encoded 'spiketrain', not binarized yet
        X: torch.Tensor, input signal
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
        #           samples, timesteps, spikes_per_timestep
        W_sumspikes = torch.sum(W_unflat, dim=2) # NOTE: BECAUSE, Sum isntead of mean, since mean will be around 0.5 or slightly lower or higher, because we have extremes [0, 1]
        # sum gives a semi-count of how many times we have near-1 values

        # mi = 0.
        # # Do minibatch calculation of mutual information: 10% size
        minibatch_samples = np.random.choice(n_samples, 1, replace=False)
        # for i_sample in minibatch_samples:
        #     mi += torch_mutual_info(X[i_sample, :], W_sumspikes[i_sample, :])
        # mi /= n_samples
        mi = compute_mutual_information(X, W_sumspikes)

        mi_Z1 = 0
        mi_Z2 = 0
        if Z1 is not None:
            # mi_Z1 = 0.
            # for i_sample in minibatch_samples:
            #     mi_Z1 += torch_mutual_info(X[i_sample, :], Z1[i_sample, :])
            # mi_Z1 /= n_samples
            mi_Z1 = compute_mutual_information(X, Z1)
        if Z2 is not None:
            # mi_Z2 = 0.
            # for i_sample in minibatch_samples:
            #     mi_Z2 += torch_mutual_info(X[i_sample, :], Z2[i_sample, :])
            # mi_Z2 /= n_samples
            # for i in range(n_samples):
            #     mi_Z2 += torch_mutual_info(X[i, :], Z2[i, :])

            mi_Z2 = compute_mutual_information(X, Z2)

        # Instead of punishing many spikes, i.e. every spike (L1 norm),
        # we can try punishing the difference in spikes from the area of the input signal

        # [batchsize, timesteps]
        area_X = torch.sum(X, dim=1)
        # [batchsize, area]
        n_spikes = torch.sum(W, dim=1)

        # E.g., we want to have 525 spikes if the area of X is 525,
        # so we add a loss of the L1 or L2 distance:

        L1 = torch.abs(n_spikes - (area_X * n_spikes_per_timestep))
        L2 = torch.square(n_spikes - area_X)

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

class RecurrentClassifier3(torch.nn.Module):
    def __init__(self, encoder_output_size, lif_beta=0.5, l1_sz=1000, n_classes=2, window_size=1000, stride=800, num_steps=50):
        super().__init__()

        print("Using Recurrent SNN Classifier 3")

        self.window_size = window_size
        self.stride = stride
        self.num_steps = num_steps

        V1 = 0.5 # shared recurrent connection
        # V2 = torch.rand(n_classes) # unshared recurrent connections

        self.fc1 = torch.nn.Linear(encoder_output_size, l1_sz)
        self.rlif1 = snn.RLeaky(beta=lif_beta, linear_features=l1_sz, learn_beta=True, learn_recurrent=True)
        self.fc2 = torch.nn.Linear(l1_sz, n_classes)
        self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1, learn_beta=True, learn_recurrent=True)

    def forward(self, x):
        # At timestep 0, initalize spk1 and mem1
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()

        spk_rec = []
        mem_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class RepeatLayer(torch.nn.Module):
    def __init__(self, source_size, target_size):
        super(RepeatLayer, self).__init__()
        assert target_size > source_size, f"target_size must be greater than source_size: source={source_size}, target={target_size}."
        assert target_size % source_size == 0, f"target_size must be a multiple of source_size: source={source_size}, target={target_size}."

        self.repeat_param = target_size // source_size
        print(self.repeat_param)

    def forward(self, x):
        return x.repeat_interleave(self.repeat_param, dim=1)

class ThresholdedFeatureExtractor(torch.nn.Module):
    def __init__(self,
                 n_timesteps:int,
                 n_spikes_per_timestep:int,
                 n_channels:int,
                 l1_sz:int,
                 drop_p:float,
                 conv:bool,
                 l2_sz:int=0):
        super().__init__()

        self.conv = conv
        self.n_timesteps = n_timesteps
        self.n_spikes_per_timestep = n_spikes_per_timestep
        self.n_channels = n_channels
        self.l1_sz = l1_sz
        self.l2_sz = l2_sz
        self.drop_p = drop_p

        if conv: # Using conv to compress input
            n_out_channels = 1
            self.n_out_channels = n_out_channels
            input_size = n_timesteps * n_out_channels # 1 channel after compression
            output_size = input_size * n_spikes_per_timestep

            # For compressing input
            self.conv1 = torch.nn.Conv1d(in_channels=n_channels, out_channels=n_out_channels, kernel_size=n_channels, padding='same')
            self.avgpool1 = torch.nn.AvgPool1d(kernel_size=n_out_channels, stride=1)

        else: # Flattening and thus increasing input size
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
        if self.conv:
            x = x.reshape((batch_size, self.n_channels, self.n_timesteps))  # (batch_size, num_channels, n_timesteps)
            x = self.conv1(x) # perform same convolution on the channels to compress the input
            x = self.avgpool1(x)
            x = x.view(batch_size, -1)
        else:
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

            if np.isnan(x.detach().cpu().numpy()).any():
                print("nan in encodr B4 matchZ1")
                print(x)
                exit(0)

            Z1 = self.match_Z1(x)

            if np.isnan(x.detach().cpu().numpy()).any():
                print("nan in encoder after Z1")
                print(x)
                exit(0)

            x = self.match_out(x)

            if np.isnan(x.detach().cpu().numpy()).any():
                print("nan in encoder forward")
                print(x)
                exit(0)

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

        # NOTE: thesholds break the gradient propagation, so they cannot be used (like before),
        # therefore we use a surrogate: sigmoid w/ slope=25
        alpha = 25.0
        thresholded_feats = torch.sigmoid(alpha * (extracted_feats - self.threshold_adder.unsqueeze(0)))

        with torch.no_grad():
            self.threshold_adder.clamp_(0.001, 1.0)

        return thresholded_feats, Z1, Z2

    def update_drop_p(self, new_drop_p):
        self.drop1.p = new_drop_p
        self.drop2.p = new_drop_p

    def print_learnable_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Learnable Parameters (encoder):", total_params)
