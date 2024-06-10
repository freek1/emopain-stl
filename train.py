import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import multiprocessing as mp
from sklearn.model_selection import train_test_split, LeaveOneOut
import os

from utils.STL import SpikeThresholdLearning
from utils.RateCoder import RateCoder
from utils.LatencyCoder import LatencyCoder
from utils.EncoderLoss import EncoderLoss
from utils.helpers import train_STL_encoder, classify_svm
from utils.load_data import load_data_emopain

def main(config: dict, input_data: torch.Tensor, target_labels: torch.Tensor, fold_num: int, train_index: list, test_index: list, device: torch.device, folder: str):
    data_type = config["data_type"]
    batch_sz = config["batch_sz"]
    window_size = config["window_size"]
    stride = config["stride"]
    n_spikes_per_timestep = config["n_spikes_per_timestep"]
    num_steps = config["num_steps"]
    encoder_epochs = config["encoder_epochs"]
    classifier_epochs = config["classifier_epochs"]
    theta = config["theta"]
    l1_sz = config["l1_sz"]
    l2_sz = config["l2_sz"]
    l1_cls = config["l1_cls"]
    drop_p = config["drop_p"]
    encoding_method = config["encoding_method"]
    avg_window_sz = config["avg_window_sz"]
    suff = config["suff"] # STL, rate, latency
    SVM = config["SVM"]
    SRNN = config["SRNN"]
    
    print(f"FOLDER = '{folder}'")

    # Split into test, val and train
    test_data, test_labels = input_data[test_index], target_labels[test_index]
    train_data, val_data, train_labels, val_labels = train_test_split(input_data[train_index], target_labels[train_index], test_size=0.2, random_state=42)
    
    n_samples, n_channels, n_timesteps = train_data.shape
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Count the number of positive and negative samples in the training set
    pos_count = (train_labels == 1).sum().item()
    neg_count = (train_labels == 0).sum().item()

    # Compute weights for each sample
    total_samples = len(train_labels)
    weights = torch.zeros(total_samples)
    weights[train_labels == 1] = 1.0 / pos_count
    weights[train_labels == 0] = 1.0 / neg_count
    
    # Create the WeightedRandomSampler
    # NOTE: num_samples = batch_sz*2, since we want to over-sample the minority class.
    num_samples = batch_sz * 2
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    # Create DataLoaders with the sampler
    train_loader = DataLoader(train_dataset, batch_size=num_samples, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
    
    # Initialize the encoder
    if encoding_method == "rate":
        encoder = RateCoder(window_size=window_size, n_channels=n_channels, n_spikes_per_timestep=n_spikes_per_timestep)
    elif encoding_method == "latency":
        encoder = LatencyCoder(window_size=window_size, n_channels=n_channels, n_spikes_per_timestep=n_spikes_per_timestep)
    elif encoding_method == "STL":
        encoder = SpikeThresholdLearning(window_size=window_size, n_spikes_per_timestep=n_spikes_per_timestep, n_channels=n_channels, l1_sz=l1_sz, l2_sz=l2_sz, drop_p=drop_p)
        encoder_optimizer = AdamW(encoder.parameters(), lr=0.005)
        encoder_loss_fn = EncoderLoss()
    else:
        raise ValueError(f"Invalid encoding method: {encoding_method}")
    
    print(f"Encoder params: \t{sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    
    # Train the encoder, if method is STL
    if encoding_method == "STL":
        encoder = train_STL_encoder(encoder, device, train_loader, val_loader, encoder_optimizer, encoder_loss_fn, encoder_epochs, folder, verbose=True)
    
    # Get/save the spike-trains
    if not os.path.exists(f"{folder}/spiketrains"):
        os.makedirs(f"{folder}/spiketrains")
        
        with torch.no_grad():
            for i_subject, (X, y) in enumerate(train_loader):
                spk_subj = []
                for window in range(0, X.size(2) - window_size, window_size): # NOTE: No stride here
                    X_window = X[:, :, window:window + window_size]
                    X_window = X_window.to(device)
                    y = y.to(device)
                    
                    spiketrain, Z1, Z2 = encoder(X_window)
                    spk_inputs = (spiketrain > theta).type(torch.float).cpu().detach().numpy()
                    spk_subj = np.append(spk_subj, spk_inputs)
                np.save(f"spiketrains/{folder}/spiketrain_{i_subject}_{suff}.npy", spk_subj.flatten())
                
                # TODO: Compute sparsity here!!! And save with
    else:
        print("Spiketrains already exist. Skipping...")
         
    # Initialize the classifier
    if SVM:
        classify_svm(train_labels, test_labels, encoder, n_timesteps, window_size, n_spikes_per_timestep, n_channels, theta, folder, data_type, suff, fold_num, avg_window_sz)
        
    if SRNN:
        classify_srnn()
        
if __name__ == "__main__":
    mp.set_start_method('spawn') # Fix for Linux systems deadlock
    torch.manual_seed(1957)
    np.random.seed(1957)
    
    device = torch.device("cpu")
    
    data_type = "emg"
    batch_sz = 4
    window_size = 3000
    stride = window_size // 4
    n_spikes_per_timestep = 10
    num_steps = 10
    encoder_epochs = 75
    classifier_epochs = 10
    theta = 0.99
    l1_sz = 3000
    l2_sz = 3000
    l1_cls = 3000
    drop_p = 0.0
    encoding_method = "STL"
    avg_window_sz = 100
    fold_num = 0

    SVM = True
    SRNN = False
    
    if encoding_method == "rate":
        suff = "_rate"
    elif encoding_method == "latency":
        suff = "_latency"
    else: suff = ""
    
    config = {
        "data_type": data_type,
        "batch_sz": batch_sz,
        "window_size": window_size,
        "stride": stride,
        "n_spikes_per_timestep": n_spikes_per_timestep,
        "num_steps": num_steps,
        "encoder_epochs": encoder_epochs,
        "classifier_epochs": classifier_epochs,
        "theta": theta,
        "l1_sz": l1_sz,
        "l2_sz": l2_sz,
        "l1_cls": l1_cls,
        "drop_p": drop_p,
        "encoding_method": encoding_method,
        "avg_window_sz": avg_window_sz,
        "suff": suff,
        "SVM": SVM,
        "SRNN": SRNN
    }
    
    cv = LeaveOneOut()
    input_data, target_labels = load_data_emopain(data_type)
    print("data loaded: " )
    
    if SRNN:
        folder = "emopain_srnn"
    elif SVM:
        folder = "emopain_svm"
    if l1_sz > 0:
        folder += "_S" # stacked
    else:
        folder += "_V" # vanilla
        
    os.makedirs(f"results/{folder}", exist_ok=True)
    os.makedirs(f"imgs/{folder}", exist_ok=True) 

    args = []
    for fold_num, (train_index, test_index) in enumerate(cv.split(input_data, target_labels)):
        args.append((config, input_data, target_labels, fold_num, train_index, test_index, device, folder))
    
        if len(args) == 1:
            break
    
    main(*args[0])