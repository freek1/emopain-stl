import csv
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import multiprocessing as mp
from sklearn.model_selection import train_test_split, LeaveOneOut
import pandas as pd
import os
import time
import glob
import json

from utils.STAL import SpikeThresholdLearning
from utils.RateCoder import RateCoder
from utils.LatencyCoder import LatencyCoder
from utils.EncoderLoss import EncoderLoss
from utils.helpers import train_STL_encoder, train_SRNN_classifier, train_SRNN_classifier_nowindow, classify_svm, classify_srnn, classify_srnn_nowindow
from utils.load_data import load_data_emopain

# See __main__ below for config settings.

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
    l2_cls = config["l2_cls"]
    drop_p = config["drop_p"]
    encoding_method = config["encoding_method"]
    avg_window_sz = config["avg_window_sz"]
    suff = config["suff"] # STL, rate, latency
    SVM = config["SVM"]
    SRNN = config["SRNN"]
    generate_spikes = config["generate_spikes"]
    
    print(f"FOLDER = '{folder} {encoding_method} {data_type}{suff}'")

    # Split into test, val and train
    test_data, test_labels = input_data[test_index], target_labels[test_index]
    train_data, val_data, train_labels, val_labels = train_test_split(input_data[train_index], target_labels[train_index], test_size=0.2, random_state=42)
    
    n_samples, n_timesteps, n_channels = train_data.shape
    
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
    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)

    # Create DataLoaders with the sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, sampler=sampler)
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
    
    # Get/save the spike-trains
    saved_spiketrains = glob.glob(f"results/{folder}/spiketrains/labels_*_{data_type}_{fold_num}{suff}.npy")
    if len(saved_spiketrains) < 3 or generate_spikes:
        print(f"Generating spiketrain...: results/{folder}/spiketrains/labels_train_{data_type}_{fold_num}{suff}.npy")
        
        # Train the encoder, if method is STL
        if encoding_method == "STL":
            encoder = train_STL_encoder(encoder, device,
                        train_loader, val_loader,
                        encoder_optimizer, encoder_loss_fn,
                        encoder_epochs, window_size,
                        stride, folder, data_type,
                        fold_num, suff, verbose = True)
        
        generate_spiketrains(encoder, train_loader, fold_num, theta, suff, "train", data_type, window_size, device, folder)
        generate_spiketrains(encoder, val_loader, fold_num, theta, suff, "val", data_type, window_size, device, folder)
        generate_spiketrains(encoder, test_loader, fold_num, theta, suff, "test", data_type, window_size, device, folder)
    else:    
        print("Spiketrains already generated: ", f"results/{folder}/spiketrains/train_{data_type}_{fold_num}{suff}.npy")
    
    # Initialize the classifier
    if SVM:
        classify_svm(n_spikes_per_timestep, n_channels, folder, data_type, suff, fold_num, avg_window_sz)
        
    if SRNN:
        classifier = train_SRNN_classifier_nowindow(batch_sz, n_spikes_per_timestep, n_channels, data_type, num_steps, encoder, l1_cls, l2_cls, window_size, stride, device, folder, suff, fold_num, classifier_epochs)
        classify_srnn_nowindow(classifier, 0, folder, data_type, fold_num, suff, device, window_size, n_channels, n_spikes_per_timestep)
        
def generate_spiketrains(encoder, loader, fold_num, theta, suff, split, data_type, window_size, device, folder):
    """ Code to generate spiketrains from the encoder. 
    Also saves them"""
    batch_spiketrains = []
    batch_labels = []
    for X, y in loader:
        spk_batch = []
        for window in range(0, X.size(1) - window_size + 1, window_size):
            X_window = X[:, window:window + window_size].to(device)
            X_window = X_window.nan_to_num_(0)
            
            spiketrain, Z1, Z2 = encoder(X_window)
            spiketrain_cpu = spiketrain.cpu().detach().numpy()
            spk_inputs = (spiketrain_cpu > theta).astype(np.float32)
            spk_batch.append(spk_inputs)
        
        spk_batch = np.concatenate(spk_batch, axis=1)
        batch_spiketrains.append(spk_batch)
        batch_labels.append(y)
    
    spiketrains = np.vstack(batch_spiketrains)
    labels = np.hstack(batch_labels)
    np.save(f"results/{folder}/spiketrains/{split}_{data_type}_{fold_num}{suff}.npy", spiketrains)
    np.save(f"results/{folder}/spiketrains/labels_{split}_{data_type}_{fold_num}{suff}.npy", labels)
    
    # For test, we compute the sparsity, since its the single subject (then we have 1 for each subj)
    if split == "test":
        n_spikes = np.sum(spiketrains)
        total_spikes = np.prod(spiketrains.shape)
        sparsity = n_spikes / total_spikes

        with open(f'results/{folder}/spiketrains/sparsities_{data_type}{suff}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fold_num, sparsity])
        
if __name__ == "__main__":
    mp.set_start_method('spawn') # Fix for Linux systems deadlock
    torch.manual_seed(1957)
    np.random.seed(1957)
    
    device = torch.device("cuda") # cuda

    data_types = ["emg, energy, angle"] # emg, energy, angle
    batch_sz = 46 # Gets overridden later for specific data_type
    window_size = 3000 # Used for the encoder
    stride = window_size // 4 # 75% overlap
    n_spikes_per_timestep = 5
    num_steps = 0 # Recurrent steps for the SRNN [unnused]
    encoder_epochs = 30
    classifier_epochs = 50
    theta = 0.99 # Threshold parameter for making spiketrains (semi-binary floats to actual ints)
    l1_sz = 3000 # Size of the first layer in the STL encoder
    l2_sz = 3000 # Size of the second layer in the STL encoder
    l1_cls = 500 # Size of the layer in the classifier
    l2_cls = 0 # Set to 0 to ignore
    drop_p = 0.5 # Dropout setting
    encoding_method = "STL" # rate, latency, STL
    # NOTE: To activate the STL-Stacked, set l1sz (and l2sz) to your liking > 0
    #       To use STL-Vanilla, set l1_sz=l2_sz=0.
    avg_window_sz = 100 # For averaging the spiketrains to use as features for the SVM classifier

    generate_spikes = False # Override for making the spiketrains (if they are already generated in the folder)

    # Set either one to True
    SVM = False
    SRNN = True
    
    # Batch sizes for each data type
    bsz = [None, None, None]
    if encoding_method == "rate":
        suff = "_rate"
        bsz = [32, 4, 16]
    elif encoding_method == "latency":
        suff = "_latency"
        bsz = [4, 16, 16]
    elif encoding_method == "STL" and l1_sz == 0:
        suff = "_STL-V"
        bsz = [8, 4, 8]
    elif encoding_method == "STL" and l1_sz > 0:
        suff = "_STL-S"
        bsz = [32, 8, 16]
        
    # override bsz:
    bsz = [4] * 3
    
    args = []
    for data_type in data_types:
        # Batch size findings from hyperparam search
        if data_type == "emg":
            batch_sz = bsz[0]
        if data_type == "energy":
            batch_sz = bsz[1]
        if data_type == "angle":
            batch_sz = bsz[2]
        
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
            "l2_cls": l2_cls,
            "drop_p": drop_p,
            "encoding_method": encoding_method,
            "avg_window_sz": avg_window_sz,
            "suff": suff,
            "SVM": SVM,
            "SRNN": SRNN,
            "generate_spikes": generate_spikes
        }
        
        # Leave One Subject Out crossval
        cv = LeaveOneOut()
        input_data, target_labels = load_data_emopain(data_type)
        print(f"{data_type.capitalize()} data loaded:", input_data.shape, target_labels.shape)
        
        if SRNN:
            # folder = f"fixmi/emopain_srnn_{n_spikes_per_timestep}sp_{drop_p}dp"
            folder = f"emopain_protective"
        elif SVM:
            folder = f"emopain_svm_{n_spikes_per_timestep}sp"
            
        os.makedirs(f"results/{folder}", exist_ok=True)
        os.makedirs(f"results/{folder}/spiketrains", exist_ok=True)
        os.makedirs(f"imgs/{folder}", exist_ok=True) 

        # Load the configurations in a list, to be executed later.
        for fold_num, (train_index, test_index) in enumerate(cv.split(input_data, target_labels)):
            cls = "svm" if SVM else "srnn" if SRNN else "---"
            # Check if the spiketrain image is generated, if so, no need to compute whole fold again. 
            if os.path.exists(f"imgs/{folder}/spiketrain_{cls}_{data_type}{suff}_{fold_num}.png"):    
                print("Skipping fold", fold_num, data_type, suff)
                continue 
        
            if not os.path.exists(f"results/{folder}/results_{data_type}{suff}.csv"):
                df = pd.DataFrame(columns=["fold", "train_acc", "val_acc", "test_acc", "test_preds", "test_labels", "sparsity"])
                df.set_index('fold', inplace=True)
                df.to_csv(f"results/{folder}/results_{data_type}{suff}.csv", index=True)
                
                # To save sparsities for each subj
                df = pd.DataFrame(columns=["fold", "sparsity"])
                df.set_index('fold', inplace=True)
                df.to_csv(f"results/{folder}/spiketrains/sparsities_{data_type}{suff}.csv", index=True)
                
            args.append((config, input_data, target_labels, fold_num, train_index, test_index, device, folder))
            # Comment to run all subjects
            # if len(args) == 1:
            #     break
            
            # save config as json
            with open(f"results/{folder}/config_{data_type}.json", "w") as f:
                json.dump(config, f)
    
    print(f"Starting {len(args)} runs...")
    start = time.time()
    for arg in args:
        main(*arg)
    # Use code below to process in parallel.
    # with mp.Pool(16) as p:
    #     p.starmap(main, args)
    
    end = time.time()
    print(f"{len(args)} runs took {(end-start)/60:.2f} minutes.")
