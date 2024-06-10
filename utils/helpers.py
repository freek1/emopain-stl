import numpy as np
from sklearn.svm import SVC
import torch
import torch.utils
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

from utils.STL import SpikeThresholdLearning as STL
from utils.EncoderLoss import EncoderLoss
from utils.RecurrentClassifier import RecurrentClassifier

def train_STL_encoder(encoder: STL, device: torch.device,
                      train_loader: DataLoader, val_loader: DataLoader,
                      encoder_optimizer: AdamW, encoder_loss_fn: EncoderLoss,
                      encoder_epochs: int, window_size: int,
                      stride: int, folder: str,
                      verbose: bool = True):
    enc_loss = []
    enc_loss_val = []
    
    lowest_enc_val_loss = np.inf
    best_encoder = None
    
    encoder = encoder.to(device)
    for epoch in tqdm(range(encoder_epochs), "Encoder"):
        encoder.train()
        epoch_loss = 0
        for X, y in train_loader:
            for window in range(0, X.size(1) - window_size, stride):
                X_window = X[:, window:window + window_size]
                X_window = X_window.to(device)
                X_flat = X_window.view(X_window.shape[0], -1)
                y = y.to(device)
                encoder_optimizer.zero_grad()
                
                # TODO: Maybe add normalize() ?
                
                W, Z1, Z2 = encoder(X_window)
                loss = encoder_loss_fn(W, X_flat, Z1, Z2)
                loss.backward()
                encoder_optimizer.step()
                epoch_loss += loss.item()
        enc_loss.append(epoch_loss)
        
        encoder.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                for window in range(0, X.size(1) - window_size, stride):
                    X_window = X[:, window:window + window_size]
                    X_window = X_window.to(device)
                    X_flat = X_window.view(X_window.shape[0], -1)
                    y = y.to(device)
                    W, Z1, Z2 = encoder(X_window)
                    loss = encoder_loss_fn(W, X_flat, Z1, Z2)
                    epoch_val_loss += loss.item()
        enc_loss_val.append(epoch_val_loss)
        
        if epoch_val_loss < lowest_enc_val_loss:
            lowest_enc_val_loss = epoch_val_loss
            best_encoder = encoder.state_dict()
            torch.save(best_encoder, f"results/{folder}/best_encoder.pth")
    
    if verbose:
        save_loss_plot(enc_loss, enc_loss_val, folder)
    
    encoder.load_state_dict(torch.load(f"results/{folder}/best_encoder.pth"))
    encoder.eval()
    
    return encoder

def save_loss_plot(enc_loss: list, enc_loss_val: list, folder: str):
    plt.figure(figsize=(8,6))
    plt.plot(enc_loss, label="train")
    plt.plot(enc_loss_val, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Encoder Loss")
    plt.savefig(f"imgs/{folder}/encoder_loss.png")
    plt.close()

def classify_svm(train_labels, val_labels, test_labels, n_spikes_per_timestep, n_channels, folder, data_type, suff, fold_num, avg_window_sz):
    classifier_svm = SVC(kernel='linear', C=1.0, random_state=1957)
    print("Training SVM...")
    
    train_spiketrains = glob.glob(f"results/{folder}/spiketrains/train_*{suff}.npy")
    val_spiketrains = glob.glob(f"results/{folder}/spiketrains/val_*{suff}.npy")
    test_spiketrains = glob.glob(f"results/{folder}/spiketrains/test_*{suff}.npy")
    
    spk_inp_val = []
    spk_inp_train = []
    
    for train_spktr in train_spiketrains:
        spk_train = np.load(train_spktr)
        spk_inp_train.append(spk_train)
    for val_spktr in val_spiketrains:
        spk_val = np.load(val_spktr)
        spk_inp_val.append(spk_val)
        
    spk_inp_train = np.concatenate(spk_inp_train, axis=0)
    spk_inp_val = np.concatenate(spk_inp_val, axis=0)
    # combine train and val
    spk_inp_train = np.vstack([spk_inp_train, spk_inp_val])
    
    train_val_labels = torch.cat([train_labels, val_labels])
    
    for test_spktr in test_spiketrains:
        spk_test = np.load(test_spktr)
        spk_inp_test = spk_test

    n_spikes = np.sum(spk_inp_test)
    total_spikes = np.prod(spk_inp_test.shape)
    sparsity = n_spikes / total_spikes
    print("Sparsity", sparsity)

    # Average the spikes over 100 timesteps
    spk_inp_train_avg = spk_inp_train.reshape(-1, avg_window_sz, spk_inp_train.shape[1] // avg_window_sz).sum(axis=1)
    spk_inp_test_avg = spk_inp_test.reshape(-1, avg_window_sz, spk_inp_test.shape[1] // avg_window_sz).sum(axis=1)
    
    # compute sample weights: each sample is weighted by the class frequency
    pos_count = (train_val_labels == 1).sum().item()
    neg_count = (train_val_labels == 0).sum().item()
    weights = torch.zeros(train_val_labels.shape)
    weights[train_val_labels == 1] = 1.0 / pos_count
    weights[train_val_labels == 0] = 1.0 / neg_count
    
    classifier_svm.fit(spk_inp_train_avg, train_val_labels, sample_weight=weights)
    test_preds = classifier_svm.predict(spk_inp_test_avg)
    ts_acc = (test_preds == test_labels.numpy()).mean()*100
    
    print(f"Test Accuracy: \t\t{ts_acc:.3f}%")
    
    with open(f"results/{folder}/results_{data_type}{suff}.csv", "a") as f:
        # fold_num, train_acc, val_acc, test_acc, test_pred, test_label, sparsity
        f.write(f"{fold_num},{-1},{-1},{ts_acc:.3f},{test_preds.item()},{test_labels.item()},{sparsity}\n")
    
    idx = 0
    
    W0 = spk_inp_test[idx]
    y0 = test_labels[idx]
    spk_avg0 = spk_inp_test_avg[idx]
    
    W0_channels = W0.reshape(-1, n_channels, n_spikes_per_timestep)
    
    try:
        spk_avg_channels = spk_avg0.reshape(-1, n_channels, n_spikes_per_timestep)
    except:
        print("Failed to reshape spk_avg0", spk_avg0.shape)
        return

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    cs = plt.cm.tab20.colors
    pixels, channels, spikes = np.where(W0_channels == 1)
    y_pos = 1 + 0.01 * spikes + n_spikes_per_timestep * 0.01 * channels
    colors = np.array(cs)[channels]
    
    plt.scatter(pixels, y_pos, color=colors, marker='o', s=2)
    plt.title(f"{data_type.capitalize()} - Fold {fold_num+1} (y={y0})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    
    plt.subplot(2,1,2)
    for channel in range(n_channels):
        plt.plot(np.mean(spk_avg_channels[:, channel, :], axis=1), label=f"ch {channel+1}", color=cs[channel])
    plt.title(fr"Sum over timewindow of {avg_window_sz} steps of avg(spikes $\in \psi$) ")
    plt.ylabel("Sum(spikes)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"imgs/{folder}/spiketrain_svm_{data_type}{suff}_{fold_num+1}.png")
    plt.close()

    return

def classify_srnn():
    # TODO after the rest, implement this and test!
    classifier = RecurrentClassifier(encoder.output_size, lif_beta=0.5, num_steps=num_steps, l1_sz=l1_cls, n_classes=2, window_size=window_size, stride=stride)
    
