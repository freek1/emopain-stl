import numpy as np
from sklearn.svm import SVC
import torch
import torch.utils
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import pandas as pd
import snntorch.functional as SF

from utils.STL import SpikeThresholdLearning as STL
from utils.EncoderLoss import EncoderLoss
from utils.RecurrentClassifier import RecurrentClassifier

def train_STL_encoder(encoder: STL, device: torch.device,
                      train_loader: DataLoader, val_loader: DataLoader,
                      encoder_optimizer: AdamW, encoder_loss_fn: EncoderLoss,
                      encoder_epochs: int, window_size: int,
                      stride: int, folder: str, data_type: str,
                      fold_num:int, suff: str, verbose: bool = True):
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
                X_window = X_window.nan_to_num_(0)
                
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
                    X_window = X_window.nan_to_num_(0)
                    
                    X_flat = X_window.view(X_window.shape[0], -1)
                    y = y.to(device)
                    W, Z1, Z2 = encoder(X_window)
                    loss = encoder_loss_fn(W, X_flat, Z1, Z2)
                    epoch_val_loss += loss.item()
        enc_loss_val.append(epoch_val_loss)
        
        if epoch_val_loss < lowest_enc_val_loss:
            lowest_enc_val_loss = epoch_val_loss
            best_encoder = encoder.state_dict()
            torch.save(best_encoder, f"results/{folder}/best_encoder_{data_type}{suff}.pth")
    
    if verbose:
        save_enc_loss_plot(enc_loss, enc_loss_val, folder, suff, fold_num)
    
    encoder.load_state_dict(torch.load(f"results/{folder}/best_encoder_{data_type}{suff}.pth"))
    encoder.eval()
    
    return encoder

def train_SRNN_classifier(batch_sz, data_type, num_steps, encoder, l1_cls, window_size, stride, device, folder, suff, fold_num, classifier_epochs):    
    # loss_fn = SF.ce_rate_loss()
    
    train_spiketrains = np.load(f"results/{folder}/spiketrains/train_{data_type}_{fold_num}{suff}.npy")
    train_labels = np.load(f"results/{folder}/spiketrains/labels_train_{data_type}_{fold_num}{suff}.npy")
    val_spiketrains = np.load(f"results/{folder}/spiketrains/val_{data_type}_{fold_num}{suff}.npy")
    val_labels = np.load(f"results/{folder}/spiketrains/labels_val_{data_type}_{fold_num}{suff}.npy")
    test_spiketrains = np.load(f"results/{folder}/spiketrains/test_{data_type}_{fold_num}{suff}.npy")
    test_labels = np.load(f"results/{folder}/spiketrains/labels_test_{data_type}_{fold_num}{suff}.npy")
    
    len_spiketrain = train_spiketrains.shape[1]
    classifier = RecurrentClassifier(encoder.output_size, lif_beta=0.5, num_steps=num_steps, l1_sz=l1_cls, n_classes=2)
    print(f"Classifier params: \t{sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")
    
    classifier.to(device)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_spiketrains = torch.Tensor(train_spiketrains)
    train_labels = torch.Tensor(train_labels)
    val_spiketrains = torch.Tensor(val_spiketrains)
    val_labels = torch.Tensor(val_labels)
    test_spiketrains = torch.Tensor(test_spiketrains)
    test_labels = torch.Tensor(test_labels)
    
    cls_loss = []
    cls_loss_val = []
    cls_acc = []
    cls_acc_val = []
    
    lowest_cls_val_loss = np.inf
    best_classifier = None
    
    for epoch in tqdm(range(classifier_epochs), "Classifier"):
        classifier.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx in range(0, train_spiketrains.shape[0], batch_sz):
            y = train_labels[batch_idx:batch_idx+batch_sz].long().to(device)
            for window in range(0, train_spiketrains.shape[1] - encoder.output_size + 1, stride):
                W_window = train_spiketrains[batch_idx:batch_idx+batch_sz, window:window + encoder.output_size].to(device)

                classifier_optimizer.zero_grad()
                spk, mem = classifier(W_window)
                _, preds = spk.sum(dim=0).max(1)
                
                loss = torch.zeros((1), dtype=torch.float, device=device)
                for step in range(num_steps):
                    loss += loss_fn(mem[step], y)

                loss.backward()
                classifier_optimizer.step()
                
                epoch_loss += loss.item()
                epoch_correct += preds.eq(y).sum().item()
                epoch_total += y.size(0)
        cls_loss.append(epoch_loss)
        cls_acc.append(epoch_correct / epoch_total)
        
        with torch.no_grad():
            classifier.eval()
            epoch_val_loss = 0
            epoch_val_correct = 0
            epoch_val_total = 0
            for batch_idx in range(0, val_spiketrains.shape[0], batch_sz):
                y = val_labels[batch_idx:batch_idx+batch_sz].long().to(device)
                for window in range(0, val_spiketrains.shape[1] - encoder.output_size + 1, stride):
                    W_window = val_spiketrains[batch_idx:batch_idx+batch_sz, window:window + encoder.output_size].to(device)
                    
                    classifier_optimizer.zero_grad()
                    spk, mem = classifier(W_window)
                    _, preds = spk.sum(dim=0).max(1)
                    
                    loss = torch.zeros((1), dtype=torch.float, device=device)
                    for step in range(num_steps):
                        loss += loss_fn(mem[step], y)
                        
                    epoch_val_loss += loss.item()
                    epoch_val_correct += preds.eq(y).sum().item()
                    epoch_val_total += y.size(0)
            cls_loss_val.append(epoch_val_loss)
            cls_acc_val.append(epoch_val_correct / epoch_val_total)

        if epoch_val_loss < lowest_cls_val_loss:
            lowest_cls_val_loss = epoch_val_loss
            best_classifier = classifier.state_dict()
            torch.save(best_classifier, f"results/{folder}/best_classifier_{data_type}{suff}.pth")
    
    save_cls_loss_plot(cls_loss, cls_loss_val, folder, suff, data_type)
    
    classifier.load_state_dict(torch.load(f"results/{folder}/best_classifier_{data_type}{suff}.pth"))
    classifier.eval()
    
    # Remove cuda variables
    train_spiketrains = train_spiketrains.cpu()
    train_labels = train_labels.cpu()
    val_spiketrains = val_spiketrains.cpu()
    val_labels = val_labels.cpu()
    test_spiketrains = test_spiketrains.cpu()
    test_labels = test_labels.cpu()
    classifier.cpu()
    torch.cuda.empty_cache()
    
    return classifier

def train_SRNN_classifier_nowindow(batch_sz, data_type, num_steps, encoder, l1_cls, l2_cls, window_size, stride, device, folder, suff, fold_num, classifier_epochs):    
    # loss_fn = SF.ce_rate_loss()
    
    train_spiketrains = np.load(f"results/{folder}/spiketrains/train_{data_type}_{fold_num}{suff}.npy")
    train_labels = np.load(f"results/{folder}/spiketrains/labels_train_{data_type}_{fold_num}{suff}.npy")
    val_spiketrains = np.load(f"results/{folder}/spiketrains/val_{data_type}_{fold_num}{suff}.npy")
    val_labels = np.load(f"results/{folder}/spiketrains/labels_val_{data_type}_{fold_num}{suff}.npy")
    test_spiketrains = np.load(f"results/{folder}/spiketrains/test_{data_type}_{fold_num}{suff}.npy")
    test_labels = np.load(f"results/{folder}/spiketrains/labels_test_{data_type}_{fold_num}{suff}.npy")
    
    len_spiketrain = train_spiketrains.shape[1]
    classifier = RecurrentClassifier(len_spiketrain, lif_beta=0.5, num_steps=num_steps, l1_sz=l1_cls, l2_sz=l2_cls, n_classes=2)
    print(f"Classifier params: \t{sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")
    
    classifier.to(device)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_spiketrains = torch.Tensor(train_spiketrains)
    train_labels = torch.Tensor(train_labels)
    val_spiketrains = torch.Tensor(val_spiketrains)
    val_labels = torch.Tensor(val_labels)
    test_spiketrains = torch.Tensor(test_spiketrains)
    test_labels = torch.Tensor(test_labels)
    
    cls_loss = []
    cls_loss_val = []
    cls_acc = []
    cls_acc_val = []
    
    lowest_cls_val_loss = np.inf
    best_classifier = None
    
    for epoch in tqdm(range(classifier_epochs), "Classifier"):
        classifier.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx in range(0, train_spiketrains.shape[0], batch_sz):
            y = train_labels[batch_idx:batch_idx+batch_sz].long().to(device)
            W = train_spiketrains[batch_idx:batch_idx+batch_sz].to(device)

            classifier_optimizer.zero_grad()
            spk, mem = classifier(W)
            _, preds = spk.sum(dim=0).max(1)
            
            loss = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                loss += loss_fn(mem[step], y)

            loss.backward()
            classifier_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_correct += preds.eq(y).sum().item()
            epoch_total += y.size(0)
        cls_loss.append(epoch_loss)
        cls_acc.append(epoch_correct / epoch_total)
        
        with torch.no_grad():
            classifier.eval()
            epoch_val_loss = 0
            epoch_val_correct = 0
            epoch_val_total = 0
            for batch_idx in range(0, val_spiketrains.shape[0], batch_sz):
                y = val_labels[batch_idx:batch_idx+batch_sz].long().to(device)
                W = val_spiketrains[batch_idx:batch_idx+batch_sz].to(device)
                
                classifier_optimizer.zero_grad()
                spk, mem = classifier(W)
                _, preds = spk.sum(dim=0).max(1)
                
                loss = torch.zeros((1), dtype=torch.float, device=device)
                for step in range(num_steps):
                    loss += loss_fn(mem[step], y)
                    
                epoch_val_loss += loss.item()
                epoch_val_correct += preds.eq(y).sum().item()
                epoch_val_total += y.size(0)
            cls_loss_val.append(epoch_val_loss)
            cls_acc_val.append(epoch_val_correct / epoch_val_total)

        if epoch_val_loss < lowest_cls_val_loss:
            lowest_cls_val_loss = epoch_val_loss
            best_classifier = classifier.state_dict()
            torch.save(best_classifier, f"results/{folder}/best_classifier_{data_type}{suff}.pth")
    
    save_cls_loss_plot(cls_loss, cls_loss_val, folder, suff, data_type, fold_num)
    
    classifier.load_state_dict(torch.load(f"results/{folder}/best_classifier_{data_type}{suff}.pth"))
    classifier.eval()
    
    # Remove cuda variables
    train_spiketrains = train_spiketrains.cpu()
    train_labels = train_labels.cpu()
    val_spiketrains = val_spiketrains.cpu()
    val_labels = val_labels.cpu()
    test_spiketrains = test_spiketrains.cpu()
    test_labels = test_labels.cpu()
    classifier.cpu()
    torch.cuda.empty_cache()
    
    return classifier

def save_enc_loss_plot(enc_loss: list, enc_loss_val: list, folder: str, suff: str, fold_num: int):
    plt.figure(figsize=(8,6))
    plt.plot(enc_loss, label="train")
    plt.plot(enc_loss_val, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Encoder Loss")
    plt.savefig(f"imgs/{folder}/encoder_loss{suff}_{fold_num}.png")
    plt.close()
    
def save_cls_loss_plot(cls_loss: list, cls_loss_val: list, folder: str, suff: str, data_type: str, fold_num: int):
    plt.figure(figsize=(8,6))
    plt.plot(cls_loss, label="train")
    plt.plot(cls_loss_val, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Classifier Loss")
    plt.savefig(f"imgs/{folder}/classifier_loss_{data_type}{suff}_{fold_num}.png")
    plt.close()

def classify_svm(n_spikes_per_timestep, n_channels, folder, data_type, suff, fold_num, avg_window_sz):
    classifier_svm = SVC(kernel='linear', C=1.0, random_state=1957)
    print("Training SVM...")

    # NOTE: load from emopain_svm.
    train_spiketrains = np.load(f"results/emopain_svm/spiketrains/train_{data_type}_{fold_num}{suff}.npy")
    train_labels = np.load(f"results/emopain_svm/spiketrains/labels_train_{data_type}_{fold_num}{suff}.npy")
    val_spiketrains = np.load(f"results/emopain_svm/spiketrains/val_{data_type}_{fold_num}{suff}.npy")
    val_labels = np.load(f"results/emopain_svm/spiketrains/labels_val_{data_type}_{fold_num}{suff}.npy")
    test_spiketrains = np.load(f"results/emopain_svm/spiketrains/test_{data_type}_{fold_num}{suff}.npy")
    test_labels = np.load(f"results/emopain_svm/spiketrains/labels_test_{data_type}_{fold_num}{suff}.npy")
    
    # combine train and val
    spk_inp_train = np.vstack([train_spiketrains, val_spiketrains])
    train_val_labels = np.concatenate([train_labels, val_labels])
    
    spk_inp_test = test_spiketrains
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
    ts_acc = (test_preds == test_labels).mean()*100
    
    print(f"Test Accuracy: \t\t{ts_acc:.3f}%")
    
    # Save results
    results_file = pd.read_csv(f"results/{folder}/results_{data_type}{suff}.csv", index_col=0)
    results = pd.DataFrame([{
        'fold': fold_num, 
        'train_acc': -1, 
        'val_acc': -1, 
        'test_acc': ts_acc, 
        'test_preds': test_preds, 
        'test_labels': test_labels, 
        'sparsity': sparsity
    }])
    results_file = pd.concat([results_file, results])
    results_file.to_csv(f"results/{folder}/results_{data_type}{suff}.csv")
    
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
    plt.title(f"{data_type.capitalize()} - Fold {fold_num} (y={y0})")
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
    
    plt.savefig(f"imgs/{folder}/spiketrain_svm_{data_type}{suff}_{fold_num}.png")
    plt.close()

    return

def classify_srnn(classifier, encoder_output_size, folder, data_type, fold_num, suff, device, window_size, n_channels, n_spikes_per_timestep):
    print("Training SRNN...")
    classifier.to(device)
    classifier.eval()

    test_spiketrains = np.load(f"results/{folder}/spiketrains/test_{data_type}_{fold_num}{suff}.npy")
    test_labels = np.load(f"results/{folder}/spiketrains/labels_test_{data_type}_{fold_num}{suff}.npy")
    
    print(test_spiketrains.shape)
    print(test_labels.shape)
    
    spk_inp_test = test_spiketrains
    n_spikes = np.sum(spk_inp_test)
    total_spikes = np.prod(spk_inp_test.shape)
    sparsity = n_spikes / total_spikes
    print("Sparsity", sparsity)
    
    test_input = torch.Tensor(test_spiketrains)
    test_target = torch.Tensor(test_labels)
    cls_correct_test = 0
    cls_total_test = 0
    for window in range(0, test_spiketrains.shape[1] - encoder_output_size + 1, encoder_output_size):
        spk_inputs = test_input[:, window:window + encoder_output_size].to(device)
        y = test_target.to(device)
        
        spk, mem = classifier(spk_inputs)
        _, preds = spk.sum(dim=0).max(1)
        preds = preds.detach().cpu().numpy()

    cls_correct_test += (preds == test_labels).sum().item()
    cls_total_test += y.size(0)
    cls_acc_test = cls_correct_test / cls_total_test * 100
    print(f"Test Accuracy: \t\t{cls_acc_test:.3f}%")
    
    # Save results
    results_file = pd.read_csv(f"results/{folder}/results_{data_type}{suff}.csv", index_col=0)
    results = pd.DataFrame([{
        'fold': fold_num, 
        'train_acc': -1, 
        'val_acc': -1, 
        'test_acc': cls_acc_test, 
        'test_preds': preds, 
        'test_labels': test_labels, 
        'sparsity': sparsity
    }])
    results_file = pd.concat([results_file, results])
    results_file.to_csv(f"results/{folder}/results_{data_type}{suff}.csv")
    
    idx = 0
    
    W0 = test_spiketrains[idx]
    y0 = test_labels[idx]
    
    W0_channels = W0.reshape(-1, n_channels, n_spikes_per_timestep)
    print("Making spiketrain figure...")
    plt.figure(figsize=(8,6))
    cs = plt.cm.tab20.colors
    pixels, channels, spikes = np.where(W0_channels == 1)
    y_pos = 1 + 0.01 * spikes + n_spikes_per_timestep * 0.01 * channels
    colors = np.array(cs)[channels]
    plt.scatter(pixels, y_pos, color=colors, marker='o', s=2)
    plt.title(f"{data_type.capitalize()} - Fold {fold_num} (y={y0})")
    plt.savefig(f"imgs/{folder}/spiketrain_srnn_{data_type}{suff}_{fold_num}.png")
    plt.close()
    
    # Delete cuda variables
    classifier.cpu()
    test_target.cpu()
    test_input.cpu()
    torch.cuda.empty_cache()
    
    return

def classify_srnn_nowindow(classifier, encoder_output_size, folder, data_type, fold_num, suff, device, window_size, n_channels, n_spikes_per_timestep):
    print("Training SRNN...")
    classifier.to(device)
    classifier.eval()

    test_spiketrains = np.load(f"results/{folder}/spiketrains/test_{data_type}_{fold_num}{suff}.npy")
    test_labels = np.load(f"results/{folder}/spiketrains/labels_test_{data_type}_{fold_num}{suff}.npy")
    
    print(test_spiketrains.shape)
    print(test_labels.shape)
    
    spk_inp_test = test_spiketrains
    n_spikes = np.sum(spk_inp_test)
    total_spikes = np.prod(spk_inp_test.shape)
    sparsity = n_spikes / total_spikes
    print("Sparsity", sparsity)
    
    test_input = torch.Tensor(test_spiketrains)
    test_target = torch.Tensor(test_labels)
    cls_correct_test = 0
    cls_total_test = 0

    spk_inputs = test_input.to(device)
    y = test_target.to(device)
    
    spk, mem = classifier(spk_inputs)
    _, preds = spk.sum(dim=0).max(1)
    preds = preds.detach().cpu().numpy()

    cls_correct_test += (preds == test_labels).sum().item()
    cls_total_test += y.size(0)
    cls_acc_test = cls_correct_test / cls_total_test * 100
    print(f"Test Accuracy: \t\t{cls_acc_test:.3f}%")
    
    # Save results
    results_file = pd.read_csv(f"results/{folder}/results_{data_type}{suff}.csv", index_col=0)
    results = pd.DataFrame([{
        'fold': fold_num, 
        'train_acc': -1, 
        'val_acc': -1, 
        'test_acc': cls_acc_test, 
        'test_preds': preds, 
        'test_labels': test_labels, 
        'sparsity': sparsity
    }])
    results_file = pd.concat([results_file, results])
    results_file.to_csv(f"results/{folder}/results_{data_type}{suff}.csv")
    
    idx = 0
    
    W0 = test_spiketrains[idx]
    y0 = test_labels[idx]
    
    W0_channels = W0.reshape(-1, n_channels, n_spikes_per_timestep)
    print("Making spiketrain figure...")
    plt.figure(figsize=(8,6))
    cs = plt.cm.tab20.colors
    pixels, channels, spikes = np.where(W0_channels == 1)
    y_pos = 1 + 0.01 * spikes + n_spikes_per_timestep * 0.01 * channels
    colors = np.array(cs)[channels]
    plt.scatter(pixels, y_pos, color=colors, marker='o', s=2)
    plt.title(f"{data_type.capitalize()} - Fold {fold_num} (y={y0})")
    plt.savefig(f"imgs/{folder}/spiketrain_srnn_{data_type}{suff}_{fold_num}.png")
    plt.close()
    
    # Delete cuda variables
    classifier.cpu()
    test_target.cpu()
    test_input.cpu()
    torch.cuda.empty_cache()
    
    return