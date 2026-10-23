"""Paper hyperparameters with methods recovered from emopain-snn bd50395d.

An auditable restoration candidate, not a claim of regenerated paper scores.
Original loss arithmetic and final-window decisions are deliberate. Independent
validation-selected checkpoints fix the historical state aliasing.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from utils.STAL import SpikeThresholdLearning
from utils.EncoderLoss import EncoderLoss
from utils.RecurrentClassifier import FullRecordingRecurrentClassifier


def paper_config(modality):
    return dict(profile='paper', task='pain', data_type=modality,
                batch_sz={'emg':32,'energy':8,'angle':16}[modality],
                window_size=3000, stride=750, n_spikes_per_timestep=5,
                num_steps=5, encoder_epochs=30, classifier_epochs=25,
                theta=.99, l1_sz=3000, l2_sz=3000, l1_cls=500, l2_cls=0,
                drop_p=.5, encoder_lr=.005, classifier_lr=.00075,
                lif_beta=.99, learn_beta=True, classifier_layout='snn_windows',
                prediction_rule='last_complete_window', normalization='snn_window',
                encoder_loss='snn', sampling='snn_2batch', checkpoint_selection='best',
                encoding_method='STL', avg_window_sz=100, suff='_STL-S',
                SVM=False, SRNN=True, generate_spikes=True)


def normalize_window(inputs):
    """Exact SNN normalize_data2 arithmetic; intentionally divide by max."""
    nan = torch.isnan(inputs)
    minimum = torch.where(nan, inputs.new_tensor(float('inf')), inputs).min(1,keepdim=True)[0]
    maximum = torch.where(nan, inputs.new_tensor(float('-inf')), inputs).max(1,keepdim=True)[0]
    maximum = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    return torch.nan_to_num((inputs - minimum) / maximum, nan=0.)


def training_loader(x, y, batch_size):
    counts = torch.bincount(y, minlength=2)
    if (counts == 0).any():
        raise ValueError('The training split must contain both classes for balanced sampling')
    weights = torch.zeros(len(y))
    for label in (0,1):
        weights[y == label] = 1. / counts[label].item()
    count = 2 * batch_size
    return DataLoader(TensorDataset(x,y), batch_size=count,
                      sampler=WeightedRandomSampler(weights,count,replacement=True))


def window_starts(length, config):
    size, stride = config['window_size'], config['stride']
    if size < 1 or stride < 1 or length < size:
        raise ValueError('A positive complete window and stride are required')
    return range(0, length-size+1, stride)


def snapshot(model):
    return {name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}


def predict_recording(encoder, classifier, x, config, device):
    """Original final-complete-window decision; density counts all windows."""
    spikes, elements = 0, 0
    with torch.no_grad():
        for start in window_starts(x.shape[1],config):
            inputs=normalize_window(x[:,start:start+config['window_size']].to(device))
            encoded,_,_=encoder(inputs)
            binary=(encoded > config['theta']).float()
            spk,_=classifier(binary)
            pred=spk.sum(0).argmax(1).cpu()
            spikes+=binary.sum().item();elements+=binary.numel()
    return pred,spikes/elements


def train_fold(config, x, y, train_index, test_index, fold, device, output):
    """Train original-style overlapping windows with explicit paper settings."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c=config
    expected=dict(profile='paper', task='pain', classifier_layout='snn_windows',
                  prediction_rule='last_complete_window', normalization='snn_window',
                  encoder_loss='snn', sampling='snn_2batch', checkpoint_selection='best',
                  generate_spikes=True, encoding_method='STL', SVM=False, SRNN=True)
    for key,value in expected.items():
        if c.get(key)!=value:
            raise ValueError(f'Unsupported paper profile setting: {key}={c.get(key)!r}')
    if c['encoder_epochs'] < 1 or c['classifier_epochs'] < 1:
        raise ValueError('Training epochs must be positive')
    train_idx,val_idx=train_test_split(np.asarray(train_index),test_size=.2,random_state=42)
    train=training_loader(x[train_idx],y[train_idx],c['batch_sz'])
    val=DataLoader(TensorDataset(x[val_idx],y[val_idx]),batch_size=c['batch_sz'])
    encoder=SpikeThresholdLearning(c['window_size'],c['n_spikes_per_timestep'],x.shape[2],
                                  c['l1_sz'],c['l2_sz'],c['drop_p']).to(device)
    criterion=EncoderLoss(mode=c['encoder_loss'])
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=c['encoder_lr'])
    history={'encoder':[],'classifier':[]}
    best=float('inf');best_encoder_epoch=None
    for epoch in tqdm(range(c['encoder_epochs']),desc=f'{c["data_type"]}:{fold} encoder'):
        encoder.train();train_losses=[]
        for inputs,_ in train:
            for start in window_starts(inputs.shape[1],c):
                window=normalize_window(inputs[:,start:start+c['window_size']].to(device))
                optimizer.zero_grad()
                encoded,z1,z2=encoder(window)
                loss=criterion(encoded,window.reshape(window.shape[0],-1),z1,z2)
                if not torch.isfinite(loss):raise ValueError('Non-finite encoder loss')
                loss.backward();optimizer.step();train_losses.append(loss.item())
        encoder.eval();val_losses=[]
        with torch.no_grad():
            for inputs,_ in val:
                for start in window_starts(inputs.shape[1],c):
                    window=normalize_window(inputs[:,start:start+c['window_size']].to(device))
                    encoded,z1,z2=encoder(window)
                    val_losses.append(criterion(encoded,window.reshape(window.shape[0],-1),z1,z2).item())
        val_loss=float(np.mean(val_losses))
        if not np.isfinite(val_loss):raise ValueError('Non-finite encoder validation loss')
        history['encoder'].append({'epoch':epoch+1,'train_loss':float(np.mean(train_losses)),'val_loss':val_loss})
        if val_loss < best:
            best=val_loss;best_encoder_epoch=epoch+1
            torch.save(snapshot(encoder),output/'encoder.pt')
    encoder.load_state_dict(torch.load(output/'encoder.pt',map_location=device,weights_only=True))
    encoder.eval();encoder.requires_grad_(False)
    del optimizer
    # SNN RecurrentClassifier3 receives one complete encoded window at each
    # recurrent step; it does not truncate a concatenated recording.
    classifier=FullRecordingRecurrentClassifier(encoder.output_size,c['lif_beta'],c['l1_cls'],
                                                c['num_steps'],learn_beta=c['learn_beta']).to(device)
    optimizer=torch.optim.AdamW(classifier.parameters(),lr=c['classifier_lr'])
    cross_entropy=torch.nn.CrossEntropyLoss()
    best=float('inf');best_classifier_epoch=None
    for epoch in tqdm(range(c['classifier_epochs']),desc=f'{c["data_type"]}:{fold} classifier'):
        epoch_losses=[]
        for training,loader in [(True,train),(False,val)]:
            classifier.train(training);losses=[]
            for inputs,labels in loader:
                labels=labels.to(device)
                for start in window_starts(inputs.shape[1],c):
                    with torch.no_grad():
                        window=normalize_window(inputs[:,start:start+c['window_size']].to(device))
                        encoded,_,_=encoder(window)
                        binary=(encoded > c['theta']).float()
                    with torch.set_grad_enabled(training):
                        if training:optimizer.zero_grad()
                        _,mem=classifier(binary)
                        loss=sum(cross_entropy(mem[step],labels) for step in range(c['num_steps']))
                        if not torch.isfinite(loss):raise ValueError('Non-finite classifier loss')
                        if training:loss.backward();optimizer.step()
                    losses.append(loss.item())
            epoch_losses.append(float(np.mean(losses)))
        history['classifier'].append({'epoch':epoch+1,'train_loss':epoch_losses[0],'val_loss':epoch_losses[1]})
        if epoch_losses[1] < best:
            best=epoch_losses[1];best_classifier_epoch=epoch+1
            torch.save(snapshot(classifier),output/'classifier.pt')
    classifier.load_state_dict(torch.load(output/'classifier.pt',map_location=device,weights_only=True))
    classifier.eval()
    pred,density=predict_recording(encoder,classifier,x[test_index],c,device)
    result=dict(fold=int(fold),train_acc=-1,val_acc=-1,
                test_acc=float((pred==y[test_index]).float().mean().item()*100),
                test_preds=pred.tolist(),test_labels=y[test_index].tolist(),sparsity=density)
    details=dict(config=c,train_indices=train_idx.tolist(),val_indices=val_idx.tolist(),
                 test_indices=np.asarray(test_index).tolist(),selected_encoder_epoch=best_encoder_epoch,
                 selected_classifier_epoch=best_classifier_epoch,history=history,result=result,
                 sampling_draws_per_epoch=2*c['batch_sz'],classifier_input_size=encoder.output_size,
                 windows_per_recording=len(window_starts(x.shape[1],c)))
    (output/'details.json').write_text(json.dumps(details,indent=2,allow_nan=False))
    return result
