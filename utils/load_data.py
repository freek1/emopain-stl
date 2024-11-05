import pickle
import numpy as np
import torch

def load_data_emopain(data_type: str):
    """ 
    Returns input data and target labels, (n_samples, n_timesteps, n_channels).
    The data are normalized between [0, 1].
    """
    assert data_type in ["emg", "energy", "angle"], "data_type must be either 'emg', 'energy' or 'angle'."

    suffix = ""
    use_protective_label = True
    if use_protective_label:
        print("----- NOTE:")
        print("SET TO USING RATER INFORMATION FOR PROTECTIVE LABELS")
        suffix = {'_rater'}
    
    # Load data
    with open(f'data/data_Cs_rater.pickle', 'rb') as f:
        data_Cs = pickle.load(f)
    with open(f'data/data_Ps_rater.pickle', 'rb') as f:
        data_Ps = pickle.load(f)
    
    n_channels = 0 # placeholder
    if data_type == "emg":
        n_channels = 4
        channel_type = "sEMG_probe"
    elif data_type == "energy":
        n_channels = 12 # Try excluding the final two channels (seem like outliers)
        channel_type = "joint_energy"
    elif data_type == "angle":
        n_channels = 12 # Also for angle..
        channel_type = "joint_angle"

    input_data = []
    target_labels = []
    for data_C in data_Cs:
        stacked_data_C = np.stack([data_C[f"{channel_type}_{i_channel+1}"] for i_channel in range(n_channels)], axis=1)
        input_data.append(stacked_data_C)
        target_labels.append(data_C["protective_label"])
    
    for data_P in data_Ps:
        stacked_data_P = np.stack([data_P[f"{channel_type}_{i_channel+1}"] for i_channel in range(n_channels)], axis=1)
        input_data.append(stacked_data_P)
        target_labels.append(data_P["protective_label"])

    cut_cnt = 0
    for i, subject in enumerate(input_data):
        # Cut the longest subjects by removing first and last 6_000 samples
        if subject.shape[0] > 18_000:
            cut_cnt += 1
            cut_subj = subject[6_000:-6_000]
            input_data[i] = cut_subj

    # pad all to longest sequence
    max_len = max([data.shape[0] for data in input_data])
    padded_input_data = [np.pad(data, ((0, max_len - data.shape[0]), (0, 0)), mode='constant') for data in input_data] 

    input_data = np.array(padded_input_data)

    input_data = torch.tensor(input_data, dtype=torch.float)
    target_labels = torch.tensor(target_labels, dtype=torch.long)

    n_samples = input_data.shape[0]
    n_timesteps = input_data.shape[1]
    n_channels = input_data.shape[2]

    # normalize data to [0, 1]
    for chan in range(n_channels):
        for s in range(n_samples):
            x = input_data[s, :, chan]
            # Normalize the data between 0 and 1
            min_ = torch.min(x, dim=0, keepdim=True)[0]
            x = x - min_ # get all positive
            max_ = torch.max(x, dim=0, keepdim=True)[0]
            x = x / max_
            input_data[s, :, chan] = x
    
    return input_data, target_labels
    