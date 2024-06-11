# emopain-stl
Learnable Spike-Train Encoder Architecture for EmoPain

First, download the preprocessed data from WeTransfer (in the future, will include script to process yourself).
Place `data_Cs.pickle` and `data_Ps.pickle` in `/data` folder.

Set up the environment by: 
1. `python3 -m venv .env` (I used Python 3.10.6)
2. `source .env/bin/activate` (or similar path for Windows)
3. `pip3 install -r reqs.txt`

To run experiments, execute: `python3 train.py` with the desired configuration specified inside the `train.py` file.
