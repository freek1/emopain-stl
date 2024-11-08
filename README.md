# STAL: Spike Threshold Adaptive Learning
Learnable Spike-Train Encoder Architecture for the EmoPain dataset.

[![STAL Architecture](https://i.postimg.cc/wBLbV902/LAST-png.png)](https://postimg.cc/QBNbNZLK)

---
Paper is published as pre-print to be found at [ArXiv](https://arxiv.org/abs/2407.08362) and [ResearchGate](https://www.researchgate.net/publication/382150158_STAL_Spike_Threshold_Adaptive_Learning_Encoder_for_Classification_of_Pain-Related_Biosignal_Data)!

---

To use as is, first obtain permission from the maintainers of the EmoPain dataset. The code is written to be adaptable to any dataset, examples will follow in the future. When data are obtained, place in the data/ folder and follow instructions below. 

Set up the environment by (I used Python 3.10.6): 
1. `python3 -m venv .env`
2. `source .env/bin/activate` (or similar path for Windows)
3. `pip3 install -r reqs.txt`

To run experiments, execute: `python3 train.py` with the desired configuration specified inside the `train.py` file.



Example scripts will follow. If any documentation is unclear, feel free to contact me at: freek.hens@ru.nl
