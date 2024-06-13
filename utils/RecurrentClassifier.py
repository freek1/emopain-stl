import torch
import snntorch as snn

class RecurrentClassifier(torch.nn.Module):
    def __init__(self, encoder_output_size: int, lif_beta: float, l1_sz: int, n_classes: int, num_steps: int):
        super().__init__()
        
        print("Using Recurrent Classifier: Recurrent LIF Neurons.")
        
        self.num_steps = num_steps
        
        self.fc1 = torch.nn.Linear(encoder_output_size, l1_sz)
        self.rlif1 = snn.RLeaky(beta=lif_beta, linear_features=l1_sz) #, learn_beta=True, learn_recurrent=True)
        self.fc2 = torch.nn.Linear(l1_sz, n_classes)
        self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1) #, learn_beta=True, learn_recurrent=True)
        
    def forward(self, x):
        # At timestep 0, initalize spk1 and mem1
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        
        spk_rec = []
        mem_rec = []
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)
        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
