# import torch
# import snntorch as snn

# class RecurrentClassifier(torch.nn.Module):
#     def __init__(self, encoder_output_size: int, lif_beta: float, l1_sz: int, l2_sz:int, n_classes: int, num_steps: int):
#         super().__init__()
        
#         print("Using Recurrent Classifier: Recurrent LIF Neurons.")
        
#         self.num_steps = num_steps
        
#         self.fc1 = torch.nn.Linear(encoder_output_size, l1_sz)
#         self.rlif1 = snn.RLeaky(beta=lif_beta, linear_features=l1_sz) #, learn_beta=True, learn_recurrent=True)
#         # self.rlif1 = snn.RLeaky(beta=lif_beta, all_to_all=False) #, learn_beta=True, learn_recurrent=True)
        
#         self.fc2 = torch.nn.Linear(l1_sz, l2_sz)
#         self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=l2_sz)
        
#         self.fc3 = torch.nn.Linear(l2_sz, n_classes)
#         self.rlif3 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1) #, learn_beta=True, learn_recurrent=True)
#         # self.rlif2 = snn.RLeaky(beta=lif_beta, all_to_all=False) #, learn_beta=True, learn_recurrent=True)
        
#     def forward(self, x):
#         # At timestep 0, initalize spk1 and mem1
#         spk1, mem1 = self.rlif1.init_rleaky()
#         spk2, mem2 = self.rlif2.init_rleaky()
#         spk3, mem3 = self.rlif3.init_rleaky()
        
#         spk_rec = []
#         mem_rec = []
#         for _ in range(self.num_steps):
#             cur1 = self.fc1(x)
#             spk1, mem1 = self.rlif1(cur1, spk1, mem1)
#             cur2 = self.fc2(spk1)
#             spk2, mem2 = self.rlif2(cur2, spk2, mem2)
#             cur3 = self.fc3(spk2)
#             spk3, mem3= self.rlif3(cur3, spk3, mem3)
#             spk_rec.append(spk2)
#             mem_rec.append(mem2)
        
#         return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

import torch
import snntorch as snn

class RecurrentClassifier(torch.nn.Module):
    def __init__(self, encoder_output_size: int, lif_beta: float, l1_sz: int, l2_sz: int, n_classes: int, num_steps: int):
        super().__init__()

        print("Using Recurrent Classifier: Recurrent LIF Neurons.")

        self.num_steps = num_steps
        
        self.fc1 = torch.nn.Linear(encoder_output_size, l1_sz)
        self.rlif1 = snn.RLeaky(beta=lif_beta, linear_features=l1_sz, V=1)
        
        # Code for adapting to 2 hidden layers (not used in the paper):
        # self.l2_sz = l2_sz
        # if l2_sz > 0:
        #     self.fc2 = torch.nn.Linear(l1_sz, l2_sz)
        #     self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=l2_sz)
        #     self.fc3 = torch.nn.Linear(l2_sz, n_classes)
        #     self.rlif3 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1)
        # else:
        
        self.fc2 = torch.nn.Linear(l1_sz, n_classes)
        self.rlif2 = snn.RLeaky(beta=lif_beta, linear_features=n_classes, V=1)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk_rec, mem_rec = [], []

        # Code for adapting to 2 hidden layers (not used in the paper):
        # if self.l2_sz > 0:
        #     spk2, mem2 = self.rlif2.init_rleaky()
        #     spk3, mem3 = self.rlif3.init_rleaky()
        #     for _ in range(self.num_steps):
        #         cur1 = self.fc1(x)
        #         spk1, mem1 = self.rlif1(cur1, spk1, mem1)
        #         cur2 = self.fc2(spk1)
        #         spk2, mem2 = self.rlif2(cur2, spk2, mem2)
        #         cur3 = self.fc3(spk2)
        #         spk3, mem3 = self.rlif3(cur3, spk3, mem3)
        #         spk_rec.append(spk3)
        #         mem_rec.append(mem3)
        # else:
        
        spk2, mem2 = self.rlif2.init_rleaky()
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
