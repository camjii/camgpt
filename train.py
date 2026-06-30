import torch
import torch.nn as nn
import torch.nn.functional as F
class Train(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

        
#WIP


