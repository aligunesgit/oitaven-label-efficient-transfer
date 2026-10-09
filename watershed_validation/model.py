import torch
from torch import nn


class Network(nn.Module):
    def __init__(self,kind):
        super().__init__();assert kind in ['spectral_mlp','patch_cnn','joint_cnn'];self.kind=kind
        self.spectral=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU()) if kind!='patch_cnn' else None
        self.spatial=nn.Sequential(nn.Conv2d(5,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.AdaptiveAvgPool2d(1)) if kind!='spectral_mlp' else None
        self.embedding=nn.Sequential(nn.Linear(64 if kind=='joint_cnn' else 32,64),nn.LayerNorm(64),nn.ReLU())
        self.head=nn.Linear(64,7)

    def forward(self,x):
        branches=[]
        if self.spectral is not None:branches.append(self.spectral(x[:,:,x.shape[2]//2,x.shape[3]//2]))
        if self.spatial is not None:branches.append(self.spatial(x).flatten(1))
        return self.head(self.embedding(torch.cat(branches,dim=1)))
