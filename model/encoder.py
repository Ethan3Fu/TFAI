import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

'''Time & Frequence encoder'''
class TFencoder(nn.Module):
    def __init__(self, configs, T=True):
        super().__init__()
        self.dim_length = configs.length
        if T:
            self.seq = configs.t_seq
        else:
            self.seq = configs.f_seq   
        self.linear_embedding = nn.Linear(configs.length, configs.length)
        self.cls_token = nn.Parameter(torch.randn(1,1,configs.length))
        self.position_embedding = nn.Parameter(torch.randn(1, self.seq+1, configs.length))
        encoderlayer = TransformerEncoderLayer(configs.length, dim_feedforward=configs.ndim, nhead=configs.nhead, batch_first=True)
        self.encoder = TransformerEncoder(encoderlayer, configs.TF_nlayer)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.seq, self.dim_length)
        x = self.linear_embedding(x)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        x = self.encoder(x)
        return x
    
'''Cross encoder'''
class Cross_encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layer = configs.T_decoder_layer
        self.T_decoder = nn.ModuleList([nn.TransformerDecoderLayer(configs.length, dim_feedforward=configs.ndim, 
                                                                   nhead=configs.nhead, batch_first=True) for _ in range(configs.T_decoder_layer)])
        self.F_encoder = nn.ModuleList([nn.TransformerEncoderLayer(configs.length, dim_feedforward=configs.ndim, 
                                                                    nhead=configs.nhead, batch_first=True) for _ in range(configs.F_encoder_layer)])
    def forward(self, t, f):
        for i in range(self.layer):
            f = self.F_encoder[i](f)
            t = self.T_decoder[i](t, f)
        return t
    
'''Linearprojection'''
class Linearprojection(nn.Module):
    def __init__(self, configs, T=True):
        super().__init__()
        self.dim_length = configs.length
        if T:
            self.seq = configs.t_seq
        else:
            self.seq = configs.f_seq   
        self.linear_embedding = nn.Linear(configs.length, configs.length)
        self.cls_token = nn.Parameter(torch.randn(1,1,configs.length))
        self.position_embedding = nn.Parameter(torch.randn(1, self.seq+1, configs.length))

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.seq, self.dim_length)
        x = self.linear_embedding(x)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x