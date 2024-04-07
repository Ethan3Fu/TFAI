import torch
from torch import nn
from model.encoder import TFencoder, Cross_encoder, Linearprojection

class train_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq = configs.t_seq

        self.T_encoder = TFencoder(configs)
        self.T_mom_encoder = TFencoder(configs)

        self.F_encoder = TFencoder(configs, False)
        self.F_mom_encoder = TFencoder(configs, False)

        self.T_projection = build_mlp(configs.num_projection, configs.length, configs.ndim)
        self.T_mom_projection = build_mlp(configs.num_projection, configs.length, configs.ndim)

        self.F_projection = build_mlp(configs.num_projection, configs.length, configs.ndim)
        self.F_mom_projection = build_mlp(configs.num_projection, configs.length, configs.ndim)

        self.model_paris = [[self.T_encoder, self.T_mom_encoder],
                            [self.T_projection, self.T_mom_projection],
                            [self.F_encoder, self.F_mom_encoder],
                            [self.F_projection, self.F_mom_projection]]
        
        for model in self.model_paris:
            for param_b, param_m in zip(model[0].parameters(), model[1].parameters()):
                param_m.data.copy_(param_b.data)
                param_m.requires_grad = False

        self.T_prediction = build_mlp(configs.num_prediction, configs.ndim, configs.logit_dim)
        self.F_prediction = build_mlp(configs.num_prediction, configs.ndim, configs.logit_dim)

        self.T = configs.T
        self.m = configs.m
        
        self.cross_encoder = Cross_encoder(configs)

        self.projection = build_mlp(configs.num_projection, configs.length, configs.ndim)
        self.prediction = build_mlp(configs.num_prediction, configs.ndim, 2)
        
    def _update_momentum_encider(self, m):
        for model in self.model_paris:
            for param_b, param_m in zip(model[0].parameters(), model[1].parameters()):
                param_m.data = param_m.data * m + param_b.data * (1.0 - m)

    def forward(self, x_t, x_f):
        h_t = self.T_encoder(x_t)
        h_f = self.F_encoder(x_f)

        q_t = self.T_prediction(self.T_projection(h_t[:,0,:]))
        q_f = self.F_prediction(self.F_projection(h_f[:,0,:]))

        with torch.no_grad():
            self._update_momentum_encider(self.m)
            k_t = self.T_mom_projection(self.T_mom_encoder(x_t)[:,0,:])
            k_f = self.F_mom_projection(self.F_mom_encoder(x_f)[:,0,:])

        q_t = nn.functional.normalize(q_t, dim=1)
        q_f = nn.functional.normalize(q_f, dim=1)
        k_t = nn.functional.normalize(k_t, dim=1)
        k_f = nn.functional.normalize(k_f, dim=1)
 
        logits_t = q_t @ k_f.t() / self.T
        logits_f = q_f @ k_t.t() / self.T

        time_emb = h_t
        freq_emb = h_f

        with torch.no_grad():
            B = x_t.shape[0]
            t2f = logits_t.softmax(dim=1)
            f2t = logits_f.softmax(dim=1)

            t2f.fill_diagonal_(0)
            f2t.fill_diagonal_(0)
        
        time_neg, freq_neg = [], []
        for i in range(B):
            time_neg_idx = torch.multinomial(f2t[i], 1).item()
            time_neg.append(time_emb[time_neg_idx])
            freq_neg_idx = torch.multinomial(t2f[i], 1).item()
            freq_neg.append(freq_emb[freq_neg_idx])
        time_emb_neg = torch.stack(time_neg, dim=0)
        freq_emb_neg = torch.stack(freq_neg, dim=0)

        time_emb_all = torch.cat([time_emb, time_emb, time_emb_neg], dim=0)
        freq_emb_all = torch.cat([freq_emb, freq_emb_neg, freq_emb], dim=0)

        z = self.cross_encoder(time_emb_all, freq_emb_all)

        z = self.prediction(self.projection(z[:,0,:]))
    
        return logits_t, logits_f, z
    
class finetune_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq = configs.t_seq

        self.T_encoder = TFencoder(configs)
        self.F_encoder = TFencoder(configs, False)
        
        self.cross_encoder = Cross_encoder(configs)
        
        self.projection = build_mlp(configs.num_projection, configs.length, configs.ndim)
        self.fc = build_mlp(configs.num_prediction, configs.ndim, configs.num_class)

    def forward(self, x_t, x_f):
        h_t = self.T_encoder(x_t)
        h_f = self.F_encoder(x_f)
        
        time_emb = h_t
        freq_emb = h_f
        
        z = self.cross_encoder(time_emb, freq_emb)

        self.fea = self.projection(z[:,0,:])
        z = self.fc(self.fea)

        return z
    
    def get_fea(self):
        return self.fea

def build_mlp(num_layer, input, hidden):
    if num_layer == 0:
        return nn.Identity()
    else:
        mlp = []
        for i in range(num_layer):
            dim1 = input if i==0 else hidden
            dim2 = hidden
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if i < num_layer-1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            else:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)
