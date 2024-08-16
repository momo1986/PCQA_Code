import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class net_one(nn.Module):
    def __init__(self, feature_size = 512, dropout=0.1):
        expansion = 4
        init_tau = np.log(5.0)
        init_b = 0
        super().__init__()
        self.fc_text = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, feature_size * expansion, bias=False),
            nn.BatchNorm1d(feature_size * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size * expansion, feature_size, bias=False)
        )
        self.fc_image = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, feature_size * expansion, bias=False),
            nn.BatchNorm1d(feature_size * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size * expansion, feature_size, bias=False)
        )

        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, text, image):
        x_t = self.fc_text(text).unsqueeze(1)
        x_i = self.fc_image(image).unsqueeze(1)
        # x_t_norm = x_t/x_t.norm(dim=-1, keepdim=True)
        # x_i_norm = x_i/x_i.norm(dim=-1, keepdim=True)
        x_t_norm = F.normalize(x_t, p=2, dim=-1)
        x_i_norm = F.normalize(x_i, p=2, dim=-1)
        # out = torch.matmul(x_i_norm, torch.transpose(x_t_norm, 2, 1)) * self.t_prime.exp() + self.b
        out = x_i_norm @ torch.transpose(x_t_norm, 2, 1) * self.t_prime.exp() + self.b
        out = out.squeeze() * 100
        return out

class net(nn.Module):
    def __init__(self):
        fuse_feature = 7 # 3 models output + image type
        super().__init__()
        self.model0 = net_one(feature_size = 1024)
        self.model1 = net_one(feature_size = 1024)
        self.model2 = net_one(feature_size = 1152)
        self.fuse = nn.Linear(fuse_feature, 1)

    def forward(self, text1, image1, text2, image2, text3, image3, image_features):
        b = text1.shape[0]
        x_0 = self.model0(text1, image1).reshape(-1, 1)
        x_1 = self.model1(text2, image2).reshape(-1, 1)
        x_2 = self.model2(text3, image3).reshape(-1, 1)
        x = torch.cat([x_0, x_1, x_2, image_features], dim=-1)
        out = self.fuse(x).squeeze()
        return out
