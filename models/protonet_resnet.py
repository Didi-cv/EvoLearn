import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练 ResNet-18
        resnet = models.resnet18(pretrained=True)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # 特征维度 512
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return F.normalize(x, p=2, dim=1)

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Encoder()

    def forward(self, support_x, support_y, query_x):
        support_emb = self.encoder(support_x)
        query_emb = self.encoder(query_x)
        n_way = len(torch.unique(support_y))
        prototypes = []
        for c in range(n_way):
            mask = (support_y == c)
            proto = support_emb[mask].mean(0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        logits = -torch.cdist(query_emb, prototypes)
        return logits
