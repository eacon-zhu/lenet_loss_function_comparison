import torch
import torch.nn as nn
import torch.nn.functional as F

class Arcface(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        A Softmax Loss
        '''
        super(Arcface, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features#输入通道数
        self.out_features = out_features#输出通道数
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        #==================权重标准化=======================#
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)
        # ==================输出标准化=======================#
        x = F.normalize(x, dim=1)
        # ==================经过激活函数=======================#
        wf = self.fc(x)
        # ==================分子=======================#
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels] - self.m))
        # ==================其他输出=======================#
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        # ==================分母=======================#
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        # ==================取log=======================#
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
