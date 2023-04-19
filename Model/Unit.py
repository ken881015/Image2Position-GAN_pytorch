import torch
import torch.nn as nn

class CBLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBLR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.block(x)

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class CLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.block(x)

class UCBDR(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(UCBDR, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(rate),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class UCT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UCT, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)