import torch
import torch.nn as nn

class Multiply(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        x = torch.mul(x, self.alpha)
        return x

class CBLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBLR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.block(x)

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels, momentum=0.9),
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
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.Dropout(rate),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class UCTM(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(UCTM, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
            Multiply(alpha)
        )

    def forward(self, x):
        return self.block(x)

class Minibatch_stddev(nn.Module):
    def __init__(self, batch_size, group_size=4):
        super(Minibatch_stddev, self).__init__()
        # self.block = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        #     nn.Tanh()
        # )
        self.group_size = min(group_size, batch_size)

    def forward(self, input_tensor):
        shape = input_tensor.size()
        y = input_tensor.view(self.group_size, -1, shape[1], shape[2], shape[3])
        y -= torch.mean(y, dim=0, keepdim=True)
        pass
    

if __name__ == '__main__':
    x = torch.randn(4, 1, 3, 256, 256)
    x -= torch.mean(x, dim=0, keepdim=True)
    x = torch.sqrt(torch.mean(torch.square(x), dim=0))
    x = torch.mean(x, dim=[1,2,3])
    x = torch.tile(x, (4, 1, 256, 256))
    
    
    print(x.shape)
    print(x)