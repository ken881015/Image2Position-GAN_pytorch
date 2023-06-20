import torch
import torch.nn as nn
from Model.Unit import *

__all__ = ['Discriminator',]

class Discriminator(nn.Module):
    def __init__(self, in_channels, args):
        super(Discriminator, self).__init__()
        self.encoder = nn.ModuleList([
            # Layer 1
            CLR(in_channels, args.ndf),

            # Layer 2
            CBLR(args.ndf, args.ndf*2),

            # Layer 3
            CBLR(args.ndf*2, args.ndf*4),

            # Layer 4
            CBLR(args.ndf*4, args.ndf*8),

            # Layer 5
            CBLR(args.ndf*8, args.ndf*8),

            # Layer 6
            CBLR(args.ndf*8, args.ndf*8),

            # Layer 7
            CBLR(args.ndf*8, args.ndf*8),

            # Layer 8
            CBR(args.ndf*8, args.ndf*8),
        ])
        self.encoder.apply(self.weight_init)
        
        self.decoder = nn.ModuleList([
            # Layer -8
            UCBDR(args.ndf*8, args.ndf*8, 0.5),

            # Layer -7
            UCBDR(args.ndf*8*2, args.ndf*8, 0.5),

            # Layer -6
            UCBDR(args.ndf*8*2, args.ndf*8, 0.5),

            # Layer -5
            UCBDR(args.ndf*8*2, args.ndf*8, 0.),

            # Layer -4
            UCBDR(args.ndf*8*2, args.ndf*4, 0.),

            # Layer -3
            UCBDR(args.ndf*4*2, args.ndf*2, 0.),

            # Layer -2
            UCBDR(args.ndf*2*2, args.ndf, 0.),

            # Layer -1
            UCTM(args.ndf*2, 6, alpha=1.5)
        ])
        
        self.decoder.apply(self.weight_init)
        
    def weight_init(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0,std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
        if isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(1., 0.02)
            module.bias.data.zero_()
    
    def forward(self, x):
        
        # collect output of layer 1~8 
        encoder_layer_outputs = []
        
        for block in self.encoder:
            x = block(x)
            encoder_layer_outputs.append(x)
        
        encoder_layer_outputs.reverse() # layer 8 ~ 1
        
        for idx,block in enumerate(self.decoder, start=1):
            x = block(x)
            
            if idx != len(self.decoder):
                x = torch.cat((encoder_layer_outputs[idx], x), dim=1)
            
        
        return x
    
    def neg_grad(self, k_t=1.):
        for p in self.parameters():
            if p.grad is not None:
                p.grad *= -k_t