import torch
import torch.nn as nn
from Unit import *

class Generator(nn.Module):
    def __init__(self, in_channels, args):
        super(Generator, self).__init__()
        self.encoder = nn.ModuleList([
            # Layer 1
            CLR(in_channels, args.ngf),

            # Layer 2
            CBLR(args.ngf, args.ngf*2),

            # Layer 3
            CBLR(args.ngf*2, args.ngf*4),

            # Layer 4
            CBLR(args.ngf*4, args.ngf*8),

            # Layer 5
            CBLR(args.ngf*8, args.ngf*8),

            # Layer 6
            CBLR(args.ngf*8, args.ngf*8),

            # Layer 7
            CBLR(args.ngf*8, args.ngf*8),

            # Layer 8
            CBR(args.ngf*8, args.ngf*8),
        ])

        self.encoder.apply(self.weight_init)
        
        self.decoder = nn.ModuleList([
            # Layer -8
            UCBDR(args.ngf*8, args.ngf*8, 0.5),

            # Layer -7
            UCBDR(args.ngf*8*2, args.ngf*8, 0.5),

            # Layer -6
            UCBDR(args.ngf*8*2, args.ngf*8, 0.5),

            # Layer -5
            UCBDR(args.ngf*8*2, args.ngf*8, 0.),

            # Layer -4
            UCBDR(args.ngf*8*2, args.ngf*4, 0.),

            # Layer -3
            UCBDR(args.ngf*4*2, args.ngf*2, 0.),

            # Layer -2
            UCBDR(args.ngf*2*2, args.ngf, 0.),

            # Layer -1
            UCT(args.ngf*2, 3)
        ])
        
        self.decoder.apply(self.weight_init)
        
    def weight_init(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0,std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
        if isinstance(module, nn.BatchNorm2d):
            module.weight.data.uniform_()
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
