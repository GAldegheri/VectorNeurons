import torch
import torch.nn as nn
from vn_layers import VNLinearLeakyReLU, VNLinearReLU, VNLinear

class VNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 use_relu=False):
        super(VNEncoder, self).__init__()
        vnlayer = VNLinearReLU if use_relu else VNLinear 
        self.layer1 = vnlayer(in_channels, 256)
        self.layer2 = vnlayer(256, 128)
        self.layer3 = VNLinear(128, out_channels)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
    
class VNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_relu=False):
        super(VNDecoder, self).__init__()
        vnlayer = VNLinearLeakyReLU if use_relu else VNLinear
        self.layer1 = vnlayer(in_channels, 128)
        self.layer2 = vnlayer(128, 256)
        self.layer3 = VNLinear(256, out_channels)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
    
class VNAutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels=64,
                 use_relu=False):
        super(VNAutoEncoder, self).__init__()
        self.encoder = VNEncoder(in_channels, latent_channels,
                                 use_relu=use_relu)
        self.decoder = VNDecoder(latent_channels, in_channels)
        
    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_, z        
    
if __name__=="__main__":
    
    model = VNAutoEncoder(576)
    x = torch.randn((32, 576, 3))
    x_, _ = model(x)
    print("x shape:", x.shape, 
          "- x_ shape:", x_.shape)