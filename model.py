import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

def get_active_func(s):
    if s == "ReLU":
        return nn.ReLU()
    elif s == "LeakyReLU":
        return nn.LeakyReLU()
    elif s == "Tanh":
        return nn.Tanh()
    else:
        print("activation function %s, is not known, defaulting to LeakyReLU."%(s))
        return nn.LeakyReLU()
    

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.convs = nn.ModuleList()
        self.fc = nn.Linear(6656, self.latent_dim)
        
        hidden_dims = [64, 128, 256, 512, 512]

        in_channels = 1
        for h_dim in hidden_dims:
            self.convs.append(
                ConvLayer(in_channels,h_dim,5,2,activation_function="ReLU") 
            )
            in_channels = h_dim
    
    def reparameterisation(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def kl_divergence(self, mu, logvar):
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / self.latent_dim

    def encode(self, input):
        x = input
        for conv in self.convs:
            x = conv(x)
        out = torch.flatten(x,start_dim=1)
        out = self.fc(out)
        return out, out

    def forward(self, input):
        mu, logvar = self.encode(input)
        kld = self.kl_divergence(mu, logvar)
        z = self.reparameterisation(mu, logvar)
        return z, kld


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim
    ):
        super().__init__()
        hidden_dims = [512, 256, 128, 64]
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0]*13)
        self.convs = nn.ModuleList()
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            if h_dim < 256:
                self.convs.append(
                    ConvLayer(in_channels,h_dim,5,2,output_padding=(1),activation_function="ReLU",transpose=True) 
                )
            else:
                self.convs.append(
                    ConvLayer(in_channels,h_dim,5,2,activation_function="ReLU",transpose=True) 
                )
            in_channels = h_dim
        self.final_layer = ConvLayer(64, 1, 3, 2, activation_function="Tanh",transpose=True) 

    def forward(self, input):
        x = self.decoder_input(input)
        x = torch.reshape(x, (64,512,1,13))
        for conv in self.convs:
            x = conv(x)
        x = self.final_layer(x)
        #drop row from tensor bc I cannit figure out how to get them the same dim
        x = x[:,:,0:128,:] 
        return x

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channel, 
        out_channel, 
        kernel_size, 
        stride=1, 
        padding=0, 
        output_padding=0,
        bias=True,
        activation_function="ReLU",
        transpose=False
    ):
        super().__init__()

        self.activation_function = get_active_func(activation_function)
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.transpose = transpose
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.bias = bias

        if self.transpose == True:
            self.layer_seq = nn.Sequential(
                nn.ConvTranspose2d(
                self.in_channel,
                self.out_channel,
                kernel_size=self.kernel_size,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding), 
                self.activation_function
            )
        else:
            self.layer_seq = nn.Sequential(
                nn.Conv2d(
                self.in_channel,
                self.out_channel,
                kernel_size=self.kernel_size,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding),
                self.activation_function
            )

    def forward(self, input):
        out = self.layer_seq(input)
        return out

