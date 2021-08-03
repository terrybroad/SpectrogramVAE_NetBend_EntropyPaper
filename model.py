import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

def get_active_func(s):
    if s == "ReLU":
        return nn.ReLU
    elif s == "LeakyReLU":
        return nn.LeakyReLU
    eluf s == "Tanh":
    return nn.Tanh
    else:
        print("activation function {}, is not known, defaulting to LeakyReLU.".format{s})
        return nn.LeakyReLU
    

Class Encoder(nn.Module):
    def __init__(
        self,
        size,
        latent_dim
    ):
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        self.convs = nn.ModuleList()
        self.fc = nn.Linear(9216, self.latent_dim)
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]

        in_channels = 1
        # Build Encoder
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
        return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / self.code_dim

    def encode(self, input):
        x = input
        for conv in enumerate(self.convs):
            x = conv(x)
        out = x.squeeze(2).squeeze(2)
        out = self.fc(out)
        return out, out

    def forward(self, input):
        mu, logvar = self.encode(input)
        kld = self.kl_divergence(mu, logvar)
        z = self.reparameterisation(mu, logvar)
        return z, kld


Class Decoder(nn.Module):
    def __init__(
        self,
        size,
        latent_dim
    ):
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0]*4)
        self.convs = nn.ModuleList()
        
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128, 64]

        in_channels = 1
        # Build Encoder
        for h_dim in hidden_dims:
            self.convs.append(
                ConvLayer(in_channels,h_dim,5,2,activation_function="ReLU",transpose=True) 
            )
            in_channels = h_dim
        self.final_layer = ConvLayer(hidden_dims[-1], 1, 5, 2, activation_function="Tanh",transpose=True) 

    def forward(self, input):
        x = self.decoder_input(input)
        for conv in enumerate(self.convs):
            x = conv(x)
        x = self.final_layer(x)

Class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channel, 
        out_channel, 
        kernel_size, 
        stride=1, 
        padding=0, 
        bias=True,
        activation_function="ReLU",
        transpose=False
    ):
        super().__init__()

        self.size = size
        self.activation_function = get_activ_function(activation_function)
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.transpose = transpose

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None


    def forward(self, input):
        if self.transpose == True:
            out = F.ConvTranspose2d(
                input,
                self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
            )
        out = nn.BatchNorm2d(out)
        return self.activation_function(out)

