import torch.nn as nn
import torch
import torch.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim = 256, outer_dim = 2048, shrink_factor = 2,input_dim = 162506) :
        super().__init__()
        assert outer_dim // latent_dim % shrink_factor == 0

        self.shrink_factor = shrink_factor
        self.encoder_layers = []
        self.encoder_layers.append(nn.Linear(input_dim, outer_dim))
        for i in range(outer_dim // latent_dim / shrink_factor -1 ):
            self.encoder.layers.append(nn.Linear(outer_dim // (2**i), outer_dim // (2** (i + 1))))
        self.decoder_layers = []
        for i in range(outer_dim // latent_dim / shrink_factor -1 , -1):
            self.decoder.layers.append(nn.Linear(outer_dim // (2**i+1), outer_dim // (2** (i))))
        self.decoder_layers.append(nn.Linear(outer_dim // (2**i), input_dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.relu(x)

        encoded = x
        decoded = encoded

        for i, layer in enumerate(self.decoder_layers):
            decoded = layer(decoded)
            if i != (len(self.decoder_layers) -1):
                decoded = self.relu(x)
        return encoded, decoded # worry about the sign stuff in the training and validation steps


