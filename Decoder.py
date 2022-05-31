import numpy as np
import torch
from torch import nn
from lib import MLP
from torch.nn import functional as F

class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=1):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderCLS(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderCLS, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        output = self.fc(hidden_states)
        output = self.sigmoid(output)
        return output


# class Decoder(nn.Module):
#     def __init__(self, hidden_size):
#         super(Decoder, self).__init__()
#         self.decoderCLS = DecoderCLS(hidden_size)
#         self.decoderREG = DecoderRes(hidden_size+1)

#     def forward(self, hidden_states):
#         output_CLS = self.decoderCLS(hidden_states)  # (32, 41, 1)
#         output_REG = self.decoderREG(torch.cat([hidden_states, output_CLS], dim=-1)) # (32, 41, 1)
#         return output_CLS.squeeze(-1), output_REG.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.decoderCLS = DecoderCLS(hidden_size+1)
        self.decoderREG = DecoderRes(hidden_size)

    def forward(self, hidden_states):
        output_REG = self.decoderREG(hidden_states)                     # (32, 41, 1)
        hidden_states = torch.cat([hidden_states, output_REG], dim=-1)  # (32, 41, 129)
        output_CLS = self.decoderCLS(hidden_states)                     # (32, 41, 1)
        return output_CLS.squeeze(-1), output_REG.squeeze(-1)