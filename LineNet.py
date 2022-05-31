import numpy as np
import torch
from torch import nn
from lib import MLP, CrossAttention, SelfAttention
from torch.nn import functional as F
from Decoder import Decoder

from utils import plot_attention_score

BUS_NUM = 30
LINE_NUM = 41
HIDDEN_SIZE = 128


class GCNBlock(nn.Module):
    def __init__(self, mode):
        super(GCNBlock, self).__init__()
        if mode == 'bus':
            self.feature_num = BUS_NUM
            self.graph = np.loadtxt('./Data/graph/graph matrix of 30 bus.csv', delimiter=',')
        elif mode == 'branch':
            self.feature_num = LINE_NUM
            self.graph = np.loadtxt('./Data/graph/inverse graph matrix of 30 bus.csv', delimiter=',')
        else:
            raise ValueError
        
        self.graph = torch.tensor(self.graph)
        self.A_hat = self.get_A_hat(self.graph)
        self.linear = nn.Linear(HIDDEN_SIZE * self.feature_num, HIDDEN_SIZE * self.feature_num)
        self.linear2 = nn.Linear(HIDDEN_SIZE * self.feature_num, HIDDEN_SIZE * self.feature_num)

    def get_A_hat(self, graph):
        graph = graph + torch.diag(torch.ones(self.graph.shape[0]))  # add self loop
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        A_hat = torch.mm(torch.mm(D, graph), D)
        A_hat = A_hat.type(torch.float32)
        return A_hat

    def forward(self, x, batch_size, device):
        self.A_hat = self.A_hat.to(device)

        # x.shape == (batch_size, hidden_size, feature_num)
        # A_hat.shape == (feature_num, feature_num)
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        x = x.reshape(batch_size, -1, self.feature_num).transpose(1, 2)    # (batch_size, feature_num, hidden_size)
        x = torch.matmul(self.A_hat, x).transpose(1, 2)                    # (batch_size, hidden_size, feature_num)
        x = F.relu(x)
        
        temp = x

        x = x.reshape(batch_size, -1)
        x = self.linear2(x)
        x = x.reshape(batch_size, -1, self.feature_num).transpose(1, 2)    # (batch_size, feature_num, hidden_size)
        x = torch.matmul(self.A_hat, x).transpose(1, 2)                    # (batch_size, hidden_size, feature_num)
        x = F.relu(x)

        x = x + temp                                                       # (batch_size, hidden_size, feature_num)
        return x


class LineNet(nn.Module):
    def __init__(self, hidden_size):
        super(LineNet, self).__init__()
        self.hidden_size = hidden_size
        self.bus_encoder = nn.Sequential(
            MLP(24*BUS_NUM, 64*BUS_NUM),
            MLP(64*BUS_NUM, hidden_size*BUS_NUM),
            MLP(hidden_size*BUS_NUM, hidden_size*BUS_NUM)
        )
        self.line_encoder = nn.Sequential(
            MLP(24*LINE_NUM, 64*LINE_NUM),
            MLP(64*LINE_NUM, hidden_size*LINE_NUM),
            MLP(hidden_size*LINE_NUM, hidden_size*LINE_NUM),
        )
        self.gcn = GCNBlock('bus')
        self.igcn = GCNBlock('branch')

        self.l2l = SelfAttention(hidden_size, num_attention_heads=2)
        self.b2b = SelfAttention(hidden_size, num_attention_heads=2)
        self.b2l = CrossAttention(hidden_size, num_attention_heads=2)

        self.decoder = Decoder(hidden_size)

    def forward(self, input_data, batch_size, device):
        # input_data.shape == (32, 24, 71)
        load_data = input_data[:, :, :BUS_NUM].reshape(batch_size, -1)
        line_data = input_data[:, :, BUS_NUM:].reshape(batch_size, -1)

        load_encode_stage_one = self.bus_encoder(load_data).reshape(batch_size, -1, BUS_NUM)
        line_encode_stage_one = self.line_encoder(line_data).reshape(batch_size, -1, LINE_NUM)
        
        gcn_states = self.gcn(load_encode_stage_one, batch_size, device)
        igcn_states = self.igcn(line_encode_stage_one, batch_size, device)

        # TODO: or cat the two results
        load_encode_state_two = gcn_states + load_encode_stage_one           # (32, 128, 30)
        line_encode_state_two = igcn_states + line_encode_stage_one          # (32, 128, 41)

        load_encode_state_two = load_encode_state_two.transpose(1, 2)        # (32, 30, 128)
        line_encode_state_two = line_encode_state_two.transpose(1, 2)        # (32, 41, 128)

        b2b_attention = self.b2b(load_encode_state_two)                      # (32, 30, 128)
        l2l_attention = self.l2l(line_encode_state_two)                      # (32, 41, 128)
        # b2l_attention = self.b2l(l2l_attention, b2b_attention)               # (32, 41, 128)

        b2l_attention, atten_score = self.b2l(l2l_attention, b2b_attention, return_scores=True)               # (32, 41, 128)

        # for attention plot
        foo = atten_score[0, 0, :, :].cpu().detach().numpy()
        plot_attention_score(foo, 1, "Attention Head 1")
        foo = atten_score[0, 1, :, :].cpu().detach().numpy()
        plot_attention_score(foo, 2, "Attention Head 2")

        return self.decoder(b2l_attention)


def main():
    print('Start')

    model = LineNet(HIDDEN_SIZE).to('cuda')
    # print("The model have {} parameters in total.".format(sum(x.numel() for x in model.parameters())))

    input_tensor = torch.rand([32, 24, 71], device='cuda')
    output_cls, output_reg = model(input_tensor, 32, 'cuda')
    print(output_cls.shape, output_reg.shape)
    
if __name__ == "__main__":
    main()

