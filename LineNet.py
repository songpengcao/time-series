
import torch
from torch import nn
from lib import MLP


class LineNet(nn.Module):
    def __init__(self, hidden_size):
        super(LineNet, self).__init__()
        self.hidden_size = hidden_size
        self.stage_one_encoder = nn.Sequential(
            MLP(24, 64),
            MLP(64, 128),
            MLP(128, 64)
        )

        # self.decoder = # TODO

    def forward(self, input_data, batch_size, device):
        line_num = input_data.shape[1]
        self.encode_stage_one = torch.zeros([batch_size, line_num, self.hidden_size], device=device)

        for batch in range(batch_size):
            for line in range(line_num):
                print(input_data[batch][line].shape)
                temp = self.stage_one_encoder(input_data[batch][line])
                self.encode_stage_one[batch, line] = temp
        
        
        return self.encode_stage_one
        # return self.decoder(self.encode_stage_one)


def main():
    model = LineNet(64)
    print(model)

if __name__ == "__main__":
    main()

