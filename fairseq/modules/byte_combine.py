import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from command import configs


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_.
    Adopted from the AllenNLP implementation.
    """

    def __init__(self, input_dim: int, num_layers: int = 1):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self.activation = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            # As per comment in AllenNLP:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            nn.init.constant_(layer.bias[self.input_dim:], 1)

            nn.init.constant_(layer.bias[: self.input_dim], 0)
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            projection = layer(x)
            proj_x, gate = projection.chunk(2, dim=-1)
            proj_x = self.activation(proj_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (gate.new_tensor([1]) - gate) * proj_x
        return x


class ByteCombineCNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn='relu',
                 filters=[(1, 4), (2, 8), (3, 12)] + [(i, 4 * i) for i in range(4, configs.byte_len)],
                 highway_layers=2):

        # Pytorch will search for the most efficient convolution implementation
        torch.backends.cudnn.benchmark = True

        # TODO: increase filters once the byte fields went to 8
        super().__init__()

        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(
                nn.Conv1d(input_dim, out_c, kernel_size=width)
            )

        last_dim = sum(f[1] for f in filters)

        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None

        self.projection = nn.Linear(last_dim, output_dim)

    def forward(self, features):
        # features size: Batch x Seq x byte_len x Emb_dim
        B = features.size(0)
        T = features.size(1)
        byte_len = features.size(2)
        emb_dim = features.size(3)

        # BTC -> BCT, BTC: batch, sequence, embedding size
        features = features.transpose(2, 3).view(-1, emb_dim, byte_len)

        conv_result = []

        for conv in self.convolutions:
            x = conv(features)
            x, _ = torch.max(x, -1)
            x = F.relu(x)
            conv_result.append(x)

        x = torch.cat(conv_result, dim=-1)

        if self.highway is not None:
            x = self.highway(x)
        x = self.projection(x)
        x = x.view(B, T, -1)

        return x


class ByteCombineSUM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # expect input of size Batch x Seq x byte_len x input_dim
        print(x.size())
        exit()
        return torch.sum(x, dim=-2)


class ByteCombineConcat(nn.Module):
    def __init__(self, input_dim, output_dim, byte_len):
        super().__init__()
        self.linear = nn.Linear(byte_len * input_dim, output_dim)

    def forward(self, x):
        # expect input of size Batch x Seq x byte_len x input_dim
        return self.linear(x.squeeze(-1))
