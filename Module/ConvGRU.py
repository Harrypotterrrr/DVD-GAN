import torch
import torch.nn as nn
from torch.nn import init

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=torch.sigmoid):

        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.activation = activation

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, x, prev_state=None):

        if prev_state is None:

            # get batch and spatial sizes
            batch_size = x.data.size()[0]
            spatial_size = x.data.size()[2:]

            # generate empty prev_state, if None is provided
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            # prev_state = torch.zeros(state_size)

            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([x, prev_state], dim=1)

        update = self.activation(self.update_gate(stacked_inputs))
        reset = self.activation(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        """
        Generates a multi-layer convolutional GRU.
        :param input_size: integer. depth dimension of input tensors.
        :param hidden_sizes: integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        :param kernel_sizes: integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        :param n_layers: integer. number of chained `ConvGRUCell`.
        """

        super().__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = nn.ModuleList()

        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            # name = 'ConvGRUCell_' + str(i).zfill(2)
            # setattr(self, name, cell)
            # cells.append(getattr(self, name))
            cells.append(cell)

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        input_ = x
        output = []

        if hidden is None:
            hidden = [None] * self.n_layers

        for i in range(self.n_layers):

            cell = self.cells[i]
            cell_hidden = hidden[i]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden) # TODO comment
            output.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return output


if __name__ == "__main__":

    # Generate a ConvGRU with 3 cells
    # input_size and hidden_sizes reflect feature map depths.
    # Height and Width are preserved by zero padding within the module.

    model = ConvGRU(input_size=8, hidden_sizes=[32, 64, 16], kernel_sizes=[3, 5, 3], n_layers=3).cuda()

    model = nn.DataParallel(model, device_ids=[0, 1])

    x = torch.rand([8, 8, 64, 64], dtype=torch.float32).cuda()
    hidden_state = None
    output = model(x, hidden_state)

    # output is a list of sequential hidden representation tensors
    print(type(output))  # list

    # final output size
    print(output[-1].size())  # torch.Size([1, 16, 64, 64])
