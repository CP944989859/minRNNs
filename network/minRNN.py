__author__ = 'CP'
import torch
from torch.autograd import Variable

class RNN(torch.nn.Module):
    def __init__(self,cell_class, num_layer, input_size, hidden_size, use_bias = False):
    # def __init__(self, minGRUcell, num_layer, input_size, hidden_size, use_bias = False):
        super(RNN,self).__init__()
        self.num_layers = num_layer
        self.hidden_size = hidden_size
        for layer in range(num_layer):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(layer_input_size, hidden_size, use_bias)
            setattr(self, 'cell_{}'.format(layer), cell)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_param(self):
        for layer in range(self.num_layer):
            cell = self.get_cell(layer)
            cell.reset_param()

    def forward(self, input_, hx = None):
        max_time, batch_size, _ = input_.size()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())   ##
            hx = [(hx, hx) for _ in range(self.num_layers)]

        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, (layer_h_n, layer_c_n) = RNN._forward_rnn(cell = cell, input_= input_, hx = hx[layer])
            input_ = layer_output
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output  ##
        return output, new_hx

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        # max_time = input_.size(1)
        output = []
        for time in range(max_time):
            (h_next, c_next) = cell(input_[time], hx)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output,0)
        return output, hx


class minLSTMcell(torch.nn.Module):
    def __init__(self, input_size, num_hidden, x2h_bias = False):
        super(minLSTMcell, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.x2h = torch.nn.Linear(input_size, 3*num_hidden, bias = x2h_bias)

    def forward(self, input, hx):
        x = input
        hx = hx[0]
        gates = self.x2h(x)
        ingate, forgetgate, cellgate = gates.chunk(3,1)         # further simplification
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        ingate = ingate/(ingate + forgetgate)
        forgetgate = 1 - ingate

        cy = cellgate
        hy = torch.mul(hx, forgetgate) + torch.mul(ingate, cellgate)
        # cy = torch.mul(hx, forgetgate) + torch.mul(ingate, cellgate)
        # hy = torch.mul(outgate, torch.tanh(cy))
        return (hy,cy)

class minGRUcell(torch.nn.Module):
    def __init__(self, input_size, num_hidden, x2h_bias = False):
        super(minGRUcell, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.x2h = torch.nn.Linear(input_size, 2*num_hidden, bias = x2h_bias)

    def forward(self, input, hx):
        hx = hx[0]
        gates = self.x2h(input)
        ingate, cellgate = gates.chunk(2,1)
        ingate = torch.sigmoid(ingate)
        forgetgate = 1 - ingate
        cy = cellgate
        hy = torch.mul(hx, forgetgate) + torch.mul(ingate, cellgate)
        return (hy,cy)
