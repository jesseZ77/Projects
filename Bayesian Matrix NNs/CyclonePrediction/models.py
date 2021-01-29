import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist


def make_bayes_model(torch_model, prior_sd=5):
    def bayes_model(X, y):
        params = torch_model.state_dict()
        priors = {k: dist.Normal(
            loc=torch.zeros_like(v),
            scale=prior_sd * torch.ones_like(v)).to_event(len(v.shape)) for k, v in params.items()}

        lifted_module = pyro.random_module("module", torch_model, priors)
        lifted_reg_torch_model = lifted_module()
        y_pred = lifted_reg_torch_model(X)

        tausq = pyro.sample("tausq", dist.InverseGamma(0.1, 0.1))

        with pyro.plate("data", len(y)):
            pyro.sample("obs", dist.Normal(y_pred, tausq).to_event(1), obs=y)

        return y_pred
    return bayes_model


class MatrixLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.activation = activation

        self.U = nn.Parameter(torch.empty([output_dim[0], input_dim[0]]))
        self.V = nn.Parameter(torch.empty([input_dim[1], output_dim[1]]))
        self.B = nn.Parameter(torch.empty(output_dim))

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)
        torch.nn.init.kaiming_normal_(self.B)

    def forward(self, i):
        x = torch.matmul(torch.matmul(self.U, i), self.V) + self.B
        if self.activation is None:
            return x
        return self.activation(x)


class MatrixNetwork(nn.Module):
    def __init__(self, input_dim=(4, 2), output_size=2,
                 hidden_dim=(10, 10), n_hidden_layers=1,
                 hidden_activation=torch.sigmoid, output_activation=None):

        super().__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.hid_list = nn.ModuleList([MatrixLayer(self.input_dim, self.hidden_dim, self.hidden_activation)])

        for i in range(self.n_hidden_layers - 1):
            self.hid_list.append(MatrixLayer(self.hidden_dim, self.hidden_dim, torch.sigmoid))

        self.out = nn.Linear(self.hidden_dim[0] * self.hidden_dim[1], self.output_size)

    def forward(self, i):
        x = self.hid_list[0](i)

        for i in range(self.n_hidden_layers - 1):
            x = self.hid_list[i + 1](x)

        x = x.reshape(x.shape[0], -1)
        o = self.out(x)

        if self.output_activation is not None:
            return self.output_activation(o)
        else:
            return o


class LSTM(nn.Module):
    pass


class CNN(nn.Module):
    pass
