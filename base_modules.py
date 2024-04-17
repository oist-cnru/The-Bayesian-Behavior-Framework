import torch
import torch.nn as nn


class UnsqueezeModule(nn.Module):
    def __init__(self, dim: int):
        super(UnsqueezeModule, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.dim)


def make_dcnn(feature_size, out_channels):
    dcnn = nn.Sequential(
        nn.ConvTranspose2d(feature_size, 64, [1, 4], 1, 0),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 16, [2, 4], [1, 2], [0, 1]),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 16, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 4, 2, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(8,
                           8,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           output_padding=1),
        nn.ReLU(),
        nn.Conv2d(8, out_channels=out_channels,
                  kernel_size=3, padding=1)
        )  # output size 16 x 64
    
    return dcnn


def make_cnn(n_channels):

    cnn_module_list = nn.ModuleList()
    cnn_module_list.append(nn.Conv2d(n_channels, 8, 4, 2, 1))
    cnn_module_list.append(nn.ReLU())
    cnn_module_list.append(nn.Conv2d(8, 16, 4, 2, 1))
    cnn_module_list.append(nn.ReLU())
    cnn_module_list.append(nn.Conv2d(16, 16, 4, 2, 1))
    cnn_module_list.append(nn.ReLU())
    cnn_module_list.append(nn.Conv2d(16, 64, [2, 4], 2, [0, 1]))
    cnn_module_list.append(nn.ReLU())
    cnn_module_list.append(nn.Conv2d(64, 256, [1, 4], [1, 4], 0))
    cnn_module_list.append(nn.ReLU())

    cnn_module_list.append(nn.Flatten())
    phi_size = 256

    return nn.Sequential(*cnn_module_list), phi_size


def make_mlp(input_size, hidden_layers, output_size, act_fn, last_layer_linear=False):
    mlp = nn.ModuleList()
    last_layer_size = input_size
    for layer_size in hidden_layers:
        mlp.append(nn.Linear(last_layer_size, layer_size, bias=True))
        mlp.append(act_fn())
        last_layer_size = layer_size
    mlp.append(nn.Linear(last_layer_size, output_size, bias=True))
    if not last_layer_linear:
        mlp.append(act_fn())

    return nn.Sequential(*mlp)


class ContinuousActionQNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_layers=None, act_fn=nn.ReLU):
        super(ContinuousActionQNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256]
        self.input_size = input_size
        self.action_size = action_size
        self.output_size = 1
        self.hidden_layers = hidden_layers

        self.network_modules = nn.ModuleList()

        last_layer_size = input_size + action_size
        for layer_size in hidden_layers:
            self.network_modules.append(nn.Linear(last_layer_size, layer_size))
            self.network_modules.append(act_fn())
            last_layer_size = layer_size

        self.network_modules.append(nn.Linear(last_layer_size, self.output_size))

        self.main_network = nn.Sequential(*self.network_modules)

    def forward(self, x, a):

        q = self.main_network(torch.cat((x, a), dim=-1))

        return q


class ContinuousActionVNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=None, act_fn=nn.ReLU):
        super(ContinuousActionVNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256]
        self.input_size = input_size
        self.output_size = 1
        self.hidden_layers = hidden_layers

        self.network_modules = nn.ModuleList()

        last_layer_size = input_size
        for layer_size in hidden_layers:
            self.network_modules.append(nn.Linear(last_layer_size, layer_size))
            self.network_modules.append(act_fn())
            last_layer_size = layer_size

        self.network_modules.append(nn.Linear(last_layer_size, self.output_size))

        self.main_network = nn.Sequential(*self.network_modules)

    def forward(self, x):

        q = self.main_network(x)

        return q


class ContinuousActionPolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, output_distribution="Gaussian", hidden_layers=None, act_fn=nn.ReLU,
                 logsig_clip=None):
        super(ContinuousActionPolicyNetwork, self).__init__()

        if logsig_clip is None:
            logsig_clip = [-20, 2]
        if hidden_layers is None:
            hidden_layers = [256, 256]
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.logsig_clip = logsig_clip

        self.output_distribution = output_distribution  # Currently only support "Gaussian" or "DiracDelta"

        self.mu_layers = nn.ModuleList()
        self.logsig_layers = nn.ModuleList()

        last_layer_size = input_size
        for layer_size in hidden_layers:
            self.mu_layers.append(nn.Linear(last_layer_size, layer_size))
            self.mu_layers.append(act_fn())
            self.logsig_layers.append(nn.Linear(last_layer_size, layer_size))
            self.logsig_layers.append(act_fn())
            last_layer_size = layer_size
        self.mu_layers.append(nn.Linear(last_layer_size, self.output_size))
        self.logsig_layers.append(nn.Linear(last_layer_size, self.output_size))

        self.mu_net = nn.Sequential(*self.mu_layers)
        self.logsig_net = nn.Sequential(*self.logsig_layers)

    def forward(self, x):

        if self.output_distribution == "Gaussian":
            mu = self.mu_net(x)
            logsig = self.logsig_net(x).clamp(self.logsig_clip[0], self.logsig_clip[1])

            return mu, logsig

        else:
            raise NotImplementedError

    def get_log_action_probability(self, x, a):

        mu = self.mu_net(x)
        logsig = self.logsig_net(x).clamp(self.logsig_clip[0], self.logsig_clip[1])

        dist = torch.distributions.normal.Normal(loc=mu, scale=torch.exp(logsig))
        log_action_probability = dist.log_prob(a)

        return log_action_probability

    def sample_action(self, x, greedy=False):

        mu = self.mu_net(x)
        logsig = self.logsig_net(x).clamp(self.logsig_clip[0], self.logsig_clip[1])

        if greedy:
            return torch.tanh(mu).detach().cpu().numpy()

        else:
            dist = torch.distributions.normal.Normal(loc=mu, scale=torch.exp(logsig))
            sampled_u = dist.sample()

            return torch.tanh(sampled_u.detach().cpu()).numpy()
