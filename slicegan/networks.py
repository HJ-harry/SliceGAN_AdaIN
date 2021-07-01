import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def slicegan_nets(pth, Training, imtype, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            #use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5*(torch.tanh(self.convs[-1](x))+1)
            else:
                out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator


def slicegan_nets_disentangle(pth, Training, imtype, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a Generator and a Discriminator for SliceGAN-AdaIN.
    This function generates a generator with AdaIN code generator,
    additionally taking as input the code for granular size.
        :param Training: If training, we save params, if not, we load params from previous.
        This keeps the parameters consistent for older models
        :return: Generator, Discriminator
    """
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.MLP_mu = nn.ModuleList()
            self.MLP_sig = nn.ModuleList()
            self.AdaIN_code_gen = AdaIN_code_generator(128, 64)
            self.AdaIN = AdaIN()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.MLP_mu.append(nn.Linear(64, gf[lay+1]))
                self.MLP_sig.append(nn.Linear(64, gf[lay+1]))

        def forward(self, x, code):
            code_feat = self.AdaIN_code_gen(code)
            for conv, mlp_mu, mlp_sig in zip(self.convs[:-1], self.MLP_mu[:-1], self.MLP_sig[:-1]):
                x = conv(x)
                mu_y = mlp_mu(code_feat)
                # For non-negativity
                sig_y = F.relu_(mlp_sig(code_feat))
                x = F.relu_(self.AdaIN(x, mu_y, sig_y))
            #use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5*(torch.tanh(self.convs[-1](x))+1)
            else:
                out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator


class AdaIN_code_generator(nn.Module):
    def __init__(self, in_chans, chans):
        super().__init__()
        self.in_chans = in_chans
        self.chans = chans
        self.layers = nn.Sequential(
            nn.Linear(in_chans, chans),
            nn.ReLU(),
            nn.Linear(chans, chans),
            nn.ReLU(),
            nn.Linear(chans, chans),
            nn.ReLU(),
            nn.Linear(chans, chans)
        )
        self.activ = nn.ReLU()

    def forward(self, c):
        # c  : code
        # mu : mean
        # sig: std
        out = self.layers(c)
        return out


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mu_y, sig_y):
        eps = 1e-8
        mean_x = torch.mean(x, dim=[2, 3, 4])
        std_x = torch.std(x, dim=[2, 3, 4])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mean_y = mu_y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps
        std_y = sig_y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x) / std_x * std_y + mean_y
        return out