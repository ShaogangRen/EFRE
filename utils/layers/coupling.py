import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    def __init__(self, d, swap=False,  w_init_sigma=0.001, intermediate_dim=64):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(intermediate_dim),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(intermediate_dim),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, logpx=None, reverse=False):
        ''' swap: which part should be operated with; dim has to bee an even number '''
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d
        epsilon = 1.0E-10
        s_t = self.net_s_t(x[:, :in_dim]) ## size: 2*(x.shape[1] - self.d)
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.) + epsilon
        shift = s_t[:, out_dim:]

        ## [batch_size, 1]
        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), dim=1, keepdim=False)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)
        if logpx is None:
            return y, None, delta_logp
        else:
            logpx = torch.squeeze(logpx)
            delta_logp = torch.squeeze(delta_logp)
            if torch.cuda.is_available():
                delta_logp = delta_logp.cuda()
                logpx = logpx.cuda()
            else:
                delta_logp = delta_logp
                logpx = logpx
            return y, logpx + delta_logp, delta_logp

