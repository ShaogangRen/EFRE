import torch
import torch.nn as nn

class SequentialFlow(nn.Module):
    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))
        logdet = None
        if logpx is None:
            for i in inds:
                '''logdet_dx_dz if reverse=false  else  dz_dx'''
                x, _, logdet_dx_dz = self.chain[i](x, reverse=reverse)
                if logdet is None:
                    logdet = logdet_dx_dz
                else:
                    logdet += logdet_dx_dz
            return x, None, logdet
        else:
            #''' just ignore the jacobian for now'''
            for i in inds:
                '''logdet_dx_dz if reverse=false  else  logdet_dz_dx'''
                x, logpx, logdet_dx_dz = self.chain[i](x, logpx, reverse=reverse)
                if logdet is None:
                    logdet = logdet_dx_dz
                else:
                    logdet += logdet_dx_dz
            return x, logpx, logdet
