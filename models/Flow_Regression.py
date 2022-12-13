###### Code created by Shaogang Ren ###################
import torch
from utils.flow_model import FlowCell
from torch import nn
import torch.distributions as dists
import torch.optim as optim
from cdt.causality.pairwise.model import PairwiseModel
import numpy as np
from sklearn import preprocessing


class Flow_Regr(nn.Module):
    def __init__(self, data=None, sample_n=200, max_epoch=50, lr=0.01, w_init_sigma=0.001, flow_depth=3):
        super(Flow_Regr, self).__init__()
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = 20
        self.sample_n = sample_n
        scaler = preprocessing.StandardScaler().fit(data)
        data = scaler.transform(data)

        self.xdata = data
        self.max_ab = np.amax(self.xdata, axis=0)
        self.min_ab = np.amin(self.xdata, axis=0)

        self.depth = flow_depth
        self.z_dim = self.xdata.shape[1]
        print('data shape={}'.format(self.xdata.shape))
        self.data_size = self.xdata.shape[0]
        self.max_batch = self.data_size // self.batch_size
        if self.data_size % self.batch_size > 0:
            self.max_batch = self.max_batch + 1
        self.w_init_sigma = w_init_sigma
        self.cell = FlowCell(flow_depth=self.depth, z_dim=self.z_dim, w_init_sigma=self.w_init_sigma)
        self.solver = optim.Adam(self.cell.parameters(), lr=self.lr)
        self.sigma = 1.0

    def get_batch_data(self, it, bsize=None):
        if bsize is not None:
            dsize = bsize
        else:
            dsize = self.batch_size
        idx = it % self.max_batch
        if dsize * (idx + 1) > self.data_size:
            endidx = self.data_size
        else:
            endidx = dsize * (idx + 1)
        tt = self.xdata[dsize * idx:endidx, :]
        ts = torch.from_numpy(tt).float()
        if torch.cuda.is_available():
            ts = ts.cuda()
        return ts

    def get_random_data(self, bsize=None):
        if bsize is not None:
            dsize = bsize
        else:
            dsize = self.batch_size
        idxperm = np.random.permutation(self.data_size)
        idx = idxperm[0:dsize]
        tt = self.xdata[idx, :]
        ts = torch.from_numpy(tt).float()
        if torch.cuda.is_available():
            ts = ts.cuda()
        return ts


    def get_delta_data(self, approx_sample_N = 300):
        dmx = self.max_ab
        dmi = self.min_ab
        delta0 = (dmx[0] - dmi[0]) / (1.0 * approx_sample_N)
        delta1 = (dmx[1] - dmi[1]) / (1.0 * approx_sample_N)

        smp0 = [i * 1.0 * delta0 + dmi[0] for i in range(approx_sample_N)]
        smp1 = [i * 1.0 * delta1 + dmi[1] for i in range(approx_sample_N)]

        smp0 = np.expand_dims(smp0, axis=1)
        smp1 = np.expand_dims(smp1, axis=1)
        tdata_delta = np.concatenate([smp0, smp1], 1)

        tdata_delta = torch.from_numpy(tdata_delta).float()
        if torch.cuda.is_available():
            tdata_delta = tdata_delta.cuda()
        return tdata_delta, delta0, delta1


    def get_random_data_np(self, bsize=None):
        if bsize is not None:
            dsize = bsize
        else:
            dsize = self.batch_size
        idxperm = np.random.permutation(self.data_size)
        idx = idxperm[0:dsize]
        tt = self.xdata[idx, :]
        return tt

    def train(self, pair_id=None):
        findex = 0
        for ep in range(self.max_epoch):
            # print('ep={}'.format(ep))
            for it in range(self.max_batch):
                x_ = self.get_batch_data(findex)
                findex = findex + 1
                if x_.shape[0] < 2:
                    continue
                z, ndelt_px, _ = self.cell(x_)
                dist = dists.Normal(0, self.sigma)
                logp_z = torch.sum(dist.log_prob(z), 1)
                loss = -logp_z - ndelt_px
                mloss = torch.mean(loss)
                self.solver.zero_grad()
                mloss.backward()
                self.solver.step()
            if ep % 20 == 1:
                avg_llk = self.test_loglk_train(ep)
                print('pair_id={} ep={} loss={}  loglk_avg={} '.format(pair_id, ep, mloss.item(), avg_llk))
            if ep % 40 == 1:
                Vgva_px_mean, Vgvb_px_mean, Vgva_mean, Vgvb_mean, px_mean_a, px_mean_b = self.regression_error()
                print('ep={} Vgva_px_mean={}, Vgvb_px_mean={}, Vgva_mean={}, Vgvb_mean={}, px_mean_a={}, px_mean_b={}'.format(ep,
                                                                  Vgva_px_mean, Vgvb_px_mean, Vgva_mean, Vgvb_mean, px_mean_a, px_mean_b))
        print('pair_id={} loss={}  loglk_avg={} training done! '.format(pair_id, mloss.item(), avg_llk))
        return avg_llk

    def regression_error(self):
        sampling_size = min(self.sample_n, self.data_size)
        npdata = self.get_random_data_np(sampling_size)
        dim = npdata.shape[1]
        tdata = torch.from_numpy(npdata).float()
        if torch.cuda.is_available():
            tdata = tdata.cuda()
        '''==== step1: forward and backward for JJ matrix =='''
        a,b = 0,1

        xa = tdata[:, a]
        err_all_list = []
        err_all_px_list = []
        px_PTa_all_list = []
        for i in range(sampling_size):
            ta = tdata[i,:].repeat([sampling_size, 1])
            ta[:, a] = xa
            '''===x0 changes, x1 fixed ======='''
            z, ndelta_logp, _ = self.cell(ta)
            dist = dists.Normal(0, self.sigma)
            log_pz = torch.sum(dist.log_prob(z), 1)
            log_px = log_pz + ndelta_logp
            x_recon = self.cell.backward(z)
            sq_err = (x_recon - ta)**2
            err_all = torch.sum(sq_err, 0)
            px_one = torch.exp(log_px)
            #print('px_one')
            #print(px_one.shape)
            px =  torch.unsqueeze(px_one,1).repeat([1, dim])
            # print('px')
            # print(px.shape)
            px_PTa_all_list.append(torch.mean(px_one, 0))
            err_all_px = torch.sum(torch.mul(sq_err, px), 0)
            err_all_list.append(err_all)
            err_all_px_list.append(err_all_px)

        Vgva_mean = torch.mean(torch.stack(err_all_list, 0), 0)
        Vgva_px_mean = torch.mean(torch.stack(err_all_px_list, 0), 0)
        tt = torch.stack(px_PTa_all_list, 0)
        # print('tt')
        # print(tt.shape)
        px_PTa_mean =  torch.mean(torch.stack(px_PTa_all_list, 0), 0)

        xb = tdata[:, b]
        err_all_list = []
        err_all_px_list = []
        px_PTb_all_list = []
        for i in range(sampling_size):
            tb = tdata[i,:].repeat([sampling_size, 1])
            tb[:, b] = xb
            '''===x0 is fixed, x1 changes ======='''
            z, ndelta_logp, _ = self.cell(tb)
            dist = dists.Normal(0, self.sigma)
            log_pz = torch.sum(dist.log_prob(z), 1)
            log_px = log_pz + ndelta_logp
            x_recon = self.cell.backward(z)
            sq_err = (x_recon - tb)**2
            err_all = torch.sum(sq_err, 0)
            px_one = torch.exp(log_px)
            px = torch.unsqueeze(px_one, 1).repeat([1, dim])
            px_PTb_all_list.append(torch.mean(px_one, 0))
            err_all_px = torch.sum(torch.mul(sq_err, px), 0)
            err_all_list.append(err_all)
            err_all_px_list.append(err_all_px)

        Vgvb_mean = torch.mean(torch.stack(err_all_list, 0), 0)
        Vgvb_px_mean = torch.mean(torch.stack(err_all_px_list, 0), 0)
        px_PTb_mean = torch.mean(torch.stack(px_PTb_all_list, 0), 0)
        return Vgva_px_mean, Vgvb_px_mean, Vgva_mean, Vgvb_mean, px_PTa_mean, px_PTb_mean

    def causal_pair(self):
        sampling_size = min(self.sample_n, self.data_size)
        npdata = self.get_random_data_np(sampling_size)
        dim = npdata.shape[1]
        tdata = torch.from_numpy(npdata).float()
        if torch.cuda.is_available():
            tdata = tdata.cuda()
        '''==== step1: forward and backward for JJ matrix =='''
        a,b = 0,1

        xa = tdata[:, a]
        #err_all_list = []
        err_all_px_list = []
        for i in range(sampling_size):
            ta = tdata[i,:].repeat([sampling_size, 1])
            ta[:, a] = xa
            '''===x0 changes, x1 fixed ======='''
            z, ndelta_logp, _ = self.cell(ta)
            dist = dists.Normal(0, self.sigma)
            log_pz = torch.sum(dist.log_prob(z), 1)
            log_px = log_pz + ndelta_logp
            x_recon = self.cell.backward(z)
            sq_err = (x_recon - ta)**2
            #err_all = torch.sum(sq_err, 0)
            px =  torch.unsqueeze(torch.exp(log_px),1).repeat([1, dim])
            #print('sq_err shape{}'.format(sq_err.shape))
            #print('px shape{}'.format(px.shape))
            err_all_px = torch.sum(torch.mul(sq_err, px), 0)
            #err_all_list.append(err_all)
            err_all_px_list.append(err_all_px)

        #Vgva_all = torch.mean(torch.stack(err_all_list, 0), 0)
        Vgva_all_px = torch.mean(torch.stack(err_all_px_list, 0), 0)

        xb = tdata[:, b]
        #err_all_list = []
        err_all_px_list = []
        for i in range(sampling_size):
            tb = tdata[i,:].repeat([sampling_size, 1])
            tb[:, b] = xb
            '''===x0 is fixed, x1 changes ======='''
            z, ndelta_logp, _ = self.cell(tb)
            dist = dists.Normal(0, self.sigma)
            log_pz = torch.sum(dist.log_prob(z), 1)
            log_px = log_pz + ndelta_logp
            #print(log_px.shape)
            x_recon = self.cell.backward(z)
            sq_err = (x_recon - tb)**2
            err_all = torch.sum(sq_err, 0)
            #px =  torch.exp(log_px).repeat([1, dim])
            px = torch.unsqueeze(torch.exp(log_px), 1).repeat([1, dim])
            err_all_px = torch.sum(torch.mul(sq_err, px), 0)
            #err_all_list.append(err_all)
            err_all_px_list.append(err_all_px)

        #Vgvb_all = torch.mean(torch.stack(err_all_list, 0), 0)
        Vgvb_all_px = torch.mean(torch.stack(err_all_px_list, 0), 0)
        return Vgva_all_px[1], Vgvb_all_px[0]


    def test_loglk_train(self, ep):
        loglk_sum = 0
        test_batch = 3
        test_sample_size = 0
        for i_test in range(0, test_batch):
            x_ = self.get_batch_data(i_test)
            test_sample_size = test_sample_size + x_.shape[0]
            loglk_sum += torch.sum(self.loglikelihood(x_))
        loglk_mean = loglk_sum / test_sample_size
        return loglk_mean

    def loglikelihood(self, x):
        z, ndelt_px, J = self.cell(x)
        dist = dists.Normal(0, self.sigma)
        logp_z = torch.sum(dist.log_prob(z), 1)
        log_llk = logp_z + ndelt_px
        return log_llk


class FlowGraph(PairwiseModel):
    def __init__(self,sample_n=200, max_epoch=50, lr=0.01, w_init_sigma=0.001, flow_depth=2):
        super(FlowGraph, self).__init__()
        self.max_epoch = max_epoch
        self.lr = lr
        self.flow_depth = flow_depth
        self.w_init_sigma = w_init_sigma
        self.sample_n = sample_n

    def predict_proba(self, data_set, **kwargs):
        a, b = data_set
        ab = np.concatenate((a, b), axis=1)
        #ab = preprocessing.scale(ab)

        ''' standard '''
        scaler = preprocessing.StandardScaler().fit(ab)
        ab = scaler.transform(ab)

        '''scale to [0,1]'''
        max_ab = np.amax(ab, axis=0)
        min_ab = np.amin(ab, axis=0)
        #print('min_ab shape'.format(min_ab.shape))
        ab = (ab - min_ab)/(max_ab-min_ab)
        #print(ab)

        fg = Flow_Regr(data=ab, sample_n=self.sample_n, max_epoch=self.max_epoch, lr=self.lr,  w_init_sigma=self.w_init_sigma,
                           flow_depth=self.flow_depth)
        if torch.cuda.is_available():
            fg.cuda()
        avg_llk = fg.train(pair_id=kwargs["idx"])
        da, db = fg.causal_pair()

        del fg

        if da < db:
            return 1
        else:
            return -1


# ### get data #####
# data, graph = cdt.data.load_dataset('sachs')
# dt = []
# max_epoch = 2
#
# keys = data.keys()
# for k in keys:
#     dt.append(data[k])
# data_graph = np.transpose(np.array(dt))
#
# fg = FlowGraphCore(max_epoch = max_epoch, lr =0.000001, data = data_graph, w_init_sigma=0.000001)
# avg_llk = fg.train()
# fg.causal_graph1()
#



