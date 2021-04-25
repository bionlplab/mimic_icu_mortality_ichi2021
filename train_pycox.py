import math
import random
from pathlib import Path

import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from torch import nn
from torch.nn import Parameter, Dropout, ReLU, LogSoftmax, Conv1d
import torch.nn.functional as F
from torchtuples.practical import MLPVanilla
from transformers import Conv1D

from wcm.mimic.survival.utils import Data, load_data, Config, SAPS_COLUMNS, LABLES_COLUMNS, TEXT_FEATURES_COLUMNS, \
    TEXT_TOKEN_FEATURES_COLUMNS

np.random.seed(1234)
random.seed(1234)


def to_pycox(data: Data):
    columns = []
    if data.config.has_saps:
        columns += SAPS_COLUMNS
    if data.config.has_labels:
        columns += LABLES_COLUMNS
    if data.config.has_text_features:
        columns += TEXT_FEATURES_COLUMNS
    data.tte = data.df['time-to-event'].to_numpy(dtype=float)
    data.event = data.df['event'].to_numpy(dtype=bool)
    data.x_pycox = data.df[columns].values.astype('float32')
    data.y_pycox = (data.tte, data.event)

    if data.config.has_text_token_features:
        text_token_features = np.array(data.text_token_features).reshape((data.x_pycox.shape[0], -1))
        # print(text_token_features.shape)
        data.x_pycox = np.concatenate([data.x_pycox, text_token_features], axis=1)



class MLPSAPS(nn.Module):
    def __init__(self, out_features, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS)
        self.saps_net = MLPVanilla(in_features, [in_features], out_features, dropout=dropout)

    def forward(self, input):
        return self.saps_net(input)


class MLPLabels1(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS) + len(LABLES_COLUMNS)
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        out = self.score_net(input)
        return out


class MLPLabels2(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        self.label_features = len(LABLES_COLUMNS)
        self.label_net = MLPVanilla(self.label_features, [], self.label_features, dropout=dropout)

        in_features = self.saps_features + self.label_features
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        label_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)
        label_out = self.label_net(label_input)

        input = torch.cat([saps_out, label_out], dim=1)
        out = self.score_net(input)
        return out


class MLPFeatures1(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)

        self.text_features = len(TEXT_FEATURES_COLUMNS)
        self.text_out_features = 32
        self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)

        in_features = self.saps_features + self.text_out_features
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        text_features_input = input[:, self.saps_features:]

        text_features_out = self.text_features_net(text_features_input)

        input = torch.cat([saps_input, text_features_out], dim=1)
        out = self.score_net(input)
        return out


class MLPFeatures2(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)
        self.saps_out_features = self.saps_features
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        self.text_features = len(TEXT_FEATURES_COLUMNS)
        self.text_out_features = 32
        self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)

        in_features = self.saps_out_features + self.text_out_features
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        text_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)
        text_features_out = self.text_features_net(text_features_input)

        input = torch.cat([saps_out, text_features_out], dim=1)
        out = self.score_net(input)
        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ADJACENCY_MATRIX = torch.tensor([
    [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # atelectasis
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cardiomegaly
    [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # consolidation
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # edema
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # enlarged cardiomediastinum
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fracture
    [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # lung lesion
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # lung opacity
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no finding
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # pleural effusion
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pleural others
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # pneumonia
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pneumothorax
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # support devices
], dtype=torch.float, device=device)


class GCNFeatures1(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.log_softmax = LogSoftmax(dim=1)

        self.saps_features = len(SAPS_COLUMNS)
        self.saps_out_features = self.saps_features
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        self.text_features = len(TEXT_TOKEN_FEATURES_COLUMNS)
        # self.text_out_features = 256
        self.gcn_hidden_features = 64

        # self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)
        channel_in = 768
        length_in = 512
        kernel_size = 4
        stride = 1
        channel_out = 14
        length_out = int((length_in - kernel_size) / stride + 1)
        self.conv1d = Conv1d(channel_in, channel_out, kernel_size, stride=1)

        self.gcn1 = GraphConvolution(length_out, self.gcn_hidden_features)
        self.gcn2 = GraphConvolution(self.gcn_hidden_features, len(ADJACENCY_MATRIX))
        # self.gcn2 = GraphConvolution(length_out, len(ADJACENCY_MATRIX))

        # self.node_features_net = MLPVanilla(len(ADJACENCY_MATRIX), [], len(ADJACENCY_MATRIX), dropout=dropout)

        # adjacency matrix
        adj_d = torch.diag_embed(ADJACENCY_MATRIX.sum(dim=1))
        inv_sqrt_adj_d = adj_d.pow(-0.5)
        inv_sqrt_adj_d[torch.isinf(inv_sqrt_adj_d)] = 0
        self.adj_a = inv_sqrt_adj_d.mm(ADJACENCY_MATRIX).mm(inv_sqrt_adj_d)

        in_features = self.saps_out_features + len(ADJACENCY_MATRIX)
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input):
        # print(input.shape)

        saps_input = input[:, :self.saps_features]
        # print(saps_input.shape)

        text_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)

        # x = text_features_input.unsqueeze(1).repeat(1, len(ADJACENCY_MATRIX), 1)
        x = text_features_input.view(-1, 512, 768)
        x = x.transpose(1, 2)
        # print(x.shape)
        x = self.conv1d(x)
        # print(x.shape)

        x = self.relu(self.gcn1(x, self.adj_a))
        # print(x.shape)
        # exit(1)
        x = self.dropout(x)
        x = self.gcn2(x, self.adj_a)
        node_features = self.log_softmax(x)
        # print(node_features.shape)
        # exit(1)

        node_features = node_features.mean(dim=1)
        input = torch.cat([saps_out, node_features], dim=1)
        out = self.score_net(input)
        return out


def train_test_pycox(data, fold: int = 0):

    train_ds, dev_ds, test_ds = data.split3(fold)

    for ds in (train_ds, dev_ds, test_ds):
        to_pycox(ds)
        # ds.describe()

    train_batch_size = 64  # train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_pycox.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    dropout = 0.5
    if config.saps():
        net = MLPSAPS(dropout)
    elif config.saps_labels():
        net = MLPLabels1(dropout)
    elif config.saps_text_features():
        net = MLPFeatures1(dropout)
    elif config.saps_text_token_features_gcn():
        net = GCNFeatures1(dropout)
    else:
        raise KeyError

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.0001)
    callbacks = [tt.callbacks.EarlyStopping(file_path=str(top / 'foo.pt'))]
    epochs = 250
    model.fit(train_ds.x_pycox, train_ds.y_pycox, train_batch_size, epochs, callbacks,
              shuffle=True,
              val_data=tt.tuplefy(dev_ds.x_pycox, dev_ds.y_pycox),
              val_batch_size=dev_batch_size,
              verbose=config.verbose)
    model.compute_baseline_hazards()
    # y = model.predict_cumulative_hazards(test_ds.x_pycox)
    # print(y)

    surv = model.predict_surv_df(test_ds.x_pycox)
    # print(surv)
    ev = EvalSurv(surv, test_ds.tte, test_ds.event, censor_surv='km')
    c = ev.concordance_td()
    return c


if __name__ == '__main__':
    top = Path.home() / 'Data/MIMIC-CXR'

    print('SAPS', len(SAPS_COLUMNS))
    print('Labels', len(LABLES_COLUMNS))

    config = Config()
    config.tte_int = True
    config.has_saps = True
    config.has_labels = False
    config.has_text_features = False
    config.has_text_token_features = True
    config.use_gcn = True

    data = load_data(top, config)
    cindex = train_test_pycox(data, fold=4)
    print('cindex: %.4f' % cindex)

    # config.verbose = False
    # for i in range(5):
    #     cindex = train_test_pycox(data, fold=i)
    #     print('%.4f' % cindex, end='\t')
