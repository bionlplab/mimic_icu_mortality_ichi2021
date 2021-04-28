import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchtuples as tt
import tqdm
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from torch import nn
from torch.nn import Parameter, Dropout, ReLU, LogSoftmax, Conv1d, Softmax, ELU
from torchtuples.practical import MLPVanilla

from wcm.mimic.survival.utils import Data, load_data, Config, SAPS_COLUMNS, LABLES_COLUMNS, TEXT_FEATURES_COLUMNS

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


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, alpha=.2, concat=True, dropout=0.5):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.a1 = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.a2 = nn.Parameter(torch.FloatTensor(out_features, 1))
        self.softmax = Softmax(dim=1)
        self.dropout = Dropout(dropout)
        self.elu = ELU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data, gain=np.sqrt(2.0))
        nn.init.xavier_normal_(self.a1.data, gain=np.sqrt(2.0))  # Implement Xavier Uniform
        nn.init.xavier_normal_(self.a2.data, gain=np.sqrt(2.0))

    def forward(self, input, adj):
        h = torch.matmul(input, self.weight)
        # N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(1,2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return self.elu(h_prime)
        else:
            return h_prime


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# ADJACENCY_MATRIX = torch.tensor([
#     [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # atelectasis
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cardiomegaly
#     [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # consolidation
#     [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # edema
#     [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # enlarged cardiomediastinum
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fracture
#     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # lung lesion
#     [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # lung opacity
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no finding
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # pleural effusion
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pleural others
#     [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # pneumonia
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pneumothorax
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # support devices
# ], dtype=torch.float, device=device)


class GCNFeatures1(nn.Module):
    def __init__(self, adj_matrix, dropout=0.5):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.log_softmax = LogSoftmax(dim=1)

        self.saps_features = len(SAPS_COLUMNS)
        self.saps_out_features = self.saps_features
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        # self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)
        channel_in = 768
        length_in = 512
        kernel_size = 2
        stride = 1
        channel_out = len(adj_matrix)
        length_out = int((length_in - kernel_size) / stride + 1)
        self.conv1d = Conv1d(channel_in, channel_out, kernel_size, stride=1)

        gcn_hidden_features = 64
        gcn_node_features = 64

        self.gcn1 = GraphConvolution(length_out, gcn_hidden_features)
        # self.gcn3 = GraphConvolution(gcn_node_features, gcn_node_features)
        self.gcn2 = GraphConvolution(gcn_hidden_features, gcn_node_features)
        # self.gcn2 = GraphConvolution(length_out, 32)

        # self.node_features_net = MLPVanilla(len(ADJACENCY_MATRIX), [], len(ADJACENCY_MATRIX), dropout=dropout)

        # adjacency matrix
        adj_d = torch.diag_embed(adj_matrix.sum(dim=1))
        inv_sqrt_adj_d = adj_d.pow(-0.5)
        inv_sqrt_adj_d[torch.isinf(inv_sqrt_adj_d)] = 0
        self.adj_a = inv_sqrt_adj_d.mm(adj_matrix).mm(inv_sqrt_adj_d)

        in_features = self.saps_out_features + gcn_node_features
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input):
        # print(input.shape)

        saps_input = input[:, :self.saps_features]
        # print(saps_input.shape)

        text_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)

        x = text_features_input.view(-1, 512, 768)
        x = x.transpose(1, 2)
        # print(x.shape)
        x = self.conv1d(x)
        # print(x.shape)

        x = self.gcn1(x, self.adj_a)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.gcn3(x, self.adj_a)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.gcn2(x, self.adj_a)

        # node_features = self.log_softmax(x)
        node_features = self.relu(x)

        # print(node_features.shape)
        # exit(1)

        node_features = node_features.mean(dim=1)
        input = torch.cat([saps_out, node_features], dim=1)
        out = self.score_net(input)
        return out


def train_test_pycox(train_ds, dev_ds, test_ds):
    # train_ds, dev_ds, test_ds = data.split3(fold)
    for ds in (train_ds, dev_ds, test_ds):
        to_pycox(ds)
        # ds.describe()

    train_batch_size = 64  # train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_pycox.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj_matrix = torch.tensor(data.adj_matrix, dtype=torch.float, device=device)
    # adj_matrix = torch.tensor([
    #     [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # atelectasis
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cardiomegaly
    #     [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # consolidation
    #     [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # edema
    #     [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # enlarged cardiomediastinum
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fracture
    #     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # lung lesion
    #     [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # lung opacity
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no finding
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # pleural effusion
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pleural others
    #     [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # pneumonia
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pneumothorax
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # support devices
    # ], dtype=torch.float, device=device)

    dropout = 0.5
    if config.saps():
        net = MLPSAPS(dropout)
    elif config.saps_labels():
        net = MLPLabels1(dropout)
    elif config.saps_text_features():
        net = MLPFeatures1(dropout)
    elif config.saps_text_token_features_gcn():
        net = GCNFeatures1(adj_matrix, dropout)
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


def train_test_pycox_fold(fold: int = 0):
    train_ds, dev_ds, test_ds = data.split3(fold)
    cindex = train_test_pycox(train_ds, dev_ds, test_ds)
    print('cindex: %.4f' % cindex)


def bootstrap(data):
    config.verbose = False
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d%H%M%S")

    if config.saps():
        name = 'saps'
    elif config.saps_labels():
        name = 'saps_labels'
    elif config.saps_text_features():
        name = 'saps_textfeatures'
    elif config.saps_text_token_features_gcn():
        name = 'saps_text_token_gcn'
    else:
        raise KeyError

    output = top / '{}_{}.txt'.format(name, date_time)
    with open(top / output, mode='w') as file:
        for i in tqdm.tqdm(range(20)):
            train_ds, dev_ds, test_ds = data.bootstrap(1234)
            cindex = train_test_pycox(train_ds, dev_ds, test_ds)
            file.write(str(i) + ' ' + '%.4f' % cindex + '\n')
            file.flush()


    #
    #     train_val_test_file = str(top) + '/bootstrap/train-val-test-' + str(i) + '.csv'
    #     c = train_test_pycox(top, train_val_test_file, i)
    #     file.write(str(i) + ' ' + '%.4f' % c + '\n')
    #
    # with open(str(top) + '/bootstrap/gcn-fusion-.txt', mode='w') as file:
    #     for i in tqdm.tqdm(range(200)):
    #         train_val_test_file = str(top) + '/bootstrap/train-val-test-' + str(i) + '.csv'
    #         c = train_test_pycox(top, train_val_test_file, i)
    #         file.write(str(i) + ' ' + '%.4f' % c + '\n')


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

    # cindex = train_test_pycox(data, fold=4)
    # print('cindex: %.4f' % cindex)

    # config.verbose = False
    # for fold in range(5):
    #     cindex = train_test_pycox_fold(fold)
    #     print('%.4f' % cindex, end='\t')

    bootstrap(data)
