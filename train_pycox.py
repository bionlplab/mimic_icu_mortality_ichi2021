import random
from pathlib import Path

import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from torch import nn
from torchtuples.practical import MLPVanilla
<<<<<<< Updated upstream

from utils import Data, load_data, Config, SAPS_COLUMNS, LABLES_COLUMNS, TEXT_FEATURES_COLUMNS
=======
from torch.nn import Parameter, Dropout, ReLU, LogSoftmax, Conv1d, Softmax, ELU
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

from utils import Data, load_data, Config, SAPS_COLUMNS, LABELS_COLUMNS, TEXT_FEATURES_COLUMNS, TEXT_TOKEN_FEATURES_COLUMNS, IMAGE_FEATURES_COLUMNS, ADJACENCY_MATRIX
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
=======
    if data.config.has_image_features:
        columns += IMAGE_FEATURES_COLUMNS

>>>>>>> Stashed changes
    data.tte = data.df['time-to-event'].to_numpy(dtype=float)
    data.event = data.df['event'].to_numpy(dtype=bool)
    data.x_pycox = data.df[columns].values.astype('float32')
    data.y_pycox = (data.tte, data.event)

<<<<<<< Updated upstream
=======
    if data.config.has_text_token_features:
        text_token_features = np.array(data.text_token_features).reshape((data.x_pycox.shape[0], -1))
        data.x_pycox = np.concatenate([data.x_pycox, text_token_features], axis=1)

>>>>>>> Stashed changes

class MLPSAPS(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS)
        self.saps_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        return self.saps_net(input)


<<<<<<< Updated upstream
class MLPLabels1(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS) + len(LABLES_COLUMNS)
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        out = self.score_net(input)
        return out


=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream

class MLPFeatures2(nn.Module):
=======
# Only SAPS features and image features
class MLPImageFeatures(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        self.image_features = len(IMAGE_FEATURES_COLUMNS)
        self.image_out_features = 32
        self.image_features_net = MLPVanilla(self.image_features, [], self.image_out_features, dropout=dropout)

        in_features = self.saps_features + self.image_out_features
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        image_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)
        image_features_out = self.image_features_net(image_features_input)
        
        input = torch.cat([saps_out, image_features_out], dim=1)
        out = self.score_net(input)
        return out


# Text and Image features element-wise fusion
class MLPFeatures_averageFusion(nn.Module):
>>>>>>> Stashed changes
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)
        self.saps_out_features = self.saps_features
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)

        self.text_features = len(TEXT_FEATURES_COLUMNS)
<<<<<<< Updated upstream
        self.text_out_features = 32
        self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)

        in_features = self.saps_out_features + self.text_out_features
=======
        self.out_features = 128
        self.text_features_net = MLPVanilla(self.text_features, [], self.out_features, dropout=dropout)

        self.image_features = len(IMAGE_FEATURES_COLUMNS)
        self.image_features_net = MLPVanilla(self.image_features, [], self.out_features, dropout=dropout)
        
        self.fusion_net = MLPVanilla(self.out_features, [], 32, dropout=dropout)

        in_features = self.saps_out_features + 32
>>>>>>> Stashed changes
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        text_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)
        text_features_out = self.text_features_net(text_features_input)

        input = torch.cat([saps_out, text_features_out], dim=1)
        out = self.score_net(input)
        return out
<<<<<<< Updated upstream

=======
       
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

class MLPFeatures_GCN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.log_softmax = LogSoftmax(dim=1)

        self.saps_features = len(SAPS_COLUMNS)
        self.saps_out_features = self.saps_features
        self.saps_net = MLPVanilla(self.saps_features, [], self.saps_features, dropout=dropout)
        
        #self.text_features = len(TEXT_FEATURES_COLUMNS)
        self.node_features = len(TEXT_TOKEN_FEATURES_COLUMNS)

        self.text_features = len(TEXT_TOKEN_FEATURES_COLUMNS)
        self.image_features = len(IMAGE_FEATURES_COLUMNS)
        self.image_out_size = 32
        self.image_net = MLPVanilla(self.image_features, [], self.image_out_size, dropout=dropout)
        self.gcn_hidden_features = 64

        channel_in = 768
        length_in = 512
        kernel_size = 1
        stride = 1
        channel_out = 14
        length_out = int((length_in - kernel_size) / stride + 1)
        self.conv1d = Conv1d(channel_in, channel_out, kernel_size, stride=1)

        self.gcn1 = GraphConvolution(length_out, self.gcn_hidden_features)
        self.gcn2 = GraphConvolution(self.gcn_hidden_features, len(ADJACENCY_MATRIX))

        # forward matrix
        identity_matrix = torch.eye(len(ADJACENCY_MATRIX))
        temp_ADJACENCY_MATRIX = ADJACENCY_MATRIX.add(identity_matrix)
        adj_d = torch.diag_embed(temp_ADJACENCY_MATRIX.sum(dim=1))
        inv_sqrt_adj_d = adj_d.pow(-0.5)
        inv_sqrt_adj_d[torch.isinf(inv_sqrt_adj_d)] = 0
        self.adj_a = inv_sqrt_adj_d.mm(temp_ADJACENCY_MATRIX).mm(inv_sqrt_adj_d)

        in_features = self.saps_out_features + len(ADJACENCY_MATRIX) + self.image_out_size
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input):
        saps_input = input[:, :self.saps_features]
        image_features_input = input[:, self.saps_features: self.saps_features+self.image_features]
        text_features_input = input[:, self.saps_features+self.image_features:]
        #text_features_input = input[:, self.saps_features:]

        saps_out = self.saps_net(saps_input)
        image_out = self.image_net(image_features_input)
        x = text_features_input.view(-1, 512, 768)
        x = x.transpose(1, 2)
        x = self.conv1d(x)

        x = self.relu(self.gcn1(x, self.adj_a))
        x = self.dropout(x)
        x = self.gcn2(x, self.adj_a)
        node_features = self.log_softmax(x)

        node_features = node_features.mean(dim=1)
        input = torch.cat([saps_out, node_features, image_out], dim=1)
        out = self.score_net(input)
        return out
>>>>>>> Stashed changes

def train_test_pycox(data, fold: int = 0):
    data = load_data(top, config)
    train_ds, dev_ds, test_ds = data.split3(fold)

    for ds in (train_ds, dev_ds, test_ds):
        to_pycox(ds)

    train_batch_size = 64  # train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_pycox.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    dropout = 0.5
<<<<<<< Updated upstream
    if config.has_saps and not config.has_labels and not config.has_text_features:
        net = MLPSAPS(dropout)
    elif config.has_saps and config.has_labels and not config.has_text_features:
        net = MLPLabels1(dropout)
    elif config.has_saps and not config.has_labels and config.has_text_features:
        net = MLPFeatures1(dropout)
=======
    if config.saps():
        print('SAPS only')
        net = MLPSAPS(dropout)
    elif config.saps_labels():
        print('SAPS + labels')
        net = MLPLabels2(dropout)
    elif config.saps_text_features():
        print('SAPS + text')
        net = MLPFeatures1(dropout)
    elif config.saps_image_features():
        print('SAPS + image')
        net = MLPImageFeatures(dropout)
    elif config.saps_text_token_features_gcn():
        print('SAPS + GCN')
        net = MLPFeatures_GCN(dropout)
    elif config.saps_multimodal_features():
        print('SAPS + multimodal')
        net = MLPFeatures_averageFusion(dropout)
>>>>>>> Stashed changes
    else:
        raise KeyError

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.001)
    callbacks = [tt.callbacks.EarlyStopping(file_path=str(top / 'foo.pt'))]
    epochs = 250
    model.fit(train_ds.x_pycox, train_ds.y_pycox, train_batch_size, epochs, callbacks,
              shuffle=True,
              val_data=tt.tuplefy(dev_ds.x_pycox, dev_ds.y_pycox),
              val_batch_size=dev_batch_size,
              verbose=config.verbose)
<<<<<<< Updated upstream
    model.compute_baseline_hazards()
    # y = model.predict_cumulative_hazards(test_ds.x_pycox)
    # print(y)
=======
              
    model.compute_baseline_hazards()
    #model.compute_baseline_hazards(batch_size=600)
    surv = model.predict_surv_df(test_ds.x_pycox)
>>>>>>> Stashed changes

    surv = model.predict_surv_df(test_ds.x_pycox)
    # print(surv)
    ev = EvalSurv(surv, test_ds.tte, test_ds.event, censor_surv='km')
    c = ev.concordance_td()
    print(c)
    return c


if __name__ == '__main__':
    top = Path.home() / 'Data/MIMIC-CXR'

    print('SAPS', len(SAPS_COLUMNS))
    print('Labels', len(LABLES_COLUMNS))

    config = Config()
    config.tte_int = True
    config.has_saps = True
    config.has_labels = False
    config.has_text_features = True

    # cindex = train_test_pycox()
    # print('cindex: %.4f' % cindex)

    config.verbose = False
    for i in range(5):
        cindex = train_test_pycox(fold=i)
        print('%.4f' % cindex, end='\t')
