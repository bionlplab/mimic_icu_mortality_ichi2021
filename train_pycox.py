import random
from pathlib import Path

import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from torch import nn
from torchtuples.practical import MLPVanilla

from utils import Data, load_data, Config, SAPS_COLUMNS, LABLES_COLUMNS, TEXT_FEATURES_COLUMNS

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


class MLPSAPS(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS)
        self.saps_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

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


def train_test_pycox(fold: int = 0):
    data = load_data(top, config)
    train_ds, dev_ds, test_ds = data.split3(fold)

    for ds in (train_ds, dev_ds, test_ds):
        to_pycox(ds)
        # ds.describe()

    train_batch_size = 64  # train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_pycox.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    dropout = 0.5
    if config.has_saps and not config.has_labels and not config.has_text_features:
        net = MLPSAPS(dropout)
    elif config.has_saps and config.has_labels and not config.has_text_features:
        net = MLPLabels1(dropout)
    elif config.has_saps and not config.has_labels and config.has_text_features:
        net = MLPFeatures1(dropout)
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
    config.has_text_features = True

    # cindex = train_test_pycox()
    # print('cindex: %.4f' % cindex)

    config.verbose = False
    for i in range(5):
        cindex = train_test_pycox(fold=i)
        print('%.4f' % cindex, end='\t')
