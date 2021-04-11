import random
from pathlib import Path

import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from torch import nn
from torchtuples.practical import MLPVanilla

from utils import Config, load_data, Data, LABLES_COLUMNS, SAPS_COLUMNS, TEXT_FEATURES_COLUMNS

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(1234)
random.seed(1234)


def to_cctime(train_ds: Data, dev_ds: Data, test_ds: Data):
    columns = []
    if train_ds.config.has_saps:
        columns += SAPS_COLUMNS
    if train_ds.config.has_labels:
        columns += LABLES_COLUMNS
    if train_ds.config.has_text_features:
        columns += TEXT_FEATURES_COLUMNS

    for data in [train_ds, dev_ds, test_ds]:
        data.tte = data.df['time-to-event'].to_numpy(dtype=float)
        data.event = data.df['event'].to_numpy(dtype=bool)
        data.x_cctime = data.df[columns].values.astype('float32')
        if config.verbose:
            print('Len x cctime shape', data.x_cctime.shape)

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df['time-to-event'].values, df['event'].values)

    train_ds.y_cctime = labtrans.fit_transform(*get_target(train_ds.df))
    dev_ds.y_cctime = labtrans.transform(*get_target(dev_ds.df))
    test_ds.y_cctime = labtrans.transform(*get_target(test_ds.df))

    if config.verbose:
        print('Val x cctime', dev_ds.x_cctime.shape)
        print('Val y cctime', dev_ds.y_cctime[0].shape, dev_ds.y_cctime[1].shape)
        print('Val shape', tt.tuplefy(dev_ds.x_cctime, dev_ds.y_cctime).shapes())
    return labtrans


class MLPSAPSCoxTime(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS) + 1
        self.saps_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input, time):
        # print('Input shape', input.shape)
        # print('Time shape', time.shape)
        input = torch.cat([input, time], dim=1)
        return self.saps_net(input)


class MLPLabelsCoxTime(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        in_features = len(SAPS_COLUMNS) + len(LABLES_COLUMNS) + 1
        self.score_net = MLPVanilla(in_features, [in_features], 1, dropout=dropout)

    def forward(self, input, time):
        input = torch.cat([input, time], dim=1)
        return self.score_net(input)


class MLPFeaturesCoxTime(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.saps_features = len(SAPS_COLUMNS)

        self.text_features = len(TEXT_FEATURES_COLUMNS)
        self.text_out_features = 14
        self.text_features_net = MLPVanilla(self.text_features, [], self.text_out_features, dropout=dropout)

        in_features = self.saps_features + self.text_out_features + 1
        self.score_net = MLPVanilla(in_features, [], 1, dropout=dropout)

    def forward(self, input, time):
        saps_input = input[:, :self.saps_features]
        text_features_input = input[:, self.saps_features:]

        text_features_out = self.text_features_net(text_features_input)

        input = torch.cat([saps_input, text_features_out, time], dim=1)
        out = self.score_net(input)
        return out


def train_test_cctime(fold: int = 0):
    data = load_data(top, config)
    train_ds, dev_ds, test_ds = data.split3(fold)

    labtrans = to_cctime(train_ds, dev_ds, test_ds)

    train_batch_size = 64  # train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_cctime.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    dropout = 0.5
    if config.has_saps and not config.has_labels and not config.has_text_features:
        net = MLPSAPSCoxTime(dropout)
    elif config.has_saps and config.has_labels and not config.has_text_features:
        net = MLPLabelsCoxTime(dropout)
    elif config.has_saps and not config.has_labels and config.has_text_features:
        net = MLPFeaturesCoxTime(dropout)
    else:
        raise KeyError

    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    # lrfinder = model.lr_finder(train_ds.x_cctime, train_ds.y_cctime, train_batch_size, tolerance=2)
    # print(lrfinder.get_best_lr())
    # print(lrfinder.to_pandas())
    # exit(1)

    model.optimizer.set_lr(0.001)
    callbacks = [tt.callbacks.EarlyStopping(file_path=str(top / 'foo.pt'))]

    epochs = 250
    model.fit(train_ds.x_cctime, train_ds.y_cctime, train_batch_size, epochs, callbacks,
              shuffle=True,
              val_data=tt.tuplefy(dev_ds.x_cctime, dev_ds.y_cctime),
              val_batch_size=dev_batch_size,
              verbose=config.verbose)
    model.compute_baseline_hazards()
    # print(y)

    surv = model.predict_surv_df(test_ds.x_cctime)
    # print(surv)
    ev = EvalSurv(surv, test_ds.tte, test_ds.event, censor_surv='km')
    c = ev.concordance_td()
    return c


if __name__ == '__main__':
    top = Path.home() / r'Data/MIMIC-CXR'

    print('SAPS', len(SAPS_COLUMNS))
    print('Labels', len(LABLES_COLUMNS))

    config = Config()
    config.tte_int = True
    config.has_saps = True
    config.has_labels = False
    config.has_text_features = True

    # cindex = train_test_cctime()
    # print('cindex: %.4f' % cindex)

    config.verbose = False
    for i in range(5):
        cindex = train_test_cctime(fold=i)
        print('%.4f' % cindex, end='\t')
