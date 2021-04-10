from typing import Tuple

import torch
from pycox.models import CoxPH
import collections
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, CoxTime
from sklearn.model_selection import KFold, train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored
import pandas as pd
from torch import nn

np.random.seed(1234)
random.seed(1234)


class Data:
    saps_columns = ['sapsii']
    chexbert_columns = 'Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,' \
                       'Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,' \
                       'Pneumothorax,Support Devices'.split(',')

    def __init__(self, name):
        self.name = name
        self.df = None
        self.x_pycox = None
        self.y_pycox = None
        self.tte = None
        self.event = None

    def to_pycox(self):
        columns = self.saps_columns + self.chexbert_columns
        self.x_pycox = self.df[columns].values.astype('float32')
        self.tte = self.df['time-to-event'].to_numpy(dtype=float)
        self.event = self.df['event'].to_numpy(dtype=bool)
        self.y_pycox = (self.tte, self.event)

    def to_sksurv(self):
        columns = self.saps_columns + self.chexbert_columns
        self.x_sksurv = self.df[columns].values.astype('float32')
        self.tte = self.df['time-to-event'].to_numpy(dtype=float)
        self.event = self.df['event'].to_numpy(dtype=bool)
        self.y_sksurv = self.df[['event', 'time-to-event']].to_records(index=False)

    def _split(self, indices, name):
        subdata = Data(name)
        subdata.df = self.df[self.df['fold'].isin(indices)]
        return subdata

    def split(self, fold) -> Tuple['Data', 'Data', 'Data']:
        folds = [
            ({3, 4, 5, 6, 7, 8, 9}, {2}, {0, 1}),
            ({5, 6, 7, 8, 9, 0, 1}, {4}, {2, 3}),
            ({7, 8, 9, 0, 1, 2, 3}, {6}, {4, 5}),
            ({9, 0, 1, 2, 3, 4, 5}, {8}, {6, 7}),
            ({1, 2, 3, 4, 5, 6, 7}, {0}, {8, 9}),
        ]
        return self._split(folds[fold][0], 'train'), \
               self._split(folds[fold][1], 'dev'), \
               self._split(folds[fold][2], 'test')

    def describe(self):
        print(self.name)
        print('Len', self.df.shape)
        # print('X pycox shape', self.x_pycox.shape)
        # print('Y pycox shape', len(self.y_pycox))
        # print(self.df.head(5))


def load_data() -> Data:
    tte_df = pd.read_csv(top / 'tte.csv', dtype={'study_id': int, 'subject_id': int})
    study_ids = set(tte_df['study_id'])
    chexpert_df = pd.read_csv(top / 'mimic-cxr-2.0.0-chexpert.csv', dtype={'study_id':int, 'subject_id':int})
    # chexpert_df = pd.read_csv(top / 'mimic-cxr-2.0.0-negbio.csv', dtype={'study_id':int, 'subject_id':int})
    chexpert_df = chexpert_df[chexpert_df['study_id'].isin(study_ids)]
    chexpert_df = chexpert_df.set_index(['study_id']).reindex(tte_df['study_id']).reset_index()

    for i, row in tte_df.iterrows():
        assert row['study_id'] == chexpert_df.iloc[i]['study_id']

    df = pd.concat([tte_df, chexpert_df[Data.chexbert_columns]], axis=1)
    # for c in df.columns:
    #     print(c, df.iloc[0][c])
    df = df.fillna(-2)

    data = Data('whole')
    data.df = df
    data.tte_df = df
    return data


class MLPCox(nn.Module):
    def __init__(self, drop_prob=0.5):
        super().__init__()
        in_features = len(Data.chexbert_columns)
        out_features = in_features
        net = [
            nn.Linear(in_features, 32, bias=True),
            nn.BatchNorm1d(32),
            nn.Dropout(drop_prob),
            nn.Linear(32, out_features, bias=True),
            nn.Dropout(drop_prob),
            # nn.ReLU()
        ]
        self.label_net = nn.Sequential(*net)

        net = [
            nn.Linear(out_features + 1, 1, bias=False),
            # nn.ReLU()
        ]
        self.score_net = nn.Sequential(*net)

    def forward(self, input):
        label_input = input[:, 1:]
        # print(label_input.shape)
        label_out = self.label_net(label_input)
        # print(label_out.shape)

        score_input = input[:, 0:1]
        # print(score_input.shape)

        input = torch.cat([score_input, label_out], dim=1)
        # print(input.shape)
        return self.score_net(input)


def get_model():
    # out_features = 1
    # num_nodes = [16]
    # batch_norm = True
    dropout = 0.5
    # output_bias = False
    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
    net = MLPCox(drop_prob=dropout)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.005)
    return model


def train_test_pycox(fold: int = 0):
    data = load_data()
    train_ds, dev_ds, test_ds = data.split(fold)

    for ds in (train_ds, dev_ds, test_ds):
        ds.to_pycox()
        ds.describe()

    train_batch_size = train_ds.x_pycox.shape[0]
    dev_batch_size = dev_ds.x_pycox.shape[0]
    # in_features = train_ds.x_pycox.shape[1]

    model = get_model()
    callbacks = [tt.callbacks.EarlyStopping()]
    epochs = 250
    model.fit(train_ds.x_pycox, train_ds.y_pycox, train_batch_size, epochs, callbacks,
              shuffle=True,
              val_data=tt.tuplefy(dev_ds.x_pycox, dev_ds.y_pycox),
              val_batch_size=dev_batch_size,
              verbose=True)
    model.compute_baseline_hazards()
    # y = model.predict_cumulative_hazards(test_ds.x_pycox)
    # print(y)

    surv = model.predict_surv_df(test_ds.x_pycox)
    # print(surv)
    ev = EvalSurv(surv, test_ds.tte, test_ds.event, censor_surv='km')
    c = ev.concordance_td()
    print('c-index', c)


def train_test_sksurv(fold: int = 0):
    data = load_data()
    train_ds, dev_ds, test_ds = data.split(fold)

    for ds in (train_ds, dev_ds, test_ds):
        ds.to_sksurv()
        ds.describe()

    model = CoxPHSurvivalAnalysis()
    model.fit(train_ds.x_sksurv, train_ds.y_sksurv)

    prob = model.predict(test_ds.x_sksurv)
    c, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        test_ds.event, test_ds.tte, prob)
    print('c-index', c)


if __name__ == '__main__':
    top = Path.home() / r'Data/MIMIC-CXR'
    # train_test_sksurv()
    train_test_pycox()
