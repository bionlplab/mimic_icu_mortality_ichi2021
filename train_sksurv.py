import random
from pathlib import Path

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from utils import Config, Data, load_data, LABLES_COLUMNS, SAPS_COLUMNS

np.random.seed(1234)
random.seed(1234)


def to_sksurv(data):
    columns = []
    if data.config.has_saps:
        columns += SAPS_COLUMNS
    if data.config.has_labels:
        columns += LABLES_COLUMNS
    # if data.config.has_text_features:
    #     columns += data.text_feature_columns
    data.tte = data.df['time-to-event'].to_numpy(dtype=float)
    data.event = data.df['event'].to_numpy(dtype=bool)
    data.x_sksurv = data.df[columns].values.astype('float32')
    data.y_sksurv = data.df[['event', 'time-to-event']].to_records(index=False)


def train_test_sksurv(fold: int = 0):
    data = load_data(top, config)
    train_ds, test_ds = data.split2(fold)

    for ds in (train_ds, test_ds):
        to_sksurv(ds)
        # ds.describe()
        # print('X sksurv shape', ds.x_sksurv.shape)
        # print('Y sksruv shape', ds.y_sksurv.shape)

    model = CoxPHSurvivalAnalysis(verbose=config.verbose)
    model.fit(train_ds.x_sksurv, train_ds.y_sksurv)

    prob = model.predict(test_ds.x_sksurv)
    c, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        test_ds.event, test_ds.tte, prob)
    return c


if __name__ == '__main__':
    top = Path.home() / r'Data/MIMIC-CXR'

    print('SAPS', len(SAPS_COLUMNS))
    print('Labels', len(LABLES_COLUMNS))

    config = Config()
    config.tte_int = True
    config.has_saps = True
    config.has_labels = True

    # cindex = train_test_sksurv()
    # print('cindex: %.4f' % cindex)

    config.verbose = False
    for i in range(5):
        cindex = train_test_sksurv(fold=i)
        print('%.4f' % cindex, end='\t')
