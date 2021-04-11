from typing import Tuple

import numpy as np
import pandas as pd
import tqdm

FOLDS = [
    ({3, 4, 5, 6, 7, 8, 9}, {2}, {0, 1}),
    ({5, 6, 7, 8, 9, 0, 1}, {4}, {2, 3}),
    ({7, 8, 9, 0, 1, 2, 3}, {6}, {4, 5}),
    ({9, 0, 1, 2, 3, 4, 5}, {8}, {6, 7}),
    ({1, 2, 3, 4, 5, 6, 7}, {0}, {8, 9}),
]

sapsii_columns = ['sapsii']
SAPS_COLUMNS = 'age_score,hr_score,sysbp_score,temp_score,gcs_score,PaO2FiO2_score,bun_score,uo_score,sodium_score,' \
               'potassium_score,bicarbonate_score,bilirubin_score,wbc_score,comorbidity_score,' \
               'admissiontype_score'.split(',')
LABLES_COLUMNS = 'Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,' \
                 'Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,' \
                 'Pneumothorax,Support Devices'.split(',')
TEXT_FEATURES_COLUMNS = [f'text_feature_{i}' for i in range(768)]


class Config:
    def __init__(self):
        self.has_saps = False
        self.has_labels = False
        self.has_text_features = False
        self.verbose = True
        self.tte_int = False


class Data:
    def __init__(self, name, config):
        self.name = name
        self.df = None
        self.tte = None
        self.event = None
        self.config = config
        self.x_sksurv = None
        self.y_sksurv = None
        self.x_cctime = None
        self.y_cctime = None
        self.x_pycox = None
        self.y_pycox = None
        self.x_deephit = None
        self.y_deephit = None

    def _split(self, indices, name):
        subdata = Data(name, self.config)
        subdata.df = self.df[self.df['fold'].isin(indices)]
        return subdata

    def split3(self, fold) -> Tuple['Data', 'Data', 'Data']:
        return self._split(FOLDS[fold][0], 'train'), \
               self._split(FOLDS[fold][1], 'dev'), \
               self._split(FOLDS[fold][2], 'test')

    def split2(self, fold):
        return self._split(FOLDS[fold][0] | FOLDS[fold][1], 'train'), \
               self._split(FOLDS[fold][2], 'test')

    def describe(self):
        print(self.name)
        print('Len', self.df.shape)
        # print(self.df.head(5))


def load_data(top, config) -> Data:
    tte_df = pd.read_csv(top / 'tte.csv', dtype={'study_id': int, 'subject_id': int, 'hadm_id': int})
    if config.tte_int:
        tte_df['time-to-event'] = tte_df['time-to-event'].astype(int)

    df = tte_df
    if config.has_saps:
        sapsii_df = pd.read_csv(top / 'last_visit_sapsii.csv', dtype={'subject_id': int, 'hadm_id': int})
        sapsii_df = sapsii_df.fillna(0)
        hadm_ids = set(tte_df['hadm_id'])
        sapsii_df = sapsii_df[sapsii_df['hadm_id'].isin(hadm_ids)]
        sapsii_df = sapsii_df.set_index(['hadm_id']).reindex(tte_df['hadm_id']).reset_index()
        assert len(sapsii_df) == len(tte_df)
        for i, row in tte_df.iterrows():
            assert row['hadm_id'] == sapsii_df.iloc[i]['hadm_id']
        df = pd.concat([df, sapsii_df[SAPS_COLUMNS]], axis=1)

    if config.has_labels:
        labels_df = pd.read_csv(top / 'mimic-cxr-2.0.0-chexpert.csv', dtype={'study_id': int, 'subject_id': int})
        # labels_df = pd.read_csv(top / 'mimic-cxr-2.0.0-negbio.csv', dtype={'study_id':int, 'subject_id':int})
        labels_df = labels_df.replace(-1, 0.5)
        labels_df = labels_df.replace(0, -1)
        labels_df = labels_df.fillna(0)

        study_ids = set(tte_df['study_id'])
        labels_df = labels_df[labels_df['study_id'].isin(study_ids)]
        labels_df = labels_df.set_index(['study_id']).reindex(tte_df['study_id']).reset_index()
        assert len(labels_df) == len(tte_df)
        for i, row in tte_df.iterrows():
            assert row['study_id'] == labels_df.iloc[i]['study_id']
        df = pd.concat([df, labels_df[LABLES_COLUMNS]], axis=1)

    if config.has_text_features:
        # features
        with np.load(str(top / 'chexbert_pooling_features.npz')) as data:
            rows = [data[str(id)][0] for id in tqdm.tqdm(tte_df['study_id'],disable=True)]
        text_features_df = pd.DataFrame(rows, columns=TEXT_FEATURES_COLUMNS)
        assert len(text_features_df) == len(tte_df)
        df = pd.concat([df, text_features_df], axis=1)

    data = Data('whole', config)
    data.df = df
    return data
