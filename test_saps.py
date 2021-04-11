from pathlib import Path

import pandas as pd
from sksurv.metrics import concordance_index_censored

from utils import FOLDS


if __name__ == '__main__':
    top = Path.home() / 'Data/MIMIC-CXR'
    tte_df = pd.read_csv(top / 'tte.csv', dtype={'study_id': int, 'subject_id': int, 'hadm_id': int})
    tte_df['time-to-event'] = tte_df['time-to-event'].astype(int)

    sapsii_df = pd.read_csv(top / 'last_visit_sapsii.csv', dtype={'subject_id': int, 'hadm_id': int})
    hadm_ids = set(tte_df['hadm_id'])
    sapsii_df = sapsii_df[sapsii_df['hadm_id'].isin(hadm_ids)]
    sapsii_df = sapsii_df.set_index(['hadm_id']).reindex(tte_df['hadm_id']).reset_index()
    assert len(sapsii_df) == len(tte_df)
    for i, row in tte_df.iterrows():
        assert row['hadm_id'] == sapsii_df.iloc[i]['hadm_id']
    df = pd.concat([tte_df, sapsii_df['sapsii']], axis=1)

    for i in range(5):
        test_df = df[df['fold'].isin(FOLDS[i][2])]

        cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            test_df['event'], test_df['time-to-event'], test_df['sapsii'])

        print('%.4f' % cindex, end='\t')