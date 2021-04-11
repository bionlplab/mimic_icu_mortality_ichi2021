import collections
import io
import re
import zipfile
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import dateparser
import pandas as pd
import tqdm
import numpy as np
from sklearn.model_selection import KFold


class Visit:
    def __init__(self):
        self.studies = {}
        self.hadm_id = None
        self.admittime = None
        self.dischtime = None
        self.deathtime = None
        self.marital_status = None
        self.ethnicity = None

    def to_dict(self):
        return {
            'hadm_id': self.hadm_id,
            'admittime': str(self.admittime),
            'dischtime': str(self.dischtime),
            'deathtime': str(self.deathtime) if self.deathtime is not None else None,
            'marital_status': None if pd.isna(self.marital_status) else str(self.marital_status),
            'ethnicity': self.ethnicity,
            'studies': [s.to_dict() for s in sorted(self.studies.values(), key=lambda x: x.StudyDate)]
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


class Patient:
    def __init__(self):
        self.subject_id = None
        self.visits = []  # type: List[Visit]
        self.gender = None
        self.anchor_age = None
        self.anchor_year = None
        self.anchor_year_group = None
        self.dod = None

    def to_dict(self):
        return {
            'subject_id': self.subject_id,
            'gender': self.gender,
            'anchor_age': self.anchor_age,
            'anchor_year': self.anchor_year,
            'visits': [s.to_dict() for s in sorted(self.visits, key=lambda x: x.admittime)]
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


class Study:
    def __init__(self):
        self.study_id = None
        self.report_path = None
        self.StudyDate = None
        self.StudyTime = None
        self.records = {}  # type: Dict[str, Study]

    def to_dict(self):
        return {
            'study_id': self.study_id,
            'StudyDate': str(self.StudyDate),
            'StudyTime': self.StudyTime,
            'report_path': self.report_path,
            'records': [r.to_dict() for r in sorted(self.records.values(), key=lambda x: x.StudyTime)]
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


class Record:
    def __init__(self):
        self.dicom_id = None
        self.path = None
        self.StudyDate = None
        self.StudyTime = None
        self.ViewPosition = None

    def to_dict(self):
        return {
            'dicom_id': self.dicom_id,
            'dicom_path': self.path,
            'StudyDate': str(self.StudyDate),
            'StudyTime': self.StudyTime,
            'ViewPosition': None if pd.isna(self.ViewPosition) else self.ViewPosition
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

dateformat1 = '%Y-%m-%d %H:%M:%S'
dateformat2 = '%Y%m%d'


def create():
    cnt = collections.Counter()
    patients_df = pd.read_csv(top / 'patients.csv')
    admissions_df = pd.read_csv(top / 'admissions.csv')
    metadata_df = pd.read_csv(top / 'mimic-cxr-2.0.0-metadata.csv', dtype={'StudyDate': str})
    study_df = pd.read_csv(top / 'cxr-study-list.csv')
    record_df = pd.read_csv(top / 'cxr-record-list.csv')

    subject_ids = set(study_df['subject_id'])
    if False:
        subject_ids = {17039362}
        record_df = record_df[record_df['subject_id'].isin(subject_ids)]
        study_df = study_df[study_df['subject_id'].isin(subject_ids)]
        metadata_df = metadata_df[metadata_df['subject_id'].isin(subject_ids)]
    patients_df = patients_df[patients_df['subject_id'].isin(subject_ids)]
    admissions_df = admissions_df[admissions_df['subject_id'].isin(subject_ids)]

    patients = {}  # type: Dict[int, Patient]
    for subject_id in subject_ids:
        p = Patient()
        p.subject_id = subject_id
        patients[subject_id] = p

    # add demographic
    for i, row in tqdm.tqdm(patients_df.iterrows(), total=len(patients_df), desc='Add demo'):
        subject_id = row['subject_id']
        p = patients[subject_id]
        p.gender = row['gender']
        p.anchor_age = row['anchor_age']
        p.anchor_year = row['anchor_year']
        p.anchor_year_group = row['anchor_year_group']
        p.dod = row['dod']

    # add admission
    for i, row in tqdm.tqdm(admissions_df.iterrows(), total=len(admissions_df), desc='Add admission'):
        subject_id = row['subject_id']
        p = patients[subject_id]
        v = Visit()
        v.hadm_id = row['hadm_id']
        v.admittime = datetime.strptime(row['admittime'], dateformat1)
        v.dischtime = datetime.strptime(row['dischtime'], dateformat1)
        v.deathtime = None if pd.isna(row['deathtime']) else datetime.strptime(row['deathtime'], dateformat1)
        v.marital_status = row['marital_status']
        v.ethnicity = row['ethnicity']
        p.visits.append(v)

    # collect studies
    studies = {}  # type: Dict[Tuple[str, str], Study]
    for i, row in tqdm.tqdm(study_df.iterrows(), total=len(study_df), desc='Add study'):
        subject_id = row['subject_id']
        s = Study()
        s.study_id = row['study_id']
        s.report_path = row['path']
        studies[subject_id, s.study_id] = s

    # add dicom
    for i, row in tqdm.tqdm(record_df.iterrows(), total=len(record_df), desc='Add record'):
        subject_id = row['subject_id']
        study_id = row['study_id']
        s = studies[subject_id, study_id]
        r = Record()
        r.dicom_id = row['dicom_id']
        r.path = row['path']
        s.records[r.dicom_id] = r

    # add metadata
    dateformat = '%Y%m%d'
    # metadata_df['StudyTime'] = pd.to_datetime(metadata_df['StudyTime'], unit='s')
    metadata_df['StudyDate'] = pd.to_datetime(metadata_df['StudyDate'], format='%Y%m%d')
    for i, row in tqdm.tqdm(metadata_df.iterrows(), total=len(metadata_df), desc='Add metadata'):
        subject_id = row['subject_id']
        study_id = row['study_id']
        studydate = row['StudyDate']
        # studytime = datetime.utcfromtimestamp((row['StudyTime'] - 25569) * 86400.0)

        s = studies[subject_id, study_id]
        s.StudyDate = studydate
        s.StudyTime = row['StudyTime']

        dicom_id = row['dicom_id']
        if dicom_id not in s.records:
            continue
        d = s.records[dicom_id]
        d.StudyDate = studydate
        d.StudyTime = row['StudyTime']
        d.ViewPosition = row['ViewPosition']

    # add study to patient visit
    for (subject_id, study_id), study in studies.items():
        p = patients[subject_id]

        # print(p.subject_id, len(p.visits))

        found = False
        for visit in p.visits:
            # print(visit.admittime, study.StudyDate, visit.dischtime, study.StudyTime,
            #       visit.admittime.date() <= study.StudyDate.date() <= visit.dischtime.date())
            if visit.admittime.date() <= study.StudyDate.date() <= visit.dischtime.date():
                visit.studies[study_id] = study
                found = True
                break

        if not found:
            cnt['Cannot find visit'] += 1
            # print(p)
            # exit(1)

    objs = []
    for p in patients.values():
        p.visits = [v for v in p.visits if len(v.studies) != 0]
        if len(p.visits) > 0:
            obj = p.to_dict()
            obj.update(obj['visits'][-1])
            del obj['visits']
            objs.append(obj)
        else:
            cnt['Patient has no visit'] += 1

    # for p in new_patients:
    #     try:
    #         json.dumps(p.to_dict())
    #     except:
    #         print(p.to_dict())

    with open(top / 'last_visit.json', 'w') as fp:
        json.dump(objs, fp, indent=2)

    print(cnt)


def analyze():
    with open(top / 'last_visit_sapsii.json', 'r') as fp:
        patients = json.load(fp)

    cnt = collections.Counter()
    durations = []
    for p in tqdm.tqdm(patients):
        admittime = datetime.strptime(p['admittime'], dateformat1)
        dischtime = datetime.strptime(p['dischtime'], dateformat1)
        if pd.isna(p['deathtime']):
            cnt['not dead'] += 1
            durations.append((dischtime - admittime).days)

        else:
            deathtime = datetime.strptime(p['deathtime'], dateformat1)
            cnt['dead'] += 1
            durations.append((deathtime - admittime).days)
            if dischtime.date() > dischtime.date():
                cnt['dead after icu'] += 1
        cnt['study'] += len(p['studies'])

    print(cnt)
    s = cnt['dead'] + cnt['not dead']
    print(s)
    print(cnt['dead'] / s)

    cnt = collections.Counter()
    for d in durations:
        cnt[d] += 1
    print(cnt)
    print(np.average(durations))


def add_sapsii():
    with open(top / 'last_visit.json', 'r') as fp:
        patients = json.load(fp)
    subject_ids = set(p['subject_id'] for p in patients)
    hadm_ids = set(p['hadm_id'] for p in patients)
    print('Total', len(subject_ids))

    df = pd.read_csv(top / 'sapsii.csv')
    df['starttime'] = pd.to_datetime(df['starttime'])
    saps_subject_ids = set(df['subject_id'])
    print('SAPS', len(saps_subject_ids))
    print('Combined', len(subject_ids & saps_subject_ids))

    saps_map = {}  # type: Dict[Any, List]
    for i, r in df.iterrows():
        if r['hadm_id'] not in hadm_ids:
            continue
        k = r['subject_id'], r['hadm_id']
        if k not in saps_map:
            saps_map[k] = []
        saps_map[k].append(r)
    for k, v in saps_map.items():
        saps_map[k] = sorted(v, key=lambda a: a['starttime'])

    cnt = collections.Counter()
    new_patients = []
    new_rows = []
    for p in tqdm.tqdm(patients):
        k = p['subject_id'], p['hadm_id']
        if k in saps_map:
            study_date = dateparser.parse(p['studies'][0]['StudyDate'])
            admittime = dateparser.parse(p['admittime'])
            dischtime = dateparser.parse(p['dischtime'])
            found = False
            if len(saps_map[k]) == 1:
                row = saps_map[k][0]
                if study_date > row['starttime']:
                    cnt['cxr > sapsii'] += 1
                    continue
                found = True
            else:
                row = None
                for r in saps_map[k]:
                    if admittime <= r['starttime'] < dischtime:
                        if study_date > r['starttime']:
                            cnt['cxr > sapsii'] += 1
                        else:
                            row = r
                            found = True
                            break
            if found:
                p['sapsii'] = row['sapsii']
                p['sapsii_prob'] = row['sapsii_prob']
                p['stay_id'] = row['stay_id']
                new_patients.append(p)
                row['study_id'] = p['studies'][0]['study_id']
                new_rows.append(row)
            else:
                cnt['No sapsii < 24 h'] += 1
                # print(saps_map[k])
                # exit(1)
        else:
            cnt['No sapsii'] += 1
    with open(top / 'last_visit_sapsii.json', 'w') as fp:
        json.dump(new_patients, fp, indent=2)

    df = pd.DataFrame(new_rows)
    df.to_csv(top / 'last_visit_sapsii.csv', index=False)

    print(cnt)


def get_hours(delta: timedelta):
    return delta.days * 24 + delta.seconds / 3600


def create_csv():
    with open(top / 'last_visit_sapsii.json', 'r') as fp:
        patients = json.load(fp)

    # subject, event, time-to-event, fold, sapsii, sapsii-prob
    rows = []
    for p in tqdm.tqdm(patients):
        admittime = dateparser.parse(p['admittime'])
        if p['deathtime'] is None:
            event = False
            dischtime = dateparser.parse(p['dischtime'])
            # print(p['subject_id'], dischtime - admittime)
            tte = get_hours(dischtime - admittime)
            # if p['hadm_id'] == 26884919:
            #     print(admittime)
            #     print(dischtime)
            #     print(get_hours(dischtime - admittime))
            #     exit(1)
        else:
            event = True
            deathtime = dateparser.parse(p['deathtime'])
            tte = get_hours(deathtime - admittime)
        if tte <= 0:
            continue
        rows.append({
            'subject_id': p['subject_id'],
            'hadm_id': p['hadm_id'],
            'study_id': p['studies'][0]['study_id'],
            'event': event,
            'time-to-event': tte,
            # 'sapsii': p['sapsii'],
            # 'sapsii_prob': p['sapsii_prob']
        })
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kf.split(rows)):
        for idx in test:
            rows[idx]['fold'] = i
    df = pd.DataFrame(rows)
    df.to_csv(top / 'tte.csv', index=False)


def chop_strict(text: str, id = None):
    text = text.replace('&#x20;', '')
    m = re.search('FINDINGS:|Findings.\n|FINDINGS {2,}|'
                  'Findings/impression:|Findings/impression\n|Findings/pression:|'
                  'Findings/compression:|Findings/pressure:|'
                  'impression:|IMPRESSION :|Impression\n\n|Pression:\n', text, re.I)
    if m:
        text = text[m.start():]
        return text
    m = re.search('REASON FOR EXAMINATION:|REASON FOR EXAM:|HISTORY|COMPARISONS?:|'
                  'PORTABLE FRONTAL CHEST RADIOGRAPH:|INDICATION:|!! WET READ !!', text, re.I)
    if m:
        text = text[m.start():]
        return text
    raise KeyError('Cannot find section')


def get_report():
    excepts = {24315948, 24076864, 27270549, 21266774, 24154811}

    with open(top / 'last_visit_sapsii.json', 'r') as fp:
        patients = json.load(fp)

    archive = zipfile.ZipFile(top / 'mimic-cxr-reports.zip', 'r')

    cnt = collections.Counter()
    rows = []
    for p in tqdm.tqdm(patients):
        report_path = p['studies'][0]['report_path']
        rf = archive.open(report_path)
        f = io.TextIOWrapper(rf, encoding='utf8', newline='')
        text = f.read()
        rf.close()

        text = text.strip()
        try:
            text = chop_strict(text)
        except:
            m = re.search('REASON FOR EXAMINATION:|REASON FOR EXAM:|HISTORY|COMPARISONS?:|'
                          'PORTABLE FRONTAL CHEST RADIOGRAPH:|INDICATION:|!! WET READ !!', text, re.I)
            if not m and not p['hadm_id'] in excepts:
                print('No findings/impression:', p['hadm_id'])
                print(text)
                cnt['No findings'] += 1
                exit(1)
        rows.append({
            'subject_id': p['subject_id'],
            'hadm_id': p['hadm_id'],
            'study_id': p['studies'][0]['study_id'],
            'report_path': report_path,
            'text': text
        })
    archive.close()
    df = pd.DataFrame(rows)
    df.to_csv(top / 'reports.csv', index=False)
    print(cnt)


def get_image():
    with open(top / 'last_visit_sapsii.json', 'r') as fp:
        patients = json.load(fp)

    cnt = collections.Counter()
    rows = []
    for p in tqdm.tqdm(patients):
        image_path = p['studies'][0]['records'][0]['dicom_path']
        rows.append({
            'subject_id': p['subject_id'],
            'hadm_id': p['hadm_id'],
            'study_id': p['studies'][0]['study_id'],
            'image_path': image_path,
        })
    df = pd.DataFrame(rows)
    df.to_csv(top / 'images.csv', index=False)
    print(cnt)


if __name__ == '__main__':
    top = Path(r'/Users/yip4002/Data/MIMIC-CXR')
    # create()
    # add_sapsii()
    # analyze()
    create_csv()
    # get_report()
    # get_image()



