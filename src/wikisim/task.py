"""Information about task data."""

import os
import glob
import re
import json
from importlib import resources
import pandas as pd


def get_subjects(subset='wikisim'):
    """Get a list of included subject IDs."""
    with resources.open_text('bender.resources', 'subjects.json') as f:
        subj_subsets = json.load(f)

    if subset not in subj_subsets:
        raise ValueError(f'Subset of subjects not found: {subset}')

    subj_ids = [f'bender_{n:02d}' for n in subj_subsets[subset]]
    return subj_ids


def read_items(items_file):
    """Read items from a text file."""
    with open(items_file, 'r') as f:
        items = [line.rstrip() for line in f]
    return items


def read_text(items, text_dir):
    """Read text for a set of items."""
    text = []
    for item in items:
        file_name = item.replace(' ', '_') + '.txt'
        wiki_file = os.path.join(text_dir, file_name)
        if not os.path.exists(wiki_file):
            raise IOError(f'No text found for {item}.')

        with open(wiki_file, 'r') as f:
            item_text = f.read()
        text.append(item_text)
    return text


def read_images(pool, im_dir):
    """Read image files for a set of items."""
    import matplotlib.pyplot as plt
    images = {}
    for i, item in pool.iterrows():
        # file names should have underscores instead of spaces
        file_base = item.stim.replace(' ', '_')
        im_base = os.path.join(im_dir, item.subcategory, file_base)

        # search for a matching image file for this item
        res = [f for f in glob.glob(im_base + '.*')
               if re.search('\w+\.(png|jpg)', f)]
        if not res:
            raise IOError(f'No file found matching: {im_base}')
        elif len(res) > 1:
            raise IOError(f'Multiple matches for: {im_base}')
        im_file = res[0]

        # read the image
        image = plt.imread(im_file)
        images[item.stim] = image
    return images


def read_image_sets(pool, im_dir):
    """Return images grouped by category."""
    images = read_images(pool, im_dir)
    image_sets = {}
    for category, df in pool.groupby('category'):
        image_sets[category] = [images[item] for item in df['stim'].to_list()]
    return image_sets


def read_prex_run(subj_dir, run):
    """Read task data for one pre-exposure run."""
    # read the log file
    run_file = os.path.join(subj_dir, 'log_04-{:02d}-01_a_prex.txt'.format(run))
    if not os.path.exists(run_file):
        raise IOError('Log file does not exist: {}'.format(run_file))
    data = pd.read_csv(run_file, sep='\t')
    return data


def read_prex(subj_dir):
    """Read all pre-exposure task data."""
    df_run = []
    for run in range(1, 7):
        dr = read_prex_run(subj_dir, run)
        dr.loc[:, 'run'] = run
        df_run.append(dr)
    df = pd.concat(df_run, ignore_index=True)
    return df


def read_run(subj_dir, period, run):
    """Read log data for one run."""
    match = f'log_??-{run:02d}-??_{period}.txt'
    match_files = glob.glob(os.path.join(subj_dir, match))
    if len(match_files) > 1:
        raise IOError(f'Multiple matches found for: {match}')
    elif not match_files:
        raise IOError(f'No match found for: {match}')
    run_file = match_files[0]
    data = pd.read_csv(run_file, sep='\t')
    return data


def read_period(subj_dir, period, pool=None, reindex=True):
    """Read log data for a period."""
    n_run = {
        'c_loc': 4, 'a_prex': 6, 'p_study': 1, 'p_feedback': 1,
        'ab_study': 4, 'ab_feedback': 4, 'bcxy_study': 6,
        'ac_test': 5, 'bcxy_test': 1, 'ab_test': 1
    }
    trial_type = period.split('_')[0].upper()
    df_run = []
    for run in range(1, n_run[period] + 1):
        dr = read_run(subj_dir, period, run)
        dr.loc[:, 'run'] = run
        df_run.append(dr)
    df = pd.concat(df_run, ignore_index=True)
    keys = ['run', 'trial', 'onset', 'trial_type']
    df['trial_type'] = trial_type
    if 'group' in df:
        keys += ['group']
    if 'cond' in df:
        df['category'] = df['cond'].map(
            {1: 'face', 2: 'face', 3: 'scene', 4: 'scene', 5: 'object'}
        )
        df['subcategory'] = df['cond'].map(
            {1: 'female', 2: 'male', 3: 'manmade', 4: 'natural', 5: 'object'}
        ).astype('category')
        df['group_type'] = df['category'].map(
            {'face': 'ABC', 'scene': 'ABC', 'object': 'XY'}
        )
        if 'bcxy' in period:
            df['trial_type'] = df['group_type'].map({'ABC': 'BC', 'XY': 'XY'})
        elif period in ['ab_study', 'ab_feedback']:
            df['trial_type'] = df['run'].map(
                {1: 'AB1', 2: 'AB2', 3: 'AB3', 4: 'AB4'}
            )
        keys += ['group_type', 'category', 'subcategory']
    if pool is not None:
        stim_dict = pool['stim'].to_dict()
        item_dict = {
            'itemno1': 'item1', 'itemno2': 'item2', 'itemno': 'item',
            'probe1': 'probe1', 'probe2': 'probe2', 'probe3': 'probe3'
        }
        for source, dest in item_dict.items():
            if source in df:
                df[dest] = (df[source] - 1).map(stim_dict)
                keys.append(dest)
    if 'test' in period or 'feedback' in period:
        df['response'] = df['resp']
        df['response_time'] = df['rt'] / 1000
        keys += ['target', 'response', 'response_time', 'correct']
    if reindex:
        df = df.reindex(keys, axis=1)
    return df


def read_data(data_dir, periods, subjects=None, **kwargs):
    """Read data for multiple periods and subjects."""
    if subjects is None:
        subjects = get_subjects('react')
    df_all = []
    for subject in subjects:
        subj_dir = os.path.join(data_dir, subject, 'behav', 'log')
        df_subject = []
        for period in periods:
            dp = read_period(subj_dir, period, **kwargs)
            df_subject.append(dp)
        ds = pd.concat(df_subject, axis=0, ignore_index=True)
        ds['subject'] = subject
        df_all.append(ds)
    df = pd.concat(df_all, axis=0, ignore_index=True)
    df['trial_type'] = df['trial_type'].astype('category')
    order = ['AB1', 'AB2', 'AB3', 'AB4', 'AC', 'BC', 'XY', 'AB']
    inc_order = [o for o in order if o in df['trial_type'].unique()]
    df['trial_type'].cat.reorder_categories(inc_order, ordered=True, inplace=True)
    return df


def prex_vols(subj_dir):
    """Task information for each pre-exposure task volume."""
    data = read_prex(subj_dir)
    vols = data.groupby(['run', 'itemno'], as_index=False).mean()
    vols = vols[['run', 'cond', 'itemno', 'group', 'correct', 'rt']]
    vols = vols.astype({'itemno': int, 'cond': int, 'group': int})
    return vols
