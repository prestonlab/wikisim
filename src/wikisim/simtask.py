"""Process MTurk similarity task data."""

import numpy as np
from scipy import stats
import pandas as pd


def gen_status(n_trial):
    """Generate status messages for procedure spreadsheet."""
    for i in range(n_trial):
        message = f'trial {i + 1} of {n_trial}'
        progress = f'{i / n_trial:.4f}'
        print(f'"status_message": "{message}", "status_progress": {progress}')


def remove_incomplete(phase, field):
    """Remove incomplete participants from a phase."""
    include = []
    for subject, df in phase.groupby('subject'):
        if not df.isnull().agg('any')[field]:
            include.append(subject)
    complete = phase.loc[np.isin(phase.subject, include)].copy()
    return complete


def _parse_trials(raw, include_trials, cols=None):
    """Parse trials to filter and standardize column names."""
    # get just the relevant trials
    trials = raw.loc[raw['Exp_Procedure*Trial Type'].isin(include_trials)]

    # get all columns mappings to include
    std_cols = {'Exp_ID': 'subject',
                'Exp_Date': 'datetime',
                'Exp_TimeDif': 'duration',
                'Exp_Condition Number': 'condition'}
    all_cols = {**std_cols, **cols} if cols is not None else std_cols

    # filter and rename columns; reset the index
    parsed = trials.copy()
    parsed = parsed.filter(items=list(all_cols.keys()), axis=1)
    parsed = parsed.rename(mapper=all_cols, axis=1)

    # convert datetime string to datetime
    parsed.datetime = pd.to_datetime(parsed.datetime)

    # add trial type code
    parsed.loc[:, 'trial_type'] = trials['Exp_Procedure*Trial Type']
    parsed = parsed.reset_index(drop=True)
    return parsed


def parse_instruct(raw):
    """Parse instruction trials."""
    instruct = _parse_trials(raw, ['instruct'])
    instruct.trial_type = 'instruct'
    return instruct


def parse_fam(raw, pool):
    """Parse familiarity responses."""
    # get trials, filter and rename columns
    cols = {'Exp_Stimuli*Answer': 'stim',
            'Exp_Response*Resp_Value': 'response',
            'Exp_Response*RT': 'response_time'}
    fam = _parse_trials(raw, ['MCpicFamiliar'], cols=cols)
    fam.loc[:, 'trial_type'] = 'familiarity'

    # onset field relative to start of phase
    onset = np.cumsum(fam.duration.to_numpy())
    fam.loc[:, 'onset'] = np.insert(onset[:-1], 0, 0)

    # fix field formatting
    fam = fam.astype({'response': 'int'})
    fam.response_time = fam.response_time / 1000

    # add item number field
    fam.loc[:, 'stim_id'] = np.zeros(fam.shape[0], dtype='int')
    for i, stim in fam.stim.items():
        fam.loc[i, 'stim_id'] = pool.index(stim)

    # set column order
    fam = fam.reindex(['subject', 'onset', 'duration', 'trial_type',
                       'datetime', 'condition', 'stim', 'stim_id',
                       'response', 'response_time'], axis=1)
    return fam


def _filter_sim(raw):
    """Filter raw data to get information relevant to similarity judgments."""
    # get trials, filter and rename columns
    cols = {'Exp_Response*Target_Answer': 'stim',
            'Exp_Response*Response1': 'response1',
            'Exp_Response*Response1_RT': 'response_time1',
            'Exp_Response*Response2': 'response2',
            'Exp_Response*Response2_RT': 'response_time2',
            'Exp_Response*All_Answers': 'stim_array'}
    include_trials = ['similarityJudgment', 'similarityJudgmentFlip']
    sim = _parse_trials(raw, include_trials, cols=cols)

    # onset field relative to start of phase
    onset = np.cumsum(sim.duration.to_numpy())
    sim.loc[:, 'onset'] = np.insert(onset[:-1], 0, 0)

    # fix field formatting
    sim.response_time1 = sim.response_time1 / 1000
    sim.response_time2 = sim.response_time2 / 1000
    sim.loc[sim.trial_type == 'similarityJudgment', 'trial_type'] = 'similarity'
    sim.loc[sim.trial_type == 'similarityJudgmentFlip', 'trial_type'] = 'catch'
    return sim


def parse_sim(raw, pool):
    """Parse similarity judgment data."""
    # get similarity judgment trials and fields
    filtered = _filter_sim(raw)

    # remove participants with any missing second responses
    sim = remove_incomplete(filtered, 'response2')

    for i, trial in sim.iterrows():
        # all responses in standard order (query, resp1, resp2, others)
        stims = [trial.stim, trial.response1, trial.response2]
        array = trial.stim_array.split(',')
        others = [stim for stim in array if stim not in stims]
        all_stims = stims + others

        # full response information in string form
        response_ids = [pool.index(stim) for stim in all_stims]
        sim.loc[i, 'response'] = ':'.join(map(str, response_ids))

        # full array information
        array_ids = [pool.index(stim) for stim in array]
        sim.loc[i, 'array'] = ':'.join(map(str, array_ids))

    # set column order
    sim = sim.reindex(['subject', 'onset', 'duration', 'trial_type',
                       'datetime', 'condition', 'stim',
                       'response1', 'response_time1',
                       'response2', 'response_time2',
                       'response', 'array'], axis=1)
    return sim


def score_sim(sim):
    """Score catch trials in similarity data."""
    score = sim.copy()
    score.loc[:, 'correct'] = np.nan
    catch = score.loc[score.trial_type == 'catch']
    for i, trial in catch.iterrows():
        if trial.stim in [trial.response1, trial.response2]:
            score.loc[i, 'correct'] = 1
        else:
            score.loc[i, 'correct'] = 0
    return score


def parse_prac(raw):
    """Parse practice similarity judgment data."""
    # get trials, filter and rename columns
    cols = {'Exp_Stimuli*Answer': 'query',
            'Exp_Stimuli*Alt Cue 1': 'option1',
            'Exp_Stimuli*Alt Cue 2': 'option2',
            'Exp_Response*Response1': 'response',
            'Exp_Response*RT': 'response_time'}
    prac = _parse_trials(raw, ['similarityTriad'], cols=cols)
    prac.loc[:, 'trial_type'] = 'practice'

    # onset field
    onset = np.cumsum(prac.duration.to_numpy())
    prac.loc[:, 'onset'] = np.insert(onset[:-1], 0, 0)

    # standardize trial information
    prac.response_time = prac.response_time / 1000
    for i, trial in prac.iterrows():
        option1 = trial.option1.split('/')[-1].split('.')[0]
        option2 = trial.option2.split('/')[-1].split('.')[0]
        prac.loc[i, 'option1'] = option1
        prac.loc[i, 'option2'] = option2
        if isinstance(trial.response, float) and np.isnan(trial.response):
            prac.loc[i, 'response'] = 'semantic'

    prac = prac.reindex(['subject', 'onset', 'duration', 'trial_type',
                         'datetime', 'condition', 'query', 'option1', 'option2',
                         'response', 'response_time'], axis=1)
    return prac


def score_prac(prac):
    """Score practice phase data based on condition."""
    score = prac.copy()
    score.loc[:, 'correct'] = 0
    condition = score.condition.to_numpy()
    response = score.response.to_numpy()
    test_key = {1: 'semantic', 2: 'semantic', 3: 'visual', 4: 'visual'}
    for cond, answer in test_key.items():
        score.loc[(condition == cond) & (response == answer), 'correct'] = 1
    return score


def parse_dem(raw):
    """Parse demographics information."""
    subject = raw.groupby('Exp_ID').first()
    subject = subject.filter(like='Dem_', axis=1)
    cols = {'Dem_Username': 'username',
            'Dem_Gender': 'gender',
            'Dem_Age': 'age',
            'Dem_Education': 'education',
            'Dem_Race': 'race',
            'Dem_Fluent': 'english_fluent',
            'Dem_AgeEnglish': 'english_age',
            'Dem_Country': 'country'}
    subject = subject.filter(items=list(cols.keys()), axis=1)
    subject = subject.rename(mapper=cols, axis=1)

    for i, row in subject.iterrows():
        if '|' in subject.loc[i, 'gender']:
            for j, entry in subject.loc[i].iteritems():
                new_entry = entry.split('|')[0]
                subject.loc[i, j] = new_entry

    subject = subject.astype({'age': int})
    return subject


def parse_debrief(raw):
    """Parse debriefing information."""
    subject = raw.groupby('Exp_ID').first()
    deb = subject.filter(like='FiQ_', axis=1)

    cols = {('FiQ_How often did you feel like there was a clear choice of '
             'which items were most similar?'): 'clear',
            ('FiQ_How often were your choices influenced by how similar '
             'the pictures looked?'): 'vis',
            ('FiQ_How often were your choices influenced by what you knew '
             'of the pictures?'): 'sem',
            ('FiQ_In cases in which it was easier to make similarity judgments '
             'and you did not select at random, how did you make your choices? '
             'Please be specific.'): 'description',
            ('FiQ_Is there any other feedback you can provide that could be '
             'helpful to us? For example, was the task too difficult, '
             'boring, etc.?'): 'feedback',
            ('FiQ_Did the experiment go smoothly or were there problems? Your '
             'compensation will not depend on your answer below, so please be '
             'honest!'): 'quality',
            'FiQ_Which of the following problems did you have?_none':
                'problems_none',
            'FiQ_Which of the following problems did you have?_blank page':
                'problems_blank'}
    deb = deb.rename(mapper=cols, axis=1)
    deb.loc[:, 'condition'] = subject['Exp_Condition Number']
    deb = deb.reindex(['condition', 'clear', 'vis', 'sem',
                       'description', 'feedback',
                       'quality', 'problems_none', 'problems_blank'], axis=1)
    return deb


def read_mturk(data_file, pool_file):
    """Read and parse MTurk data."""
    # read all data
    raw = pd.read_csv(data_file, low_memory=False)
    pool = pd.read_csv(pool_file)

    # parse into phases
    stim = pool.stim.to_list()
    data = {'instruct': parse_instruct(raw),
            'fam': parse_fam(raw, stim),
            'prac': parse_prac(raw),
            'sim': parse_sim(raw, stim),
            'dem': parse_dem(raw),
            'deb': parse_debrief(raw)}

    # score practice test
    data['prac'] = score_prac(data['prac'])

    # score catch trials
    data['sim'] = score_sim(data['sim'])

    # add familiarity ratings to similarity task
    data['sim'] = add_fam(data['fam'], data['sim'])
    return data, pool


def duration(phase_times):
    """Duration of an experiment phase."""
    dt = phase_times.to_numpy()
    td = dt[-1] - dt[0]
    return td.seconds / 60


def session_duration(data):
    """Duration of experiment phases."""
    dfs = []
    phases = ['instruct', 'fam', 'prac', 'sim']
    for name in phases:
        df = data[name]
        phase_dur = df.groupby('subject')['duration'].sum() / 60
        phase_dur.name = name
        dfs.append(phase_dur)
    df_dur = pd.concat(dfs, axis=1, sort=False)
    df_dur.loc[:, 'total'] = df_dur.sum(1)
    return df_dur


def add_fam(fam, sim):
    """Add a familiarity rating column to similarity rating data."""
    sim = sim.copy()
    sim.loc[:, 'stim_fam'] = np.nan
    sim.loc[:, 'fam'] = ''
    for subject in fam['subject'].unique():
        fam_subj = fam.loc[fam['subject'] == subject]
        sim_subj = sim.loc[(sim['subject'] == subject) &
                           (sim['trial_type'] == 'similarity')]
        if len(sim_subj) == 0:
            continue

        df = sim_subj['response'].str.split(':', expand=True).astype(int)
        mat = df.to_numpy()

        resp = np.zeros(mat.shape, int)
        for stim_id in fam_subj['stim_id']:
            if np.count_nonzero(mat == stim_id) == 0:
                continue
            response = fam_subj.loc[fam['stim_id'] == stim_id, 'response']
            resp[mat == stim_id] = response

        s = pd.Series([':'.join(r.astype(str)) for r in resp], index=df.index)
        sim.loc[s.index, 'fam'] = s
        sim.loc[s.index, 'stim_fam'] = resp[:, 0]
    return sim


def session_summary(data):
    """Summarize session statistics."""
    # condition for each subject
    cond = data['fam'].groupby('subject').condition.first()
    cond.name = 'condition'

    # mean familiarity rating over all items
    fam = data['fam'].groupby('subject').response.mean()
    fam.name = 'familiarity'

    # performance on the object practice test
    test_perf = data['prac'].groupby('subject').correct.mean()
    test_perf.name = 'test'

    # performance for similarity ratings on catch trials
    catch = data['sim'].query('trial_type == "catch"')
    catch_perf = catch.groupby('subject').correct.mean()
    catch_perf.name = 'catch'

    # debriefing questions about visual vs. semantic ratings
    deb = data['deb'].loc[:, ['vis', 'sem']]

    # session start date and time
    start = data['instruct'].groupby('subject')['datetime'].first()
    start.name = 'start_time'

    # duration of each experiment phase
    dur = session_duration(data)

    # place all stats in one data frame; sort by condition, then performance
    age = data['dem']['age']
    df = pd.concat((start, cond, fam, test_perf, catch_perf, deb, dur, age), 1,
                   sort=False)
    df = df.sort_values(by=['condition', 'test', 'catch'],
                        ascending=[True, False, False])
    return df


def read_fam(fam_file):
    """Read MTurk familiarity stats."""
    fam = pd.read_csv(fam_file, index_col=0)
    fam = fam.astype({'stim_id': int})
    fam.loc[fam['condition'].isin([1, 3]), 'category'] = 'face'
    fam.loc[fam['condition'].isin([2, 4]), 'category'] = 'scene'
    fam.loc[(fam['stim_id'] >= 0) & (fam['stim_id'] < 30), 'subcategory'] = 'female'
    fam.loc[(fam['stim_id'] >= 30) & (fam['stim_id'] < 60), 'subcategory'] = 'male'
    fam.loc[(fam['stim_id'] >= 60) & (fam['stim_id'] < 90), 'subcategory'] = 'manmade'
    fam.loc[(fam['stim_id'] >= 90) & (fam['stim_id'] < 120), 'subcategory'] = 'natural'
    return fam


def subsample(items, n, density):
    """Subsample items to match a distribution of familiarity."""
    item_fam = items['fam'].to_numpy()
    item_ind = items.index.to_numpy()

    # sampling probability as a fraction of the current density
    density_orig = stats.gaussian_kde(item_fam)
    p_target = density(item_fam)
    p_orig = density_orig(item_fam)
    p = p_target / p_orig
    p /= np.sum(p)

    # sample without replacement
    ind = np.random.choice(item_ind, n, p=p, replace=False)
    ind.sort()
    sub = items.loc[ind].copy()
    return sub


def match_fam_density(items, target_subcategory, pmin=0, pmax=1):
    """Match familiarity density to a target distribution."""
    # trim target distribution to be in range of other items
    other = items.query(f'subcategory != "{target_subcategory}"')
    fam_min = other.groupby('subcategory')['fam'].quantile(pmin).max()
    fam_max = other.groupby('subcategory')['fam'].quantile(pmax).min()
    trim = items.query(f'fam >= {fam_min} & fam <= {fam_max}')

    # estimate density of the target distribution
    target = trim.query(f'subcategory == "{target_subcategory}"')
    target_fam = target['fam'].to_numpy()
    density = stats.gaussian_kde(target_fam)
    min_n = len(target_fam)

    # subsample other subcategories to match the target
    sample = trim.groupby('subcategory').apply(
        lambda x: subsample(x, min_n, density)
    )
    sample.index = sample.index.droplevel(0)
    return sample
