"""Module to estimate embedding models."""

import os
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as sd
import pandas as pd
import psiz.trials
import psiz.models


def unpack_array(stim_series):
    """Get standard response array for similarity data."""
    stim_series = stim_series.copy().reset_index(drop=True)
    n_stim = len(stim_series[0].split(':'))
    mat = np.zeros((stim_series.shape[0], n_stim), dtype='int')
    for i, array in enumerate(stim_series):
        mat[i, :] = list(map(int, array.split(':')))
    return mat


def psiz_obs(sim):
    """Convert similarity data to psiz observations."""
    # exclude catch trials
    sim = sim.loc[sim.trial_type == 'similarity']

    # unpack response information
    response = unpack_array(sim.response)

    # convert to standard observations format
    n_trial = response.shape[0]
    n_select = 2 * np.ones(n_trial, dtype=int)
    resp_min = np.min(response)
    obs = psiz.trials.Observations(response - resp_min, n_select=n_select)
    return obs


def pool_embed(sim, pool, n_dim=6):
    """Estimate embedding for each item in a pool."""
    # estimate embedding
    n_stimuli = pool.shape[0]
    obs = psiz_obs(sim)
    emb = psiz.models.Exponential(n_stimuli, n_dim, n_group=1)
    emb.fit(obs)

    # convert to dataframe
    z = pd.DataFrame(emb.z, columns=[f'dim{i}' for i in range(n_dim)])

    # concatenate with the pool
    stim = pool.copy().reset_index(drop=True)
    df = pd.concat((stim, z), axis=1)
    return df


def indiv_embed(sim, pool, save_dir=None, overwrite=False, n_dim=6):
    """Calculate embedding for each individual participant."""
    subjects = sim.subject.unique()
    df_list = []
    for subject in subjects:
        # if an embedding was created previously, load that
        if save_dir is not None:
            save_file = os.path.join(save_dir, f'embed_{subject}.csv')
            if os.path.exists(save_file) and not overwrite:
                df = pd.read_csv(save_file, index_col=0)
                df_list.append(df)
                continue

        # get subject data and relevant pool
        subj_sim = sim.loc[sim.subject == subject]
        condition = subj_sim.iloc[0].loc['condition']
        if condition in [1, 3]:
            subj_pool = pool.loc[pool.category == 'face']
        else:
            subj_pool = pool.loc[pool.category == 'scene']

        # calculate embedding
        df = pool_embed(subj_sim, subj_pool, n_dim=n_dim)
        df.loc[:, 'subject'] = subject
        df.loc[:, 'condition'] = condition

        if save_dir is not None:
            df.to_csv(save_file)

        df_list.append(df)
    full = pd.concat(df_list, axis=0, ignore_index=True)
    return full


def cond_embed(sim, pool, n_dim=6):
    """Calculate embedding for each condition."""
    conds = sim.condition.unique()
    conds.sort()

    emb = {}
    for cond in conds:
        cond_sim = sim.loc[sim.condition == cond]
        if cond in [1, 3]:
            cond_pool = pool.loc[pool.category == 'face']
        else:
            cond_pool = pool.loc[pool.category == 'scene']

        n_stimuli = cond_pool.shape[0]
        obs = psiz_obs(cond_sim)
        emb_cond = psiz.models.Exponential(n_stimuli, n_dim, n_group=1)
        emb_cond.fit(obs)
        emb[cond] = emb_cond
    return emb


def embed_rdm(emb):
    """Calculate a dissimilarity matrix from an embedding."""
    n_item = emb.z.shape[0]
    ind = np.triu_indices(n_item, 1)
    dsm = sd.squareform(emb.distance(emb.z[ind[0]], emb.z[ind[1]]))
    return dsm


def save_embed_rdm(model_dir, model_name, embed, conds, items):
    """Save a model representational dissimilarity matrix."""
    shape = np.array([embed[cond].z.shape for cond in conds])
    n_item = int(np.sum(shape[:, 0]))
    n_dim = shape[0, 1]

    rdm = np.ones((n_item, n_item))
    vectors = np.zeros((n_item, n_dim))
    for cond in conds:
        if cond in [1, 3]:
            ind = slice(None, 60)
        else:
            ind = slice(60, None)
        rdm[ind, ind] = embed_rdm(embed[cond])
        vectors[ind, :] = embed[cond].z

    mat = {'rdm': rdm, 'vectors': vectors, 'items': items}
    model_file = os.path.join(model_dir, f'mat_{model_name}.mat')
    sio.savemat(model_file, mat)


def cond_convergence(sim, n_dim=6, n_partition=10, n_shuffle=4, n_back=2,
                     model=psiz.models.Exponential):
    """Test convergence of each condition."""
    df_list = []
    part_vec = np.tile(np.arange(n_partition), n_shuffle)
    samp_vec = np.repeat(np.arange(n_shuffle), n_partition)
    for condition in sim.condition.unique():
        print(f'Assessing convergence for condition {condition}...')
        obs = psiz_obs(sim.loc[sim.condition == condition])
        res = psiz.utils.assess_convergence(
            obs, model, n_stimuli=60, n_dim=n_dim, n_partition=n_partition,
            n_shuffle=n_shuffle, n_back=n_back, verbose=1)
        n = np.tile(res['n_trial_array'], n_shuffle)
        val_vec = res['val'].flatten()
        dfc = pd.DataFrame({'partition': part_vec, 'sample': samp_vec,
                            'n': n, 'val': val_vec, 'condition': condition})
        df_list.append(dfc)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df.dropna().sort_values(['condition', 'partition'])
    return df
