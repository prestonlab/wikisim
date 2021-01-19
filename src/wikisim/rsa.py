"""Representational similarity analysis."""

import os
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy import optimize
import scipy.spatial.distance as sd
import pandas as pd

from mindstorm import prsa
from wikisim import task
from wikisim import model


def rank_dsm(dsm):
    """Rank transform a dissimilarity matrix."""
    res = sd.squareform(stats.rankdata(sd.squareform(dsm)))
    return res


def define_network(roi_names, roi_labels, net_ids, net_names, net_labels):
    """Define ROIs in a network."""
    nn = [net_names[i] for i in net_ids]
    nl = [net_labels[i] for i in net_ids]
    df = pd.DataFrame({'roi_label': roi_labels, 'net': nn, 'net_label': nl,
                       'net_id': net_ids}, index=roi_names)
    return df


def get_rois(roi_group):
    """Get a list of ROI names for a given ROI group."""
    if isinstance(roi_group, list):
        rois = roi_group
        labels = roi_group
        net_ids = [0] * len(roi_group)
        net_names = ['roi'] * len(roi_group)
        net_labels = ['roi'] * len(roi_group)
    elif roi_group == 'func':
        rois = ['r_ofa', 'b_ffa', 'b_pat', 'b_opa', 'b_rsc', 'b_ppa']
        labels = ['OFA', 'FFA', 'ATFA', 'OPA', 'RSC', 'PPA']
        net_ids = [0, 0, 0, 1, 1, 1]
        net_names = ['face', 'scene']
        net_labels = ['Face', 'Scene']
    elif roi_group == 'pmat':
        rois = ['b_hopa_ang', 'b_hopa_prec', 'b_hopa_rscpcc', 'b_phc_jl_mni',
                'b_fshs_phpc', 'b_fshs_ahpc',
                'b_prc_jl_mni', 'b_hopa_amyg', 'b_hopa_fus', 'b_hopa_itc',
                'b_hopa_tpo', 'b_hopa_pope', 'b_hopa_ptri', 'b_hopa_ofc']
        labels = ['ANG', 'PREC', 'PCC', 'PHC',
                  'pHPC', 'aHPC',
                  'PRC', 'AMYG', 'FUS', 'ITC', 'TPO', 'OPER', 'TRIA', 'OFC']
        net_ids = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        net_names = ['pm', 'hpc', 'at']
        net_labels = ['PM', 'HPC', 'AT']
    elif roi_group == 'pm-at':
        rois = ['b_hopa_ang', 'b_hopa_prec', 'b_hopa_rscpcc', 'b_phc_jl_mni',
                'b_prc_jl_mni', 'b_hopa_amyg', 'b_hopa_fus', 'b_hopa_itc',
                'b_hopa_tpo', 'b_hopa_pope', 'b_hopa_ptri', 'b_hopa_ofc']
        labels = ['ANG', 'PREC', 'PCC', 'PHC',
                  'PRC', 'AMYG', 'FUS', 'ITC', 'TPO', 'OPER', 'TRIA', 'OFC']
        net_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        net_names = ['pm', 'at']
        net_labels = ['PM', 'AT']
    else:
        rois = [roi_group]
        labels = [roi_group]
        net_ids = [0]
        net_names = ['roi']
        net_labels = ['ROI']

    nn = [net_names[i] for i in net_ids]
    nl = [net_labels[i] for i in net_ids]
    df = pd.DataFrame({'roi_label': labels, 'net': nn, 'net_label': nl,
                       'net_id': net_ids}, index=rois)
    return df


def load_roi_pattern(roi_dir, subject):
    """Load ROI patterns from exported data set."""
    mat_file = os.path.join(roi_dir, f'pattern_{subject}.txt')
    tab_file = os.path.join(roi_dir, f'pattern_{subject}.csv')
    vols = pd.read_csv(tab_file, index_col=0)
    patterns = np.loadtxt(mat_file)
    return patterns, vols


def load_roi_rdm(roi_dir, subject, item_groups=None, rank=False):
    """Load item group RDMs for an ROI."""
    if item_groups is None:
        item_groups = {'face': slice(None, 60), 'scene': slice(60, None)}

    patterns, vols = load_roi_pattern(roi_dir, subject)
    rdm = {}
    for group, ind in item_groups.items():
        rdv = sd.pdist(patterns[ind, :], 'correlation')
        if rank:
            rdv = stats.rankdata(rdv)
        rdm[group] = sd.squareform(rdv)
    return rdm


def load_roi_corr(roi_dir, rois, subj_ids=None, item_groups=None):
    """Load activity correlations between ROIs."""
    if subj_ids is None:
        subj_ids = task.get_subjects()

    if item_groups is None:
        item_groups = {'face': slice(None, 60), 'scene': slice(60, None)}

    n_roi = len(rois)
    n_subj = len(subj_ids)
    corr = {}
    for group in item_groups.keys():
        corr[group] = np.zeros((n_subj, n_roi, n_roi))

    for i, subj_id in enumerate(subj_ids):
        roi_list = []
        for roi in rois:
            patterns, vols = load_roi_pattern(os.path.join(roi_dir, roi), subj_id)
            roi_list.append(np.mean(patterns, 1)[:, None])
        roi_mat = np.hstack(roi_list)

        for group, ind in item_groups.items():
            x = roi_mat[ind, :].T
            corr_mat = 1 - sd.squareform(sd.pdist(x, 'correlation'))
            corr[group][i, :, :] = corr_mat
    return corr


def load_roi_zstat(res_dir, roi, subj_ids=None):
    """Load z-statistic from permutation test results."""
    if subj_ids is None:
        subj_ids = task.get_subjects()

    z = []
    for subj_id in subj_ids:
        subj_file = os.path.join(res_dir, 'roi', roi, f'zstat_{subj_id}.csv')
        zdf = pd.read_csv(subj_file, index_col=0).T
        zdf.index = [subj_id]
        z.append(zdf)
    df = pd.concat(z)
    return df


def load_net_zstat(rsa_dir, model_set, category, rois, subj_ids=None,
                   suffix=None):
    """Load z-statistics for a model for a set of ROIs."""
    # get the directory with results for this model set and category
    res_dir = os.path.join(rsa_dir, f'prex_{model_set}_{category}')
    if suffix is not None:
        res_dir += suffix
    if not os.path.exists(res_dir):
        raise IOError(f'Results directory not found: {res_dir}')

    # get subjects to load
    if subj_ids is None:
        subj_ids = task.get_subjects()

    # load each ROI
    df_list = []
    for name, roi in rois.iterrows():
        rdf = load_roi_zstat(res_dir, name, subj_ids)
        mdf = pd.DataFrame({'subj_id': rdf.index, 'roi': roi.roi_label,
                            'net': roi.net_label, 'net_id': roi.net_id},
                           index=rdf.index)
        full = pd.concat((mdf, rdf), axis=1)

        df_list.append(full)
    df = pd.concat(df_list, ignore_index=True)
    return df


def load_set_zstat(rsa_dir, model_sets, category, rois, subj_ids=None):
    """Load z-statistics for multiple model sets."""
    df_list = []
    for name, models in model_sets.items():
        dfs = load_net_zstat(rsa_dir, name, category, rois, subj_ids)
        dfs = dfs.rename(columns=models)
        dfs.loc[:, 'set'] = name
        df_list.append(dfs)
    df = pd.concat(df_list, ignore_index=True)
    return df


def resid_rdm(rdm, control_rdms, rank=False):
    """Residualize an RDM after controlling for other RDMs."""
    # get vector forms of the rdms
    if rank:
        rdv = stats.rankdata(sd.squareform(rdm))
        control_rdv = [stats.rankdata(sd.squareform(control_rdm))
                       for control_rdm in control_rdms]
    else:
        rdv = sd.squareform(rdm)
        control_rdv = [sd.squareform(control_rdm)
                       for control_rdm in control_rdms]

    # prepare model
    X = np.vstack((*control_rdv, np.ones(rdv.shape))).T
    y = rdv
    beta, rnorm = optimize.nnls(X, y)
    res = y - X.dot(beta)
    rdm_res = sd.squareform(res)
    return rdm_res


def model_set_prsa(model_set, ref_name, model_names, n_perm=10000):
    """Run partial RSA on a model set."""
    # unpack into reference and other models
    ref_rdm = model_set[ref_name]
    other_set = {name: model_set[name] for name in model_names}

    # create permuted models
    model_rdms = list(other_set.values())
    model_names = list(other_set.keys())
    spec = prsa.init_pRSA(n_perm, model_rdms)

    # run partial correlation analysis
    results = {}
    ref_rdv = stats.rankdata(sd.squareform(ref_rdm))
    for i in range(len(model_rdms)):
        mat = spec['model_mats'][i]
        resid = spec['model_resid'][i]
        stat = prsa.perm_partial(ref_rdv, mat, resid)
        z = prsa.perm_z(stat)
        p = 1 - stats.norm.cdf(z)
        results[model_names[i]] = {'z': z, 'p': p, 'stat': stat[0]}
    return pd.DataFrame(results).T


def sign_perm(mat, n_perm, method='fdr', tail='right'):
    """Test significance and control for family-wise error."""
    # generate random sign flips
    n_samp, n_test = mat.shape
    rand_sign = np.hstack((np.ones((n_samp, 1)),
                           np.random.choice([-1, 1], (n_samp, n_perm))))

    # apply random sign to all conditions
    mat_perm = mat[:, :, None] * rand_sign[:, None, :]
    stat_perm = np.mean(mat_perm, 0)
    if tail == 'both':
        stat_perm = np.abs(stat_perm)
    elif tail == 'left':
        stat_perm = -stat_perm

    # compare each condition to sign flip distribution
    p = np.mean(stat_perm >= stat_perm[:, :1], 1)

    # compare to pooled distribution
    if method == 'fdr':
        p_cor = np.mean(stat_perm.flatten() >= stat_perm[:, :1], 1)
    elif method == 'fdr_bh':
        out = multipletests(p, alpha=0.05, method='fdr_bh')
        p_cor = out[1]
    elif method == 'fwe':
        stat_max = np.max(stat_perm, 0)
        p_cor = np.mean(stat_max[None, :] >= stat_perm[:, :1], 1)
    elif method == 'none':
        p_cor = p.copy()
    else:
        raise ValueError(f'Invalid method: {method}')
    return p, p_cor


def roi_zstat_perm(df, model, n_perm=100000, method='fdr'):
    """Test ROI correlations using a permutation test."""
    # shape into matrix format
    rois = df.roi.unique()
    mat = df.pivot(index='subj_id', columns='roi', values=model)
    mat = mat.reindex(columns=rois)

    # run sign flipping test
    p, p_cor = sign_perm(mat.to_numpy(), n_perm, method=method)
    results = pd.DataFrame({'p': p, 'p_cor': p_cor}, index=rois)

    # add stats for mean
    zstat = mat.agg(['mean', 'sem', 'std']).T
    results = pd.concat((zstat, results), axis=1)
    results['d'] = results['mean'].abs() / results['std']
    return results


def net_zstat_perm(df, model, n_perm=100000, method='fdr'):
    """Test ROI correlations separately by network."""
    res_list = []
    for idx, zstat in df.groupby('net'):
        res = roi_zstat_perm(zstat, model, n_perm, method)
        res_list.append(res)
    results = pd.concat(res_list, 0)
    return results


def roi_resid_geo(roi_dir, model_dir, control_models, geo_edges,
                  geo_model='geo', subj_ids=None, rank=False):
    """Calculate ROI residuals within geography distance bins."""
    if subj_ids is None:
        subj_ids = task.get_subjects()

    # vector representation of dissimilarity for geo and control models
    model_names = control_models + [geo_model]
    model_rdms = model.load_category_rdms(model_dir, 'scene', model_names)
    control_rdms = [model_rdms[name] for name in control_models]
    geo_rdm = model_rdms[geo_model]
    geo_rdv = sd.squareform(geo_rdm)

    # residualize roi rdms
    resid_list = []
    for subj_id in subj_ids:
        roi_rdm = load_roi_rdm(roi_dir, subj_id)['scene']
        roi_resid = resid_rdm(roi_rdm, control_rdms, rank=rank)

        # z-score dissimilarity within subject
        resid_list.append(stats.zscore(sd.squareform(roi_resid)))

    # calculate neural dissimilarity as a function of distance bin
    means, edges, number = stats.binned_statistic(geo_rdv, resid_list, 'mean',
                                                  bins=geo_edges)
    centers = geo_edges[:-1] + np.diff(geo_edges) / 2

    df = pd.DataFrame(means, index=subj_ids, columns=centers).reset_index()
    df = df.melt(id_vars='index', var_name='distance',
                 value_name='dissimilarity')
    return df


def dependent_corr(xy, xz, yz, n, twotailed=True):
    """
    Test difference between two dependent correlation coefficients.

    Test if x is more correlated with y than with z, given that y and z
    may be correlated themselves.

    See Williams (1959) Journal of the Royal Statistical Society,
    Series B and Steiger (1980) Psychological Bulletin.

    Taken from: https://github.com/psinger/CorrelationStats

    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test
    @return: t and p-val
    """
    d = xy - xz
    determ = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz) / 2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt(
        (n - 1) * (1 + yz) / ((2 * (n - 1) / (n - 3)) * determ + av * av * cube)
    )
    p = 1 - stats.t.cdf(abs(t2), n - 3)
    if twotailed:
        p *= 2
    return t2, p


def compare_rdm_corr(x, y, z):
    """Compare correlation between one RDM and two others."""
    xy = stats.spearmanr(sd.squareform(x), sd.squareform(y))[0]
    xz = stats.spearmanr(sd.squareform(x), sd.squareform(z))[0]
    yz = stats.spearmanr(sd.squareform(y), sd.squareform(z))[0]
    n = len(sd.squareform(x))
    t, p = dependent_corr(xy, xz, yz, n)
    df = n - 3
    return t, p, df
