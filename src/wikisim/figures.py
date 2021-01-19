"""Module for making figures from RSA results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wikisim import rsa


def muted_palette(colormap, n_cond, spacing=2, buffer=4):
    """Create a palette more toward the center of a colormap."""
    n_colors = (n_cond - 1) * (spacing + 1) + 2 * buffer
    palette = sns.color_palette(colormap, n_colors=n_colors)
    muted = palette[buffer:-buffer:spacing]
    return muted


def center_chance(ax, chance, offset=None):
    """Plot chance level and center on it."""
    xlim = ax.get_xlim()
    ax.hlines(chance, xlim[0], xlim[1], linewidth=0.5)
    if offset is None:
        offset = np.max(np.abs(ax.get_ylim()))

    ax.set_ylim([-offset, offset])


def plot_sig(results, y, ax=None, alpha=0.05, plot_uncorr=False):
    """Add significance markers to a plot."""
    if ax is None:
        ax = plt.gca()

    # plot corrected significance as stars
    sig_cor = results.p_cor < alpha
    ind_cor = sig_cor.to_numpy().nonzero()[0]
    ax.plot(ind_cor, np.tile(y, ind_cor.shape), markersize=10,
            color='k', linestyle='', marker=(5, 2, 0))

    # plot uncorrected significance as pluses
    if plot_uncorr:
        sig = ~sig_cor & (results.p < alpha)
        ind = sig.to_numpy().nonzero()[0]
        ax.plot(ind, np.tile(y, ind.shape), markersize=10,
                color='k', linestyle='', marker=(4, 2, 0))


def plot_swarm_error(
    x=None, y=None, hue=None, palette=None, color='k', capsize=.5, data=None,
    n_boot=100000, ax=None, **sns_options
):
    """Plot swarm plot with error bars."""
    ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, palette=palette,
                       size=4.5, ax=ax, **sns_options)
    sns.pointplot(x=x, y=y, hue=hue, data=data, color=color, ci=95, join=False,
                  capsize=capsize, ax=ax, n_boot=n_boot, **sns_options)

    plt.setp(ax.lines, zorder=100, linewidth=1.5)
    plt.setp(ax.collections, zorder=100)

    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize='large')


def plot_roi_zstat(
    df, model, hue, palette, ax=None, sig=None, sig_offset=None, n_boot=100000,
    sig_col='p_cor', sig_alpha=0.05
):
    """Plot z-statistics by ROI."""
    if ax is None:
        ax = plt.gca()

    dodge = False if hue == 'net' else True

    # plot points and mean with error bars
    order = df.roi.unique()
    plot_swarm_error(x='roi', y=model, hue=hue, data=df, n_boot=n_boot,
                     order=order, dodge=dodge, palette=palette)

    if sig is not None:
        if sig_offset is None:
            y_min = df[model].min()
            y_max = df[model].max()
            eta = (y_max - y_min) * .15
            sig_offset = y_max + eta
            offset = y_max + eta * 1.5
        else:
            offset = None
        if sig_col != 'p_cor':
            sig = sig.copy()
            sig['p_cor'] = sig[sig_col]
        sig = sig.reindex(index=order)
        plot_sig(sig, sig_offset, alpha=sig_alpha)

    else:
        offset = None

    # plot chance and center on it
    center_chance(ax, 0, offset)

    # labels
    ax.set_ylabel('Partial correlation (z)')
    ax.get_legend().remove()


def plot_zstat_perm(
    df, model, test_type, n_perm=100000, n_boot=100000, method='fdr',
    sig_offset=3.75, max_offset=4, sig_col='p_cor', sig_alpha=0.05,
    by_network=False, ax=None
):
    """Plot zstat by ROI with significance."""
    if ax is None:
        ax = plt.gca()

    # test for significance and correct for multiple comparisons
    sig = rsa.roi_zstat_perm(
        df, model, n_perm=n_perm, method=method, by_network=by_network
    )

    if test_type == 'face':
        pal_name = 'Reds_r'
        test = 'Face'
    elif test_type == 'scene':
        pal_name = 'Blues_r'
        test = 'Scene'
    else:
        pal_name = 'Greens_r'
        test = 'Scene'

    # get subset of palette that varies in color
    # but isn't too saturated
    full = sns.color_palette(pal_name, n_colors=7)
    if len(df.net.unique()) == 2:
        palette = full[4:6]
    else:
        palette = full[3:6]

    plot_roi_zstat(df, model, 'net', palette, ax=ax, n_boot=n_boot,
                   sig=sig, sig_offset=sig_offset, sig_col=sig_col,
                   sig_alpha=sig_alpha)
    ax.set_ylim(-max_offset, max_offset)

    # label axes
    model_type = 'geography' if test_type == 'geo' else 'semantic'
    ax.set_ylabel(f'{test} {model_type} model (z)')

    # plot separating line between networks
    order = df.roi.unique()
    labels = df.groupby('roi').first().net.loc[order]
    prev = None
    shift = []
    for i, label in enumerate(labels):
        if prev is not None and label != prev:
            shift.append(i)
        prev = label
    ax.vlines(np.array(shift) - .5, *ax.get_ylim(), linewidth=0.5)
    return sig


def plot_roi_resid_geo(
    roi_dir, roi, roi_label, model_dir, control_models, geo_model='geo',
    subj_ids=None, rank=False, ax=None
):
    """Plot ROI RDM residuals as a function of geographical distance."""
    geo_edges = np.arange(0, 16000, 3000)
    df = rsa.roi_resid_geo(os.path.join(roi_dir, roi), model_dir,
                           control_models, geo_edges, geo_model=geo_model,
                           subj_ids=subj_ids, rank=rank)
    palette = sns.color_palette('Greens', 2)

    if ax is None:
        ax = plt.gca()

    g = sns.lineplot(x='distance', y='dissimilarity', color=palette[1],
                     data=df, ax=ax)
    g.set(xlim=(0, 15000), xticks=geo_edges,
          xlabel='Geographical distance (km)',
          ylabel=f'{roi_label} neural distance (z)')
    return g
