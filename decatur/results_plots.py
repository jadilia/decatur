#!/usr/bin/env python
# encoding: utf-8
"""
Produce the results plots.
"""

from __future__ import print_function, division, absolute_import

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils
from .config import data_dir


def get_classification_results(class_file, catalog_file):
    """
    Get the classification results file.
    """
    class_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              'data', class_file))

    df = pd.read_pickle(class_file)
    kebc = utils.load_catalog(catalog_file)
    class_df = pd.merge(kebc, df, on='KIC')

    return class_df


def plot_prot_porb(class_file, plot_file=None, catalog_file='kebc.csv'):
    """
    Plot P_orb / P_rot vs. P_orb.

    Parameters
    ----------
    class_file : str
        Pickle file containing the classifications and rotation periods.
    plot_file : str, optional
        Specify an alternate output plot file.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    """
    join = get_classification_results(class_file, catalog_file)

    spot_mask = join['class'] == 'sp'
    ev_mask = join['class'] == 'ev'

    fig, ax = plt.subplots()

    p_orb_p_rot = join['period_x'] / join['p_rot_1']

    colors = ['b', 'r']
    labels = ['Ellipsoidals', 'Starspots']

    for ii, mask in enumerate([ev_mask, spot_mask]):
        ax.scatter(join['period_x'][mask], p_orb_p_rot[mask], color=colors[ii],
                   s=5, label=labels[ii])

    flat_mask = join['class'] == 'fl'
    non_detections = np.repeat([5e-2], np.sum(flat_mask))
    ax.scatter(join['period_x'][flat_mask], non_detections, color='g', s=5,
               label='Non-detections')

    ax.set_xscale('log')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0, 3)
    ax.set_xlabel('$P_{orb}$ (days)')
    ax.set_ylabel('$P_{orb}/P_{rot}$')
    ax.minorticks_on()

    if plot_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        plot_file = 'rotation_periods.{}.pdf'.format(today)

    ax.legend(loc='upper left', scatterpoints=1, markerscale=4)

    plt.savefig('{}/{}'.format(data_dir, plot_file))


def synchronization_histogram(class_file, dy=0.025, plot_file=None,
                              catalog_file='kebc.csv'):
    """
    Plot histograms of P_orb / P_rot for different bins of P_orb.

    Parameters
    ----------
    class_file : str
        Pickle file containing the classifications and rotation periods.
    dy : float, optional
        The bin width in P_orb / P_rot
    plot_file : str, optional
        Specify an alternate output plot file.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    """
    df = get_classification_results(class_file, catalog_file)

    p_orb_p_rot = df['period_x'].values / df['p_rot_1'].values

    spot_mask = df['class'].values == 'sp'
    # ev_mask = df['class'].values == 'ev'

    p_orb_p_rot_bins = np.arange(0, 3 + dy, dy)
    period_bins = [0, 5, 10, 100]

    spot_hist = np.histogramdd([p_orb_p_rot[spot_mask],
                                df['period_x'].values[spot_mask]],
                               bins=[p_orb_p_rot_bins, period_bins])[0]

    # ev_hist = np.histogramdd([p_orb_p_rot[ev_mask],
    #                           df['period_x'].values[ev_mask]],
    #                          bins=[p_orb_p_rot_bins, period_bins])[0]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(5, 10))

    for ii, ax in enumerate([ax1, ax2, ax3]):
        ax.step(p_orb_p_rot_bins[:-1], spot_hist[:, ii], color='r', lw=1,
                label='Spots', where='post')
        # ax.step(p_orb_p_rot_bins[:-1], ev_hist[:, ii], color='b', lw=1,
        #         label='Ellip', where='post')

        ax.set_ylabel('Number')
        ax.text(0.5, 0.85, '${} < P_{{orb}} < {}$'.format(period_bins[ii],
                                                          period_bins[ii + 1]),
                transform=ax.transAxes)

    # ax1.legend(loc='upper left')

    ax3.set_xlabel('$P_{orb}/P_{rot}$')

    if plot_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        plot_file = 'sync_hist.{}.pdf'.format(today)

    fig.subplots_adjust(hspace=0.1)

    plt.savefig('{}/{}'.format(data_dir, plot_file))


def classification_metric(class_file, metric_file, metric1='pwidth',
                          metric2='corr', plot_file=None):
    """
    Make a scatter plot of the classification metrics color-code by class.

    Parameters
    ----------
    class_file : str
        Pickle file containing the classifications and rotation periods.
    metric_file : str
        CSV file containing the classification `metric1`.
    metric1, metric2 : str, optional
        The metrics to use.
    plot_file : str, optional
        Specify an alternate output plot file.
    """
    class_df = get_classification_results(class_file, 'kebc.csv')
    df = pd.read_csv('{}/{}'.format(data_dir, metric_file), dtype={'KIC': int})

    merge = pd.merge(class_df, df, on='KIC')

    # TODO: Deal with NaNs
    xx = merge[metric1].values
    xx[~np.isfinite(xx)] = -2

    yy = merge[metric2].values
    yy[~np.isfinite(yy)] = -2

    eb_class = merge['class'].values

    ev_mask = eb_class == 'ev'
    sp_mask = eb_class == 'sp'
    other_mask = ~ev_mask & ~sp_mask

    fig, ax = plt.subplots()
    ax.scatter(xx[other_mask], yy[other_mask], facecolors='grey',
               edgecolors='None', label='Other')
    ax.scatter(xx[ev_mask], yy[ev_mask], facecolors='b', edgecolors='None',
               label='EV')
    ax.scatter(xx[sp_mask], yy[sp_mask], facecolors='r', edgecolors='None',
               label='SP')

    ax.set_xlim(-0.01, 0.35)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Primary Eclipse Width (Phase)')
    ax.set_ylabel('Median Phase-folded Correlation')
    ax.minorticks_on()

    ax.legend(loc='lower right', scatterpoints=1, markerscale=4)

    if plot_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        plot_file = 'class_metric.{}.pdf'.format(today)

    plt.savefig('{}/{}'.format(data_dir, plot_file))


def sync_vs_e_cos_w(class_file):
    """
    Plot P_orb / P_rot vs an approximate value for e cos(omega)

    Parameters
    ----------
    class_file : str
        Pickle file containing the classifications and rotation periods.
    """
    df = get_classification_results(class_file, 'kebc.csv')

    fig, ax = plt.subplots()

    df_sp = df[df['class'] == 'sp']

    p_orb_p_rot = df_sp['period_x'] / df_sp['p_rot_1']

    e_cos_w = np.pi / 2 * (df_sp['sep'] - 0.5)

    ax.scatter(np.abs(e_cos_w), p_orb_p_rot, facecolors='r', edgecolors='None',
               s=15)

    ax.set_xlabel('$|e\cos{\omega}|$')
    ax.set_ylabel('$P_{orb}/P_{rot}$')
    ax.minorticks_on()
    ax.set_xlim(-0.01, 0.8)
    ax.set_ylim(0, 3)

    plt.savefig('{}/sync_vs_e_cos_w.pdf'.format(data_dir))


def sync_vs_t_eff(class_file, source='kic'):
    """
    Plot P_orb / P_rot and P_orb vs T_eff.

    Parameters
    ----------
    class_file : str
        Pickle file containing the classifications and rotation period
    source : {'kic', 'pin', 'cas', 'arm'}
        Source of the effective temperatures.
        KIC, Pinsonneault (2012), Casagrande (2010), Armstrong (2014)
    """
    df = get_classification_results(class_file, 'kebc.csv')

    datafile = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'data/armstrong14.tsv'))

    arm14 = pd.read_csv(datafile, delim_whitespace=True)

    merge = pd.merge(df, arm14, on='KIC')

    df_sp = merge[merge['class'] == 'sp']
    df_ev = merge[merge['class'] == 'ev']

    t_eff_columns = {'kic': 'Teff', 'pin': 'Teff(Pinsonneault)',
                     'cas': 'Teff(Casagrande)', 'arm': 'T1'}

    # Plot P_orb / P_rot vs. T_eff
    fig, ax1 = plt.subplots()

    ax1.scatter(df_sp[t_eff_columns[source]],
                df_sp['period_x'] / df_sp['p_rot_1'], facecolors='r',
                edgecolors='None', alpha=0.6, s=10, label='SP')

    ax1.scatter(df_ev[t_eff_columns[source]],
                df_ev['period_x'] / df_ev['p_rot_1'], facecolors='b',
                alpha=0.6, edgecolors='None', s=10, label='EV')

    ax1.set_xlabel('$T_{eff}$ (K)')
    ax1.set_ylabel('$P_{orb}/P_{rot}$')
    ax1.minorticks_on()
    ax1.set_xlim(3000, 8000)
    ax1.set_ylim(0, 3)
    ax1.legend(loc='upper right', scatterpoints=1, markerscale=4)

    fig.savefig('{}/sync-t_eff.pdf'.format(data_dir))

    # Plot period vs. T_eff
    fig2, ax2 = plt.subplots()

    ax2.scatter(df_ev[t_eff_columns[source]], df_ev['period_x'], color='b',
                s=10, alpha=0.6, label='EV', edgecolors='None')
    ax2.scatter(df_sp[t_eff_columns[source]], df_sp['period_x'], color='r',
                s=10, alpha=0.6, label='SP', edgecolors='None')

    ax2.set_xlabel('$T_{eff}$ (K)')
    ax2.set_ylabel('$P_{orb}$ (days)')
    ax2.minorticks_on()
    ax2.set_xlim(3000, 10000)
    ax2.set_ylim(1e-1, 1e3)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', scatterpoints=1, markerscale=4)

    fig2.savefig('{}/p_orb-t_eff.pdf'.format(data_dir))
