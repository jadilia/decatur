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
