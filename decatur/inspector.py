#!/usr/bin/env python
# encoding: utf-8
"""
Visually inspect light curves and periodograms
"""

from __future__ import print_function, division, absolute_import

import os

from builtins import input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

from . import config, eclipsing_binary, utils


def _create_results_file(merge):
    """
    Return the relevant columns of the merged rotation period and EB catalog.

    Parameters
    ----------
    merge : pandas DataFrame
        The merged DataFrame

    Returns
    -------
    results : pandas DataFrame
        Relevant columns for light curve inspection.
    """
    results = merge.loc[:, ['KIC', 'period', 'p_rot_1', 'peak_power_1',
                            'peak_1', 'peak_max']]
    results.loc[:, 'class'] = np.repeat('-1', len(results))
    results.loc[:, 'p_rot_alt'] = np.repeat([-1.], len(results))

    return results


class InspectorGadget(object):
    """
    Inspect light curves and periodograms

    Results will be saved in a pickled pandas DataFrame

    Parameters
    ----------
    pgram_results : str
        Pickle file containing the rotation periods from periodograms.
    acf_results : str
        Pickle file containing the rotation periods from ACFs.
    pgram_file : str
        Name of the HDF5 file containing the periodograms.
    results_file : str
        Specify an alternate pickle file for the inspection results.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    sort_on : str, optional
        Sort by a column in the KEBC. Must be a valid column name.
    from_db : bool, optional
        Set to False to load data from MAST instead of local database.
    zoom_pan : float, optional
        Factor by which to zoom and pan light curve plot.
    pgram_on : bool, optional
        Set to False to turn periodogram plot off.
    acf_on : bool, optional
        Set to False to turn ACF plot off.
    """
    def __init__(self, pgram_results, acf_results, pgram_file, acf_file,
                 results_file='inspect.pkl', catalog_file='kebc.csv',
                 sort_on='KIC', from_db=True, zoom_pan=0.05, pgram_on=True,
                 acf_on=True):
        self.catalog_file = catalog_file
        self.from_db = from_db
        self.zoom_pan = zoom_pan
        self.pgram_on = pgram_on
        self.acf_on = acf_on
        self.subplot_list = ['1', '2', '3']

        merge = utils.merge_catalogs(catalog_file, pgram_results, acf_results)

        self.results_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                         'data', results_file))
        if os.path.exists(self.results_file):
            self.results = pd.read_pickle(self.results_file)
        else:
            self.results = _create_results_file(merge)
            self.results.to_pickle(self.results_file)

        if sort_on[-2:] == '_r':
            # Reverse sort
            sort_on = sort_on[:-2]
            self.sort_indices = np.argsort(self.results[sort_on].values)[::-1]
        else:
            self.sort_indices = np.argsort(self.results[sort_on].values)

        # Find the last classified light target
        classified = self.results['class'] != '-1'
        classified = self.results[sort_on][self.sort_indices][classified]
        self.start_index = classified.index[-1]

        # Load the periodograms
        self.h5 = h5py.File('{}/{}'.format(config.data_dir, pgram_file))

        self.h5_acf = h5py.File('{}/{}'.format(config.data_dir, acf_file))

        self.times = [0]
        self.fluxes = [0]

        self.periods = [0]
        self.powers = [0]

        self.lags = [0]
        self.acf = [0]

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.fig_number = None

        self.light_curve = None
        self.periodogram = None
        self.acf_plot = None

        self.p_orb_line_2 = None
        self.p_rot_line_2 = None
        self.p_orb_line_3 = None
        self.p_rot_line_3 = None
        self.peak_1_line = None

        self.zoom_pan_axis = 1

    def _setup(self):
        """
        Setup the plot
        """
        plt.ion()

        if self.pgram_on and not self.acf_on:
            self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2,
                                                          figsize=(7, 12))
        elif not self.pgram_on and self.acf_on:
            self.fig, (self.ax1, self.ax3) = plt.subplots(nrows=2,
                                                          figsize=(7, 12))
        elif not self.pgram_on and not self.acf_on:
            self.fig, self.ax1 = plt.subplots(nrows=1,
                                              figsize=(7, 12))
        else:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(nrows=3,
                                                                    figsize=(7, 12))
        # Use this later to check if the window has been closed.
        self.fig_number = self.fig.number

        # Setup the light curve plot
        self.light_curve, = self.ax1.plot(self.times, self.fluxes, color='k')
        self.ax1.set_xlabel('Time (days)')
        self.ax1.set_ylabel('Relative Flux')

        if self.pgram_on:
            # Set up the periodgram plot
            self.periodogram, = self.ax2.plot(self.periods, self.powers,
                                              color='k')
            self.ax2.set_xlim(0, 45)
            self.ax2.set_xlabel('Period (days)')
            self.ax2.set_ylabel('Normalized Power')

            # Vertical lines at the measured rotation period and orbital period
            self.p_rot_line_2 = self.ax2.axvline(0, color='r')
            self.p_orb_line_2 = self.ax2.axvline(0, color='b')
        else:
            self.subplot_list.remove('2')

        if self.acf_on:
            # Setup the ACF plot
            self.acf_plot, = self.ax3.plot(self.lags, self.acf, color='k')
            self.ax3.set_xlim(0, 45)
            self.ax3.set_ylim(-1, 1)
            self.ax3.set_xlabel('Lag (days)')
            self.ax3.set_ylabel('ACF')

            # Vertical lines a orbital period, first peak, and highest peak
            self.p_rot_line_3 = self.ax3.axvline(0, color='r')
            self.peak_1_line = self.ax3.axvline(0, color='g')
            self.p_orb_line_3 = self.ax3.axvline(0, color='b')
        else:
            self.subplot_list.remove('3')

    def _print_kic_stats(self, index):
        """
        Print the statistics for a given KIC system

        Parameters
        ----------
        index : int
            Index of the system in the results DataFrame.
        """
        print('\nKIC {}'.format(self.results['KIC'][index]))
        print('-----------------------------------')
        print('P_orb    P_rot     class  p_rot_alt')
        print('-----------------------------------')
        header = '{:>6.2f}  {:>5.2f}  {:>10s}  {:>5.2f} \n'
        print(header.format(self.results['period'][index],
                            self.results['p_rot_1'][index],
                            self.results['class'][index],
                            self.results['p_rot_alt'][index]))

    def _update(self, index):
        """
        Update the plots for a different system.

        Parameters
        ----------
        index : int
            Index of the system in the results DataFrame.
        """
        if not plt.fignum_exists(self.fig_number):
            self._setup()

        self.zoom_pan_axis = 1

        kic = self.results['KIC'][index]

        eb = eclipsing_binary.EclipsingBinary.from_kic(kic,
                                                       catalog_file=self.catalog_file,
                                                       from_db=self.from_db)
        eb.detrend_and_normalize()

        self.light_curve.set_xdata(eb.l_curve.times)
        self.light_curve.set_ydata(eb.l_curve.fluxes)

        time_min = eb.l_curve.times.min()
        self.ax1.set_xlim(time_min, time_min + 45.)

        flux_max = np.percentile(np.abs(eb.l_curve.fluxes), 98)
        self.ax1.set_ylim(-flux_max, flux_max)

        if self.pgram_on:
            periods = self.h5['{}/periods'.format(kic)][:]
            powers = self.h5['{}/powers'.format(kic)][:]
            self.periodogram.set_xdata(periods)
            self.periodogram.set_ydata(powers)

            self.ax2.set_xlim(0, 45)
            self.ax2.set_ylim(0, 1.1 * powers.max())

            self.p_rot_line_2.set_xdata(self.results['p_rot_1'][index])
            self.p_orb_line_2.set_xdata(self.results['period'][index])

        if self.acf_on:
            lags = self.h5_acf['{}/lags'.format(kic)][:]
            acf = self.h5_acf['{}/acf'.format(kic)][:]
            self.acf_plot.set_xdata(lags)
            self.acf_plot.set_ydata(acf / acf.max())
            self.ax3.set_xlim(0, 4 * self.results['period'][index])

            self.peak_1_line.set_xdata(self.results['peak_1'][index])
            self.p_rot_line_3.set_xdata(self.results['peak_max'][index])
            self.p_orb_line_3.set_xdata(self.results['period'][index])

        self.fig.canvas.draw()

        self._print_kic_stats(index)

    def _classify(self, index):
        """
        User classification of the light curve.

        Parameters
        ----------
        index : int
            Index of the system in the results DataFrame
        """
        user_class = input('\nType of out-of-eclipse variability: ').lower()
        self.results.loc[index, 'class'] = str(user_class)

        alternate_p_rot = input('Alternate rotation period?: ').lower()

        if alternate_p_rot == 'y':
            user_p_rot = input('Rotation period: ')
            self.results.loc[index, 'p_rot_alt'] = float(user_p_rot)

        self.results.to_pickle(self.results_file)

    def _key_press(self, event):
        """
        Handle zooming and panning.
        """
        # Change axis for zoom-pan.
        if event.key in self.subplot_list:
            self.zoom_pan_axis = int(event.key)
            return

        ax = [self.ax1, self.ax2, self.ax3][self.zoom_pan_axis - 1]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_width = x1 - x0
        y_width = y1 - y0
        zp = self.zoom_pan

        if event.key == 'left':
            ax.set_xlim(x0 - zp * x_width, x1 - zp * x_width)
        elif event.key == 'right':
            ax.set_xlim(x0 + zp * x_width, x1 + zp * x_width)
        elif event.key == 'ctrl+left':
            ax.set_xlim(x0 - zp * x_width, x1 + zp * x_width)
        elif event.key == 'ctrl+right':
            ax.set_xlim(x0 + zp * x_width, x1 - zp * x_width)

        elif event.key == 'up':
            ax.set_ylim(y0 + zp * y_width, y1 + zp * y_width)
        elif event.key == 'down':
            ax.set_ylim(y0 - zp * y_width, y1 - zp * y_width)
        elif event.key == 'ctrl+up':
            ax.set_ylim(y0 + zp * y_width, y1 - zp * y_width)
        elif event.key == 'ctrl+down':
            ax.set_ylim(y0 - zp * y_width, y1 + zp * y_width)

    def gogo_gadget(self):
        """
        Launch the inspection program.
        """
        self._setup()

        self.fig.canvas.mpl_connect('key_press_event', self._key_press)

        ii = np.where(self.sort_indices == self.start_index)[0][0]
        self._update(self.start_index)

        while True:

            user_input = input('--> ').lower()

            if user_input == '':
                continue
            elif user_input == 'n':
                if ii + 2 > len(self.results['KIC']):
                    print('Reached end of catalog')
                else:
                    ii += 1
                    index = self.sort_indices[ii]
                    self._update(index)

            elif user_input == 'p':
                if ii - 1 < 0:
                    print('At beginning of catalog')
                else:
                    ii -= 1
                    index = self.sort_indices[ii]
                    self._update(index)

            elif user_input == 'q':
                break

            elif utils.is_int(user_input):
                try:
                    index = np.where(int(user_input) == self.results['KIC'])[0][0]
                    ii = np.where(self.sort_indices == index)[0][0]
                except IndexError:
                    print('Invalid KIC')
                    continue
                self._update(index)

            elif user_input == 'c':
                self._classify(index)
                self._print_kic_stats(index)

            elif user_input == 's':
                n_classed = np.sum(self.results['class'] != '-1')
                print('\n{} targets classified\n'.format(n_classed))

            else:
                print('Input not understood')

        plt.close()
