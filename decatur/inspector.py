#!/usr/bin/env python
# encoding: utf-8
"""
Visually inspect light curves and periodograms
"""

from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import h5py

from . import config, eclipsing_binary, utils


class InspectorGadget(object):
    """
    Inspect light curves and periodograms

    Parameters
    ----------
    p_rot_file : str
        Name of the pickle file containing the rotation periods.
    periodograms_file : str
        Name of the HDF5 file containing the periodograms.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    """
    def __init__(self, p_rot_file, periodograms_file, catalog_file='kebc.csv'):
        self.p_rot_file = p_rot_file
        self.catalog_file = catalog_file

        merge = utils.merge_catalogs(catalog_file, p_rot_file)
        self.kics = merge['KIC']
        self.p_orbs = merge['period']
        self.p_rots = merge['p_rot_1']

        # Load the periodograms
        self.h5 = h5py.File('{}/{}'.format(config.data_dir, periodograms_file))

        self.times = [0]
        self.fluxes = [0]

        self.periods = [0]
        self.powers = [0]

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.fig_number = None

        self.light_curve = None
        self.periodogram = None
        self.p_rot_line = None

    def _setup(self):
        """
        Setup the plot
        """
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, figsize=(7, 12))

        # Use this later to check if the window has been closed.
        self.fig_number = self.fig.number

        # Setup the light curve plot
        self.light_curve, = self.ax1.plot(self.times, self.fluxes, color='k')
        self.ax1.set_xlabel('Time (days)')
        self.ax1.set_ylabel('Relative Flux')

        # Set up the periodgram plot
        self.periodogram, = self.ax2.plot(self.periods, self.powers, color='k')
        self.ax2.set_xlim(0, 45)
        self.ax2.set_xlabel('Period (days)')
        self.ax2.set_ylabel('Normalized Power')

        # Vertical line at the measured rotation period
        self.p_rot_line = self.ax2.axvline(0, color='b')

    def _print_kic_stats(self, index):
        """
        Print the statistics for a given KIC system

        Parameters
        ----------
        index : int
            Index of the system in the results DataFrame.
        """
        print('\nKIC {}'.format(self.kics[index]))
        print('P_orb  P_rot')
        print('----------------')
        print('{:>5.2f} {:>5.2f}\n'.format(self.p_orbs[index],
                                           self.p_rots[index]))

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

        eb = eclipsing_binary.EclipsingBinary.from_kic(self.kics[index],
                                                       catalog_file=self.catalog_file)
        eb.detrend_and_normalize()

        self.light_curve.set_xdata(eb.l_curve.times)
        self.light_curve.set_ydata(eb.l_curve.fluxes)

        time_min = eb.l_curve.times.min()
        self.ax1.set_xlim(time_min, time_min + 45.)

        flux_max = np.percentile(np.abs(eb.l_curve.fluxes), 98)
        self.ax1.set_ylim(-flux_max, flux_max)

        periods = self.h5['{}/periods'.format(self.kics[index])][:]
        powers = self.h5['{}/powers'.format(self.kics[index])][:]
        self.periodogram.set_xdata(periods)
        self.periodogram.set_ydata(powers)

        self.ax2.set_ylim(0, 1.1 * powers.max())

        self.p_rot_line.set_xdata(self.p_rots[index])

        self.fig.canvas.draw()

        self._print_kic_stats(index)

    def gogo_gadget(self):
        """
        Launch the inspection program.
        """
        self._setup()

        ii = 0
        self._update(ii)

        while True:

            user_input = raw_input('--> ').lower()

            if user_input == '':
                continue
            elif user_input == 'n':
                if ii + 2 > len(self.kics):
                    print('Reached end of catalog')
                else:
                    ii += 1
                    self._update(ii)

            elif user_input == 'p':
                if ii - 1 < 0:
                    print('At beginning of catalog')
                else:
                    ii -= 1
                    self._update(ii)

            elif user_input == 'q':
                break

            elif utils.is_int(user_input):
                try:
                    index = np.where(int(user_input) == self.kics)[0][0]
                    ii = index
                except IndexError:
                    print('Invalid KIC')
                    continue
                self._update(index)

            else:
                print('Input not understood')

        plt.close()
