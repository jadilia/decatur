#!/usr/bin/env python
# encoding: utf-8
"""
Visually inspect light curves and periodograms
"""

from __future__ import print_function, division, absolute_import

from builtins import input
import matplotlib.pyplot as plt
import numpy as np
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
    pgram_file, acf_file : str
        Names of the HDF5 file containing the periodograms and ACFs.
    kic_list : list of int, optional
        Only display targets on this list.
    results_file : str, optional
        Specify an alternate HDF5 file for the inspection results.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    sort_on : str, optional
        Sort by a column in the KEBC. Must be a valid column name.
    class_filter : str, optional
        If not None, only light curves of this class will be shown.
    from_db : bool, optional
        Set to False to load data from MAST instead of local database.
    zoom_pan : float, optional
        Factor by which to zoom and pan light curve plot.
    pgram_on : bool, optional
        Set to False to turn periodogram plot off.
    acf_on : bool, optional
        Set to False to turn ACF plot off.
    phase_fold_on : bool, optional
        Set to True to show window with phase_folded light curve.
    use_pdc : bool, optional
        Set to False to use SAP instead of PDC flux.
    """
    def __init__(self, pgram_file, acf_file, kic_list=None,
                 results_file='inspection_data.h5', catalog_file='kebc.csv',
                 sort_on='kic', class_filter=None, from_db=True, zoom_pan=0.05,
                 pgram_on=True, acf_on=True, phase_fold_on=False,
                 use_pdc=True):
        self.catalog_file = catalog_file
        self.from_db = from_db
        self.zoom_pan = zoom_pan
        self.pgram_on = pgram_on
        self.acf_on = acf_on
        self.subplot_list = ['1', '2', '3']
        self.use_pdc = use_pdc
        self.phase_fold_on = phase_fold_on

        self.results = h5py.File('{}/{}'.format(config.repo_data_dir, results_file),
                                 'r+')

        if kic_list is not None:
            keep = np.in1d(self.results['kic'][:], kic_list)
        else:
            keep = np.repeat(True, len(self.results['class'][:]))

        if class_filter is not None:
            keep2 = self.results['class'][:] == class_filter
            keep &= keep2

        # Reference indicies in ascending order
        ref_indices = np.arange(0, len(keep))

        if sort_on[-2:] == '_r':
            # Reverse sort
            sort_on = sort_on[:-2]
            sort_arr = np.argsort(self.results[sort_on][:])[::-1]
        else:
            sort_arr = np.argsort(self.results[sort_on][:])

        self.sort_indices = sort_arr[np.in1d(sort_arr, ref_indices[keep])]

        # Find the last classified target.
        classified = self.results['class_v2'][keep] != '-1'
        if np.sum(classified) == 0:
            self.start_index = self.sort_indices[0]
        else:
            self.start_index = self.sort_indices[classified][-1]

        # Load the periodograms
        self.h5 = h5py.File('{}/{}'.format(config.data_dir, pgram_file))

        self.h5_acf = h5py.File('{}/{}'.format(config.data_dir, acf_file))

        self.kebc = utils.load_catalog(self.catalog_file)

        self.times = [0]
        self.fluxes = [0]

        self.periods = [0]
        self.powers = [0]

        self.lags = [0]
        self.acf = [0]

        self.fig = None
        self.fig2 = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.fig_number = None

        self.light_curve = None
        self.periodogram = None
        self.acf_plot = None

        self.p_orb_line_2 = None
        self.p_rot_line_2 = None
        self.p_orb_line_3 = None
        self.p_rot_line_3 = None

        self.zoom_pan_axis = 1

        self.xx = None
        self.yy = None
        self.fig_s = None
        self.ax_s = None
        self.click_point = None

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
            self.fig.subplots_adjust(bottom=0.01, top=0.99, hspace=0.1)

        if self.phase_fold_on:
            self.fig2, self.ax4 = plt.subplots()
            self.phase_fold_plot = self.ax4.scatter([0], [0], color='k', s=0.1)
            self.ax4.set_xlim(-0.1, 1.1)
            self.ax4.set_xlabel('Phase')
            self.ax4.set_ylabel('Relative Flux')

        # Use this later to check if the window has been closed.
        self.fig_number = self.fig.number

        # Setup the light curve plot
        self.light_curve, = self.ax1.step(self.times, self.fluxes, color='k',
                                          lw=0.5, where='mid')
        self.ax1.set_xlabel('Time (days)')
        self.ax1.set_ylabel('Relative Flux')

        if self.pgram_on:
            # Set up the periodgram plot
            self.periodogram, = self.ax2.step(self.periods, self.powers,
                                              color='k', lw=0.5, where='mid')
            self.ax2.set_xlim(0, 45)
            self.ax2.set_xlabel('Period (days)')
            self.ax2.set_ylabel('Normalized Power')

            # Vertical lines at the measured rotation period and orbital period
            self.p_rot_line_2 = self.ax2.axvline(0, color='r')
            self.p_orb_line_2 = self.ax2.axvline(0, color='b')
            self.p_final_line_2 = self.ax2.axvline(0, color='g', lw=2, ls='--')
        else:
            self.subplot_list.remove('2')

        if self.acf_on:
            # Setup the ACF plot
            self.acf_plot, = self.ax3.step(self.lags, self.acf, color='k',
                                           lw=0.5, where='mid')
            self.ax3.set_xlim(0, 45)
            self.ax3.set_ylim(-1, 1)
            self.ax3.set_xlabel('Lag (days)')
            self.ax3.set_ylabel('ACF')

            # Vertical lines a orbital period and highest peak
            self.p_rot_line_3 = self.ax3.axvline(0, color='r')
            self.p_orb_line_3 = self.ax3.axvline(0, color='b')
            self.p_final_line_3 = self.ax3.axvline(0, color='g', lw=2, ls='--')
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
        print('\nKIC {}'.format(self.results['kic'][index]))
        url = 'http://keplerebs.villanova.edu/overview/?k={}'.format(self.results['kic'][index])
        print(url)
        print('-----------------------------------')
        print('P_orb: {:6.2f}'.format(self.results['p_orb'][index]))

        kic_str = self.kebc.loc[self.kebc['KIC'] == self.results['kic'][index]]['kmag']
        print('K_mag: {}'.format(kic_str.values[0]))

        teff_str = self.kebc.loc[self.kebc['KIC'] == self.results['kic'][index]]['Teff']
        print('T_eff: {:.0f}\n'.format(teff_str.values[0]))

        print('{} ({})\n'.format(self.results['class_v2'][index],
                                 self.results['class'][index]))

        print('      P_auto   P_man')
        print('LSP  {:5.2f}    {:5.2f}'.format(self.results['pgram/p_rot_1'][index],
                                               self.results['pgram/p_man'][index]))
        print('ACF  {:5.2f}    {:5.2f}'.format(self.results['acf/p_rot_1'][index],
                                               self.results['acf/p_man'][index]))

        print('\nMulti: {}'.format(self.results['p_multi'][index]))
        print('\nBad data: {}'.format(self.results['bad_data'][index]))
        print()

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

        kic = self.results['kic'][index]

        eb = eclipsing_binary.EclipsingBinary.from_kic(kic,
                                                       catalog_file=self.catalog_file,
                                                       from_db=self.from_db,
                                                       use_pdc=self.use_pdc)
        eb.normalize()

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

            self.p_rot_line_2.set_xdata(self.results['pgram/p_rot_1'][index])
            self.p_orb_line_2.set_xdata(self.results['p_orb'][index])

            if self.results['pgram/p_man'][index] < -0.5:
                self.p_final_line_2.set_xdata(self.results['pgram/p_rot_1'][index])
            else:
                self.p_final_line_2.set_xdata(self.results['pgram/p_man'][index])

        if self.acf_on:
            lags = self.h5_acf['{}/lags'.format(kic)][:]
            acf = self.h5_acf['{}/acf'.format(kic)][:]
            self.acf_plot.set_xdata(lags)
            self.acf_plot.set_ydata(acf / acf.max())
            self.ax3.set_xlim(0, 45)
            self.ax3.set_ylim(-1, 1)

            self.p_rot_line_3.set_xdata(self.results['acf/p_rot_1'][index])
            self.p_orb_line_3.set_xdata(self.results['p_orb'][index])

            if self.results['acf/p_man'][index] < -0.5:
                self.p_final_line_3.set_xdata(self.results['acf/p_rot_1'][index])
            else:
                self.p_final_line_3.set_xdata(self.results['acf/p_man'][index])

        if self.phase_fold_on:
            phase = eb.phase_fold()

            self.phase_fold_plot.set_offsets(np.hstack((phase[:, None],
                                                        eb.l_curve.fluxes[:, None])))
            ymin = -1.1 * np.percentile(-eb.l_curve.fluxes[eb.l_curve.fluxes < 0], 99)
            ymax = 1.5 * np.percentile(eb.l_curve.fluxes[eb.l_curve.fluxes > 0], 99)

            self.ax4.set_ylim(ymin, ymax)

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
        print()

        if user_class not in ['sp', 'sp?', 'ev', 'ot', 'fl']:
            print('Invalid classification type.')
            return

        self.results['class_v2'][index] = str(user_class)

        if user_class == 'sp':
            for metric, name in zip(['pgram', 'acf'], ['Periodogram', 'ACF']):
                good = input('{} period correct?: '.format(name)).lower()

                if good == 'y':
                    self.results['{}/p_man'.format(metric)][index] = -1
                elif good == 'n':
                    p_man = input('Alternate {} period: '.format(name))
                    self.results['{}/p_man'.format(metric)][index] = float(p_man)
                print()

            multi = input('Multiple possible periods?: ').lower()
            if multi == 'y':
                self.results['p_multi'][index] = 'y'
            elif multi == 'n':
                self.results['p_multi'][index] = 'n'

            bad_data = input('Bad light curve segments?: ').lower()
            if bad_data == 'y':
                self.results['bad_data'][index] = 1
            elif bad_data == 'n':
                self.results['bad_data'][index] = 0

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
        index = self.start_index

        total_systems = len(self.sort_indices)

        while True:

            user_input = input('--> ').lower()

            if user_input == '':
                continue
            elif user_input == 'n':
                if ii + 2 > total_systems:
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
                    index = np.where(int(user_input) == self.results['kic'][:])[0][0]
                    ii = np.where(self.sort_indices == index)[0][0]
                except IndexError:
                    print('Invalid KIC')
                    continue
                self._update(index)

            elif user_input == 'c':
                self._classify(index)
                self._print_kic_stats(index)

            elif user_input == 's':
                n_classed = np.sum(self.results['class_v2'][:] != '-1')
                n_total = len(self.sort_indices)
                print('\n{}/{} targets classified\n'.format(n_classed,
                                                            n_total))

            else:
                print('Input not understood')

        plt.close()

    def p_orb_p_rot(self):
        """
        Display light curves by selection on the P_orb/P_rot diagram.
        """
        self._setup()
        self._update(self.start_index)

        self.xx = self.results['p_orb'][:]
        self.yy = self.results['p_orb'][:] / self.results['acf/p_rot_1'][:]

        self.xx[~np.isfinite(self.xx)] = 0
        self.yy[~np.isfinite(self.yy)] = 0

        sp_mask = self.results['class'][:] == 'sp'
        self.xx[~sp_mask] = -9
        self.yy[~sp_mask] = -9

        # Plot P_orb/P_rot vs. P_orb
        self.fig_s, self.ax_s = plt.subplots()

        self.ax_s.scatter(self.xx, self.yy, color='r')

        for line in [0.5, 1, 2]:
            self.ax_s.axhline(line, color='k', ls=':')

        self.ax_s.set_xlabel('$P_{orb} (days)$')
        self.ax_s.set_ylabel('$P_{orb}/P_{rot}$')
        self.ax_s.set_xlim(0.1, 1000)
        self.ax_s.set_ylim(0.01, 1000)
        self.ax_s.set_xscale('log')
        self.ax_s.set_yscale('log')

        self.click_point, = self.ax_s.plot([0], [0], 'ok', zorder=2)

        def on_click(event):

            if event.dblclick:

                ix = event.xdata
                iy = event.ydata
                inaxes = event.inaxes
                if inaxes is not self.ax_s:
                    return

                closest = np.sqrt((self.xx - ix) ** 2 +
                                  (self.yy - iy) ** 2).argmin()

                self.click_point.set_xdata(self.xx[closest])
                self.click_point.set_ydata(self.yy[closest])

                self._update(closest)

        self.fig_s.canvas.mpl_connect('button_press_event', on_click)

        plt.show(block=True)
