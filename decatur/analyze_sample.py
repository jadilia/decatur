#!/usr/bin/env python
# encoding: utf-8
"""
Analyze the eclipsing binary sample.
"""

from __future__ import print_function, division, absolute_import

import sys
import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, interpolate

from . import eclipsing_binary, kepler_data, exceptions, utils
from .config import data_dir


def compute_periodograms(width_max=0.25, period_min=0.01, period_max=100.,
                         oversampling=2, output_file=None,
                         catalog_file='kebc.csv'):
    """
    Compute periodograms for the Kepler eclipsing binary sample and
    store as an HDF5 file.

    Parameters
    ----------
    width_max : float, optional
        Eclipses will be interpolated over if the primary phase width
        is less than `width_max`.
    period_min, period_max : float, optional
        Results will only contain `period_min` < period < `period_max`
    oversampling : int, optional
        Oversampling factor for the periodogram. Greater oversampling
        will result in larger output files.
    output_file : str, optional
        Specify an alternate output results filename.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    """
    if output_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        output_file = 'periodograms.{}.h5'.format(today)

    kics = kepler_data.select_kics(catalog_file=catalog_file)

    h5 = h5py.File('{}/{}'.format(data_dir, output_file), 'w')
    h5.attrs['width_max'] = width_max

    total_systems = len(kics)
    print('Computing periodograms for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):
        try:
            eb = eclipsing_binary.EclipsingBinary.from_kic(kic)
        except exceptions.CatalogMatchError:
            continue

        eb.detrend_and_normalize()

        if eb.params.width_pri < width_max:
            eb.interpolate_over_eclipse()

        eb.run_periodogram(oversampling=oversampling)

        keep = (eb.periods > period_min) & (eb.periods < period_max)

        group = h5.create_group(str(kic))
        group.create_dataset('periods', data=eb.periods[keep])
        group.create_dataset('powers', data=eb.powers[keep])

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    h5.close()


def measure_rotation_periods(periodograms_file, results_file=None,
                             period_min=0.01, period_max=100.):
    """
    Measure rotation periods for the Kepler eclipsing binary sample.

    Parameters
    ----------
    periodograms_file : str
        HD5F file containing the periodograms.
    results_file : str, optional
        Specify an alternate output results filename.
    period_min, period_max : float, optional
        Will only search for `period_min` < period < `period_max`
    """
    h5 = h5py.File('{}/{}'.format(data_dir, periodograms_file), 'r')

    kics = np.array(h5.keys(), dtype=np.int64)

    dtypes = [('KIC', np.uint64), ('p_rot_1', np.float64),
              ('peak_power_1', np.float64), ('cross_corr_1', np.float64)]
    rec_array = np.recarray(len(kics), dtype=dtypes)

    total_systems = len(kics)
    print('Measuring rotation periods for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        periods = h5['{}/periods'.format(kic)][:]
        powers = h5['{}/powers'.format(kic)][:]

        keep = (periods > period_min) & (periods < period_max)

        index_max = np.argmax(powers[keep])

        p_rot_1 = periods[keep][index_max]

        eb = eclipsing_binary.EclipsingBinary.from_kic(kic)
        corr = phase_correlation(eb.l_curve.times, eb.l_curve.fluxes,
                                 p_fold=p_rot_1, t_0=eb.params.bjd_0)[1]
        cross_corr_1 = np.nanmedian(corr)

        rec_array[ii]['KIC'] = kic
        rec_array[ii]['p_rot_1'] = p_rot_1
        rec_array[ii]['peak_power_1'] = powers[keep][index_max]
        rec_array[ii]['cross_corr_1'] = cross_corr_1

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if results_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        results_file = 'rotation_periods.{}.pkl'.format(today)

    df = pd.DataFrame(data=rec_array)
    df.to_pickle('{}/{}'.format(data_dir, results_file))


def plot_prot_porb(p_rot_file, plot_file=None, catalog_file='kebc.csv'):
    """
    Plot P_orb / P_rot vs. P_orb.

    Parameters
    ----------
    p_rot_file : str
        Pickle file containing the rotation periods.
    plot_file : str, optional
        Specify an alternate output plot file.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    """
    df = pd.read_pickle('{}/{}'.format(data_dir, p_rot_file))
    kebc = utils.load_catalog(catalog_file)
    join = pd.merge(kebc, df, on='KIC')

    fig, ax = plt.subplots()

    scatter = ax.scatter(join['period'], join['period'] / join['p_rot_1'],
                         c=join['peak_power_1'], cmap='viridis_r',
                         vmin=0, vmax=1)

    ax.set_xlim(0, 45)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Orbital Period')
    ax.set_ylabel('Orbital Period $\div$ Rotation Period')
    ax.minorticks_on()

    cbar = fig.colorbar(scatter)
    cbar.ax.set_ylabel('Normalized Periodogram Power')

    if plot_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        plot_file = 'rotation_periods.{}.pdf'.format(today)

    plt.savefig('{}/{}'.format(data_dir, plot_file))


def phase_correlation(times, fluxes, p_fold, t_0=0., delta_phase=0.01,
                      cad_min=3, plot=False):
    """
    The cross-correlation with the median phase-folded light curve.

    For a given period, compute the binned, median phase-folded light curve.
    Then compute the cross-correlation with each successive cycle of light
    curve at that period.

    Parameters
    ----------
    times : array_like
        Observation times
    fluxes : array_like
        Fluxes
    p_fold: float
        The period at which to fold the light curve.
    t_0: float, optional
        The reference time, e.g., time of primary eclipse. Default: 0.
    delta_phase: float, optional
        The phase bin width. Default: 0.01
    cad_min: int, optional
        Exclude light curve sections with fewer cadences than `cad_min`.
    plot : bool, optional
        Set to True to plot phase-folded light curve and cross-correlation

    Returns
    -------
    cycle_num : numpy.ndarray
        Cycle number.
    corr : numpy.ndarray
        The cross-correlation
    """
    # Calculate the phase.
    phase = ((times - t_0) % p_fold) / p_fold
    # Calculate the cycle number.
    cycle = ((times - t_0) // p_fold).astype(int)
    # Start at zero
    cycle -= cycle.min()

    # Computed binned median
    bins = np.arange(0., 1. + delta_phase, delta_phase)
    binned_med, bin_edges = stats.binned_statistic(phase, fluxes,
                                                   statistic="median",
                                                   bins=bins)[:2]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Wrap around at the beginning and end of the binned light curve
    # using the fact that f(phase) = f(phase + 1)
    wrap = ([bin_centers[-1] - 1.], bin_centers, [1. + bin_centers[0]])
    bin_centers = np.concatenate(wrap)
    wrap = ([binned_med[-1]], binned_med, [binned_med[0]])
    binned_med = np.concatenate(wrap)

    # Linear interpolation of binned light curve.
    interp = interpolate.interp1d(bin_centers, binned_med)

    # Only use cycles with more cadences than `cad_min`.
    cycle_num = np.arange(cycle.max() + 1)[np.bincount(cycle) > cad_min]

    # Empty array to hold cross-correlation
    corr = np.zeros_like(cycle_num, dtype=float)

    for i, n in enumerate(cycle_num):

        mask = cycle == n

        p = phase[mask]
        f = fluxes[mask]
        f_i = interp(p)

        # Normalize light curves
        a = (f - np.mean(f)) / (np.std(f) * len(f))
        v = (f_i - np.mean(f_i)) / np.std(f_i)

        corr[i] = np.correlate(a, v)

    if plot:

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

        ax1.scatter(phase, fluxes, color="k", s=0.1)
        p = np.linspace(0, 1, 1000)
        ax1.plot(p, interp(p), color="r")

        ax1.set_xlim(0, 1)
        ymax = np.percentile(np.abs(fluxes), 98)
        ax1.set_ylim(-ymax, ymax)
        ax1.set_xlabel("Phase")
        ax1.set_ylabel("Normalized Flux")

        ax2.plot(cycle_num * p_fold, corr, "-k", markersize=3)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Normalized Cross-correlation")

        plt.show()

    return cycle_num, corr
