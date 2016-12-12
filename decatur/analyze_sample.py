#!/usr/bin/env python
# encoding: utf-8
"""
Analyze the eclipsing binary sample.
"""

from __future__ import print_function, division, absolute_import

import sys
import datetime

import h5py
import interpacf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, interpolate

from . import eclipsing_binary, kepler_data, utils
from .config import data_dir


def compute_periodicity(kind, width_max=0.25, period_min=0.01, period_max=100.,
                        window=1., oversampling=2, output_file=None,
                        catalog_file='kebc.csv', from_db=True):
    """
    Compute periodograms for the Kepler eclipsing binary sample and
    store as an HDF5 file.

    Parameters
    ----------
    kind : {'periodogram', 'acf'}
        The kind of periodicity metric to compute.
    width_max : float, optional
        Eclipses will be interpolated over if the primary phase width
        is less than `width_max`.
    period_min, period_max : float, optional
        Results will only contain `period_min` < period < `period_max`
    window : float, optional
        The width (in phase) to interpolate over the eclipses.
    oversampling : int, optional
        Oversampling factor for the periodogram. Greater oversampling
        will result in larger output files.
    output_file : str, optional
        Specify an alternate output results filename.
    catalog_file : str, optional
        Specify an alternate eclipsing binary catalog filename.
    from_db : bool, optional
        Set to False to download data from MAST instead of local database.
    """
    if kind == 'periodogram':
        x_var_name = 'periods'
        y_var_name = 'powers'
    elif kind == 'acf':
        x_var_name = 'lags'
        y_var_name = 'acf'
    else:
        raise ValueError('Invalid choice of metric `kind`: {}'.format(kind))

    if output_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        output_file = '{}s.{}.h5'.format(kind, today)

    kics = kepler_data.select_kics(catalog_file=catalog_file)

    # Avoid analyzing the same system twice.
    kics = np.unique(kics)

    h5 = h5py.File('{}/{}'.format(data_dir, output_file), 'w')
    h5.attrs['width_max'] = width_max

    total_systems = len(kics)
    print('Computing {}s for {} systems...'.format(kind, total_systems))

    for ii, kic in enumerate(kics):
        eb = eclipsing_binary.EclipsingBinary.from_kic(kic, from_db=from_db)

        eb.detrend_and_normalize()

        if eb.params.width_pri < width_max:
            eb.interpolate_over_eclipse(window=window)

        if kind == 'periodogram':
            eb.run_periodogram(oversampling=oversampling)
            keep = (eb.periods > period_min) & (eb.periods < period_max)
            x_var = eb.periods[keep]
            y_var = eb.powers[keep]

        elif kind == 'acf':
            eb.run_acf()
            keep = eb.lags < period_max
            x_var = eb.lags[keep]
            y_var = eb.acf[keep]

        group = h5.create_group(str(kic))
        group.create_dataset(x_var_name, data=x_var)
        group.create_dataset(y_var_name, data=y_var)

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


def phase_folded_median(phase, fluxes, delta_phase):
    """
    Compute the phase-folded median light curve.

    Parameters
    ----------
    phase : array_like
        The phase of observations.
    fluxes : array_like
        Observed fluxes
    delta_phase : float
        The phase bin width.

    Returns
    -------
    interp : scipy.interp1d object
        Linear interpolation of the phase-folded light curve.
    """
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

    return interp


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

    interp = phase_folded_median(phase, fluxes, delta_phase)

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


def correlation_at_p_orb(width_max=0.25, savefile=None):
    """
    Compute the cross correlation with the median light curve folded at the
    orbital period.

    Parameters
    ----------
    width_max : float, optional
        Eclipses will be interpolated over if the primary phase width
        is less than `width_max`.
    savefile : str, optional
        Specify an alternate output file.
    """
    kebc = utils.load_catalog()
    kics = kebc['KIC']

    correlations = np.zeros(len(kebc), dtype=float) - 1.

    total_systems = len(kics)
    print('Measuring phase correlation for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        eb = eclipsing_binary.EclipsingBinary.from_kic(kic)
        eb.detrend_and_normalize()

        if eb.params.width_pri < width_max:
            eb.interpolate_over_eclipse()

        corr = phase_correlation(eb.l_curve.times, eb.l_curve.fluxes,
                                 eb.params.p_orb, t_0=eb.params.bjd_0)[1]

        correlations[ii] = np.nanmedian(corr)

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if savefile is None:
        savefile = 'corr_at_p_orb.csv'

    dtypes = [('KIC', np.uint64), ('corr', np.float32)]
    rec_array = np.recarray(len(kics), dtype=dtypes)
    rec_array['KIC'] = kics
    rec_array['corr'] = correlations

    df = pd.DataFrame(rec_array)
    df.to_csv('{}/{}'.format(data_dir, savefile))


def find_acf_peaks(acf_file, results_file=None):
    """
    Find peaks in the auto-correlation function (ACF).

    Results are saved as a pandas DataFrame in a pickle file.

    Parameters
    ----------
    acf_file : str
        HDF5 file containing the ACFs.
    results_file : str, optional
        Specify an alternate output results filename.
    """
    h5 = h5py.File('{}/{}'.format(data_dir, acf_file), 'r')

    kics = np.array(h5.keys(), dtype=np.int64)

    dtypes = [('KIC', np.uint64), ('peak_1', np.float64),
              ('peak_2', np.float64), ('peak_3', np.float64),
              ('peak_4', np.float64), ('peak_max', np.float64),
              ('max_height', np.float64)]
    rec_array = np.recarray(len(kics), dtype=dtypes)

    total_systems = len(kics)
    print('Finding ACF peaks for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        lags = h5['{}/lags'.format(kic)][:]
        acf = h5['{}/acf'.format(kic)][:]

        peak_max, peaks, h_p = interpacf.dominant_period(lags, acf)

        rec_array[ii]['KIC'] = kic
        rec_array[ii]['peak_max'] = peak_max
        rec_array[ii]['max_height'] = h_p

        for jj in range(len(peaks))[:4]:
            rec_array[ii]['peak_{}'.format(jj + 1)] = peaks[jj]

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if results_file is None:
        today = '{:%Y%m%d}'.format(datetime.date.today())
        results_file = 'acf_peaks.{}.pkl'.format(today)

    df = pd.DataFrame(data=rec_array)
    df.to_pickle('{}/{}'.format(data_dir, results_file))
