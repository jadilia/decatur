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

from . import eclipsing_binary, kepler_data
from .config import data_dir, repo_data_dir


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

        eb.normalize()

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


def measure_rotation_periods(periodograms_file,
                             class_datafile='inspection_data.h5',
                             period_min=0.01, period_max=100.):
    """
    Measure rotation periods for the Kepler eclipsing binary sample.

    Parameters
    ----------
    periodograms_file : str
        HD5F file containing the periodograms.
    class_datafile : str, optional
        The HDF5 file to store the rotation periods in.
    period_min, period_max : float, optional
        Will only search for `period_min` < period < `period_max`
    """
    h5_pgram = h5py.File('{}/{}'.format(data_dir, periodograms_file), 'r')
    h5_class = h5py.File('{}/{}'.format(repo_data_dir, class_datafile), 'r+')

    kics = np.array(h5_class['kic'][:], dtype=np.int64)

    p_rot_1 = np.zeros(len(kics), dtype=np.float64)
    peak_power_1 = np.zeros_like(p_rot_1)

    total_systems = len(kics)
    print('Measuring rotation periods for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        periods = h5_pgram['{}/periods'.format(kic)][:]
        powers = h5_pgram['{}/powers'.format(kic)][:]

        keep = (periods > period_min) & (periods < period_max)

        index_max = np.argmax(powers[keep])

        p_rot_1[ii] = periods[keep][index_max]
        peak_power_1[ii] = powers[keep][index_max]

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if 'pgram' in list(h5_class.keys()):
        pgram = h5_class['pgram']
    else:
        pgram = h5_class.create_group('pgram')

    for dataset in ['p_rot_1', 'peak_power_1']:
        if dataset in list(pgram.keys()):
            dset = pgram[dataset]
            dset[...] = eval(dataset)
        else:
            pgram.create_dataset(dataset, data=eval(dataset))

    pgram.attrs['run_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pgram.attrs['width_max'] = h5_pgram.attrs['width_max']


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


def correlation_at_p_orb(width_max=0.25, class_datafile='inspection_data.h5',
                         detrend=True):
    """
    Compute the cross correlation with the median light curve folded at the
    orbital period.

    Parameters
    ----------
    width_max : float, optional
        Eclipses will be interpolated over if the primary phase width
        is less than `width_max`.
    class_datafile : str, optional
        The HDF5 file to store the rotation periods in.
    detrend : bool, optional
        Set to False to not detrend light curves.
    """
    h5 = h5py.File('{}/{}'.format(repo_data_dir, class_datafile), 'r+')

    kics = h5['kic']

    correlations = np.zeros(len(kics), dtype=float) - 1.

    total_systems = len(kics)
    print('Measuring phase correlation for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        eb = eclipsing_binary.EclipsingBinary.from_kic(kic)
        eb.normalize(detrend=detrend)

        if eb.params.width_pri < width_max:
            eb.interpolate_over_eclipse()

        corr = phase_correlation(eb.l_curve.times, eb.l_curve.fluxes,
                                 eb.params.p_orb, t_0=eb.params.bjd_0)[1]

        correlations[ii] = np.nanmedian(corr)

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if 'corr' in list(h5.keys()):
        group = h5['corr']
    else:
        group = h5.create_group('corr')

    if 'corr' in list(group.keys()):
        dset = group['corr']
        dset[...] = eval('correlations')
    else:
        group.create_dataset('corr', data=eval('correlations'))

    group.attrs['run_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    group.attrs['width_max'] = width_max
    group.attrs['detrend'] = detrend


def find_acf_peaks(acf_file, class_datafile='inspection_data.h5'):
    """
    Find peaks in the auto-correlation function (ACF).

    Results are saved as a pandas DataFrame in a pickle file.

    Parameters
    ----------
    acf_file : str
        HDF5 file containing the ACFs.
    class_datafile : str, optional
        The HDF5 file to store the rotation periods in.
    """
    h5_acf = h5py.File('{}/{}'.format(data_dir, acf_file), 'r')
    h5_class = h5py.File('{}/{}'.format(repo_data_dir, class_datafile))

    kics = h5_class['kic'][:]

    p_rot_1 = np.zeros(len(kics), dtype=np.float64)
    peak_height_1 = np.zeros_like(p_rot_1)

    total_systems = len(kics)
    print('Finding ACF peaks for {} systems...'.format(total_systems))

    for ii, kic in enumerate(kics):

        lags = h5_acf['{}/lags'.format(kic)][:]
        acf = h5_acf['{}/acf'.format(kic)][:]

        peak_max, peaks, h_p = interpacf.dominant_period(lags, acf)

        p_rot_1[ii] = peak_max
        peak_height_1[ii] = h_p

        sys.stdout.write('\r{:.1f}% complete'.format((ii + 1) * 100 / total_systems))
        sys.stdout.flush()

    print()

    if 'acf' in list(h5_class.keys()):
        acf = h5_class['acf']
    else:
        acf = h5_class.create_group('acf')

    for dataset in ['p_rot_1', 'peak_height_1']:
        if dataset in list(acf.keys()):
            dset = acf[dataset]
            dset[...] = eval(dataset)
        else:
            acf.create_dataset(dataset, data=eval(dataset))

    acf.attrs['run_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    acf.attrs['width_max'] = h5_acf.attrs['width_max']


def create_inspection_datafile(datafile='inspection_data.h5'):
    """
    Create an HDF5 file to store the classification data.

    Parameters
    ----------
    datafile : str, optional
        Specify an alternate datafile name.
    """
    df = pd.read_csv('{}/{}'.format(repo_data_dir, 'initial_class.csv'))

    dt = h5py.special_dtype(vlen=str)

    h5 = h5py.File('{}/{}'.format(repo_data_dir, datafile), mode='x')
    h5.create_dataset('kic', data=df['KIC'].values, dtype=np.uint64)
    h5.create_dataset('p_orb', data=df['period'].values, dtype=np.float64)
    h5.create_dataset('class', data=df['class'].values, dtype=dt)
