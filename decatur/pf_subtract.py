"""
Subtract KEBC phase-folded polyfit from light curves.
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import wget
from scipy import interpolate

from decatur import config, utils


def download_kebc_lc(kic):
    """
    Download the processed light curve and phase polyfit from the KEBC.

    Parameters
    ----------
    kic : int
        The KIC number of the star.
    """
    if not os.path.isdir('{}/lc_kebc'.format(config.data_dir)):
        os.mkdir('{}/lc_kebc'.format(config.data_dir))

    lc_dir = '{}/lc_kebc/{:09d}'.format(config.data_dir, kic)

    if not os.path.isdir(lc_dir):
        os.mkdir(lc_dir)

    base_url = 'http://keplerebs.villanova.edu/data/?k={}.00&cadence=lc&data='.format(kic)

    lc_filename = '{:09d}.00.lc.data.tsv'.format(kic)
    pf_filename = '{:09d}.00.lc.pf.data.tsv'.format(kic)

    for filename, url_end in zip([lc_filename, pf_filename], ['data', 'pf']):
        if not os.path.exists('{}/{}'.format(lc_dir, filename)):
            wget.download('{}{}'.format(base_url, url_end),
                          '{}/{}'.format(lc_dir, filename))


def load_kebc_lc(kic, flux_type='dtr', pf_type='decon'):
    """
    Load the processed light curve and phase polyfit from the KEBC.

    Parameters
    ----------
    kic : int
        The KIC number of the star.
    flux_type : {'dtr', 'raw', 'corr'}
        The type of flux to use. Detrended, raw, or corrected.
    pf_type : {'decon', 'recon'}
        Whether to use deconvolved or reconvolved polyfit fluxes.

    Returns
    -------
    times : numpy.ndarray
        Observation times in BJD.
    fluxes : numpy.ndarray
        Fluxes of `flux_type`.
    phases : numpy.ndarray
        Orbital phase.
    flux_errs : numpy.ndarray
        Flux errors
    pf_phases : numpy.ndarray
        Orbital phase of polyfit.
    pf_fluxes : numpy.ndarray
        Fluxes of polyfit of `pf_type`.
    """
    download_kebc_lc(kic)

    lc_dir = '{}/lc_kebc/{:09d}'.format(config.data_dir, kic)

    lc_filename = '{}/{:09d}.00.lc.data.tsv'.format(lc_dir, kic)

    lc_names = ['bjd', 'phase', 'raw_flux', 'raw_err', 'corr_flux', 'corr_err',
                'dtr_flux', 'dtr_err']
    df_lc = pd.read_csv(lc_filename, comment='#', delim_whitespace=True,
                        names=lc_names, usecols=np.arange(0, 8))

    times = df_lc['bjd'].values - 54833
    phases = df_lc['phase'].values
    fluxes = df_lc['{}_flux'.format(flux_type)].values
    flux_errs = df_lc['{}_err'.format(flux_type)].values

    lc_filename = '{}/{:09d}.00.lc.pf.data.tsv'.format(lc_dir, kic)

    pf_names = ['phase', 'pf_deconvolved_flux',
                'pf_deconvolved_reconvolved_flux']
    df_pf = pd.read_csv(lc_filename, comment='#', delim_whitespace=True,
                        names=pf_names, usecols=np.arange(0, 3))

    pf_dict = {'decon': 'pf_deconvolved_flux',
               'recon': 'pf_deconvolved_reconvolved_flux'}
    pf_phases = df_pf['phase'].values
    pf_fluxes = df_pf[pf_dict[pf_type]].values

    return times, fluxes, phases, flux_errs, pf_phases, pf_fluxes


def pf_subracted_lc(kic, flux_type='dtr', pf_type='decon',
                    interp_eclipses=False, window=1.):
    """
    Subtract the polyfit from the light curve.

    Parameters
    ----------
    kic : int
        The KIC number of the star.
    flux_type : {'dtr', 'raw', 'corr'}
        The type of flux to use. Detrended, raw, or corrected.
    pf_type : {'decon', 'recon'}
        Whether to use deconvolved or reconvolved polyfit fluxes.
    interp_eclipses : bool, optional
        If True interpolate over eclipses.
    window : float, optional
        Window width to interpolate over in fraction of eclispe width.

    Returns
    -------
    times : numpy.ndarray
        Observation times in BJD.
    fluxes_sub : numpy.ndarray
        Subtracted fluxes.
    """
    returns = load_kebc_lc(kic, flux_type, pf_type)
    times, fluxes, phases, flux_errs, pf_phases, pf_fluxes = returns

    pp = np.hstack([pf_phases - 1, pf_phases, pf_phases + 1])
    ff = np.hstack([pf_fluxes, pf_fluxes, pf_fluxes])

    interp = interpolate.interp1d(pp, ff)

    subtracted_lc = fluxes - interp(phases)

    if interp_eclipses:
        kebc = utils.load_catalog()
        width_pri = kebc['pwidth'][kebc['KIC'] == kic].values[0]
        width_sec = kebc['swidth'][kebc['KIC'] == kic].values[0]
        sep = kebc['sep'][kebc['KIC'] == kic].values[0]

        window /= 2

        phases += 0.5

        mask = ((phases > width_pri * window) &
                (phases < 1 - width_pri * window)) & \
               ((phases > sep + width_sec * window) |
                (phases < sep - width_sec * window))

        fluxes_sub = np.copy(subtracted_lc)
        fluxes_sub[~mask] = np.interp(times[~mask], times[mask],
                                      subtracted_lc[mask])
    else:
        fluxes_sub = subtracted_lc

    return times, fluxes_sub
