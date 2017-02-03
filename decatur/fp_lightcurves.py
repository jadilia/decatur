#!/usr/bin/env python
# encoding: utf-8
"""
Download and access re-extracted false positive light curves.
"""

from __future__ import print_function, division, absolute_import

import os

from astropy.io import fits
import numpy as np
import pandas as pd
import wget

from . import config


def download_lc():
    """
    Download re-extracted false positive light curves.

    References
    ----------
    Abdul-Masih, M., Prsa, A., Conroy, K., et al. 2016, AJ, 151, 101
    """
    # Quarters 0-17
    lc_dates = ['2009131105131', '2009166043257', '2009259160929',
                '2009350155506', '2010078095331', '2010174085026',
                '2010265121752', '2010355172524', '2011073133259',
                '2011177032512', '2011271113734', '2012004120508',
                '2012088054726', '2012179063303', '2012277125453',
                '2013011073258', '2013098041711', '2013131215648']

    lc_suffix = '_llc_fp_extract.fits'

    base_url = 'http://keplerebs.villanova.edu/includes/data/indirect/'

    am16 = pd.read_csv('decatur/data/abdul-masih16.psv', delimiter='|',
                       comment='#')

    for ii in range(len(am16)):

        kic_str = '{:09d}'.format(am16['FP'][ii])

        lc_dir = '{}/lc_extract/{}'.format(config.data_dir, kic_str)

        if not os.path.isdir(lc_dir):
            os.mkdir(lc_dir)

        for qq in range(0, 18):
            if am16['LC{}'.format(qq)][ii] == 'Q{}'.format(qq):
                filename = 'kplr{}-{}{}'.format(kic_str, lc_dates[qq],
                                                lc_suffix)

                if not os.path.exists('{}/{}'.format(lc_dir, filename)):
                    wget.download('{}{}'.format(base_url, filename),
                                  '{}/{}'.format(lc_dir, filename))


def loadlc(kic, lc_type='fed'):
    """
    Load false positive extracted light curves.

    Parameters
    ----------
    kic : int
        The KIC number of the FP star.
    lc_type : {'fed', 'ped', 'snr'}
        Choose light curve optimized for flux eclipse depth,
        percent eclipse depth, or SNR.

    Returns
    -------
    times : ndarray
        Kepler times of center of exposure.
    fluxes : ndarray
        Kepler fluxes for each quarter.
    flux_errs : ndarray
        Kepler flux errors for each exposure.
    cadences : ndarray
        Cadence number.
    quarters : ndarray
        Kepler quarter.
    flags : ndarray
        Kepler data quality flags.
    """
    hdu_indices = {'snr': 3, 'ped': 5, 'fed': 7}

    lc_dir = '{}/lc_extract/{:09d}'.format(config.data_dir, kic)

    times, fluxes, flux_errs = [], [], []
    flags, cadences, quarters = [], [], []

    # TODO: Fetch files if not downloaded
    for filename in os.listdir(lc_dir):

        if filename[-4:] == 'fits':

            hdu = fits.open('{}/{}'.format(lc_dir, filename))

            hdu_data = hdu[hdu_indices[lc_type]].data

            times = np.append(times, hdu_data['time'])
            fluxes = np.append(fluxes, hdu_data['flux'])
            flux_errs = np.append(flux_errs, hdu_data['flux_err'])
            flags = np.append(flags, hdu_data['quality'])
            cadences = np.append(cadences, hdu_data['cadenceno'])
            quarter = np.repeat(int(hdu[0].header['quarter']),
                                len(hdu_data))
            quarters = np.append(quarters, quarter)

    return times, fluxes, flux_errs, cadences, quarters, flags.astype(int)
