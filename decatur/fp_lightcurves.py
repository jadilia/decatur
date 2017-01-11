#!/usr/bin/env python
# encoding: utf-8
"""
Download and access re-extracted false positive light curves.
"""

from __future__ import print_function, division, absolute_import

import os

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
