#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import os

import pandas as pd


def load_catalog(catalog_file='kebc.csv'):
    """
    Load the Kepler Eclipsing Binary Catalog

    http://keplerebs.villanova.edu/

    Parameters
    ----------
    catalog_file : str, optional
        Name of the CSV file containing the catalog.

    Returns
    -------
    df : pandas DataFrame
        DataFrame containing the catalog.

    Raises
    ------
    IOError
        If the catalog file does not exist.
    """
    # Construct the absolute path of the catalog file
    catalog_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                'data', catalog_file))

    if not os.path.exists(catalog_file):
        raise IOError('No such catalog file: {}'.format(catalog_file))

    # Load the catalog as a pandas DataFrame
    df = pd.read_csv(catalog_file, comment='#')

    return df
