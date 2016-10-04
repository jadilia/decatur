#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import os

import pandas as pd

from .config import data_dir


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


def merge_catalogs(kebc_file, p_rot_file):
    """
    Merge the Kepler Eclipsing Binary Catalog (KEBC)
    and the rotation periods results

    Parameters
    ----------
    kebc_file : str, optional
        Name of the CSV file containing the KEBC.
    p_rot_file : str, optional
        Name of the pickle file containing the rotation periods.

    Returns
    -------
    merge : pandas DataFrame
        Merge results
    """
    p_rot_cat = pd.read_pickle('{}/{}'.format(data_dir, p_rot_file))
    kebc = load_catalog(kebc_file)

    merge = pd.merge(kebc, p_rot_cat, on='KIC')

    return merge


def is_int(string):
    """
    Returns True if a string represents an integer, False otherwise.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False
