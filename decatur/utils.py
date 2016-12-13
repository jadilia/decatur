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


def merge_catalogs(kebc_file, pgram_results, acf_results):
    """
    Merge the Kepler Eclipsing Binary Catalog (KEBC)
    and the rotation periods results

    Parameters
    ----------
    kebc_file : str, optional
        Name of the CSV file containing the KEBC.
    pgram_results : str
        Pickle file containing the rotation periods from periodograms.
    acf_results : str
        Pickle file containing the rotation periods from ACFs.

    Returns
    -------
    merge : pandas DataFrame
        Merge results
    """
    pgram_cat = pd.read_pickle('{}/{}'.format(data_dir, pgram_results))
    acf_cat = pd.read_pickle('{}/{}'.format(data_dir, acf_results))

    kebc = load_catalog(kebc_file)

    merge = pd.merge(kebc, pgram_cat, on='KIC').merge(acf_cat, on='KIC')

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


def get_classification_results(class_file, catalog_file):
    """
    Get the classification results file.
    """
    class_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              'data', class_file))

    df = pd.read_pickle(class_file)
    kebc = load_catalog(catalog_file)
    class_df = pd.merge(kebc, df, on='KIC')

    return class_df
