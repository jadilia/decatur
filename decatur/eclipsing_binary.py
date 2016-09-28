#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import numpy as np

from . import kepler_data
from . import utils
from .exceptions import CatalogMatchError


class BinaryParameters(object):
    """
    Contains the eclipsing binary parameters.

    Parameters are set to None by default.

    Parameters
    ----------
    p_orb : float, optional
        The orbital period in days
    bjd_0 : float, optional
        Reference time of primary eclipse in BJD
    depth_pri, depth_sec : float, optional
        The depths of primary and secondary eclipse in relative flux
    width_pri, width_sec : float, optional
        The widths of primary and secondary eclipse phase
    sep : float, optional
        The separation between primary and secondary eclipse in phase
    kic : int, optional
        The KIC ID number
    """
    def __init__(self, p_orb=None, bjd_0=None, depth_pri=None, depth_sec=None,
                 width_pri=None, width_sec=None, sep=None, kic=None):

        self.p_orb = p_orb
        self.bjd_0 = bjd_0
        self.depth_pri = depth_pri
        self.depth_sec = depth_sec
        self.width_pri = width_pri
        self.width_sec = width_sec
        self.sep = sep
        self.kic = kic


class LightCurve(object):
    """
    Contains the light curve data.

    Time and flux are required. Errors and additional data are optional.

    Parameters
    ----------
    times : array_like
        The observation time at mid-exposure in days
    fluxes : array_like
        Fluxes may be either in physical or relative units
    flux_errs : array_like, optional
        The flux errors, in the same units as `flux`
    cadences : array_like, optional
        The cadence numbers of the exposures
    quarters : array_like, optional
        The quarters in which the observations were made
    flags : array_like, optional
        Data quality flags
    """
    def __init__(self, times, fluxes, flux_errs=None, cadences=None,
                 quarters=None, flags=None):
        self.times = times
        self.fluxes = fluxes
        self.flux_errs = flux_errs
        self.cadences = cadences
        self.quarters = quarters
        self.flags = flags


class EclipsingBinary(object):
    """
    Eclipsing binary object.

    Parameters
    ----------
    light_curve : LightCurve object
        Object containing the light curve data
    binary_params : BinaryParameters objects
        Object containing the eclipsing binary parameters

    Raises
    ------
    CatalogMatchError
        If there are multiple entries for the system in the KEBC
    """
    def __init__(self, light_curve=None, binary_params=None):
        self.light_curve = light_curve
        self.binary_params = binary_params

    @classmethod
    def from_kic(cls, kic, catalog_file='kebc.csv', use_pdc=True,
                 long_cadence=True, from_db=True):
        """
        Instantiate an object with a Kepler Input Catalog ID.

        Parameters
        ----------
        kic : int
            The KIC ID number
        catalog_file : str, optional
            The name of the CSV file containing the KEBC
        use_pdc : bool, optional
            Defaults to True. If True, use the PDCSAP data instead of the raw SAP.
        long_cadence : bool, optional
            Whether to select long or short cadence. Defaults to True, or LC data.
        from_db : bool, optional
            Default loads data from the MySQL database.
            Set to False to load data from MAST using the kplr package.
            NOT YET IMPLEMENTED.
        """
        returns = kepler_data.loadlc(kic, use_pdc, long_cadence, from_db)
        times, fluxes, flux_errs, cadences, quarters, flags = returns

        light_curve = LightCurve(times, fluxes, flux_errs, cadences, quarters,
                                 flags)

        df = utils.load_catalog(catalog_file)

        matching_kic = df['KIC'] == kic

        number_of_entries = np.sum(matching_kic)

        if number_of_entries == 1:
            p_orb = df[matching_kic]['period'].values[0]
            bjd_0 = df[matching_kic]['bjd0'].values[0]
            depth_pri = df[matching_kic]['pdepth'].values[0]
            depth_sec = df[matching_kic]['sdepth'].values[0]
            width_pri = df[matching_kic]['pwidth'].values[0]
            width_sec = df[matching_kic]['swidth'].values[0]

            binary_params = BinaryParameters(p_orb, bjd_0, depth_pri,
                                             depth_sec, width_pri, width_sec,
                                             kic)

            return cls(light_curve, binary_params)

        elif number_of_entries < 1:
            print('No entries in EB catalog for KIC {}'.format(kic))
            print('Returning empty BinaryParameters object.')
            return cls(light_curve, BinaryParameters())

        elif number_of_entries > 1:
            raise CatalogMatchError('Multiple entries in catalog for KIC {}'.format(kic))
