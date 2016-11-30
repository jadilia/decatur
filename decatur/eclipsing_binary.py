#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import warnings

import interpacf
import matplotlib.pyplot as plt
import numpy as np
from gatspy.periodic import LombScargleFast

from . import kepler_data
from . import utils


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

        self.in_kebc = False


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

        self.fluxes_original = None
        self.fluxes_detrend = None
        self.fluxes_interp = None
        self.flux_errs_original = None
        self.flux_errs_normed = None


class EclipsingBinary(object):
    """
    Eclipsing binary object.

    Parameters
    ----------
    l_curve : LightCurve object
        Object containing the light curve data
    params : BinaryParameters objects
        Object containing the eclipsing binary parameters

    Raises
    ------
    CatalogMatchError
        If there are multiple entries for the system in the KEBC
    """
    def __init__(self, light_curve=None, binary_params=None):
        self.l_curve = light_curve
        self.params = binary_params

        self.periods = None
        self.powers = None
        self.lags = None
        self.acf = None
        self.peak_max = None
        self.all_peaks = None

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
            Defaults to True. If True, use the PDCSAP data instead of
            the raw SAP.
        long_cadence : bool, optional
            Whether to select long or short cadence. Defaults to True,
            or LC data.
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

        if number_of_entries >= 1:

            if number_of_entries > 1:
                warnings.warn('Multiple catalog entries for KIC {}. '
                              'Using first entry.'.format(kic))

            p_orb = df[matching_kic]['period'].values[0]
            bjd_0 = df[matching_kic]['bjd0'].values[0]
            depth_pri = df[matching_kic]['pdepth'].values[0]
            depth_sec = df[matching_kic]['sdepth'].values[0]
            width_pri = df[matching_kic]['pwidth'].values[0]
            width_sec = df[matching_kic]['swidth'].values[0]
            sep = df[matching_kic]['sep'].values[0]

            binary_params = BinaryParameters(p_orb, bjd_0, depth_pri,
                                             depth_sec, width_pri, width_sec,
                                             sep, kic)
            binary_params.in_kebc = True

            return cls(light_curve, binary_params)

        else:
            print('No entries in EB catalog for KIC {}'.format(kic))
            print('Returning empty BinaryParameters object.')
            return cls(light_curve, BinaryParameters())

    def detrend_and_normalize(self, poly_order=3):
        """
        Detrend and normalize light curve with a low order polynomial.

        Light curves are detrended on a per-quarter basis.

        poly_order: int, optional
            The order of the polynomial fit.
        """
        self.l_curve.fluxes_original = np.copy(self.l_curve.fluxes)
        self.l_curve.flux_errs_original = np.copy(self.l_curve.flux_errs)

        # "Empty" array to hold detrended fluxes
        fluxes_detrended = np.zeros_like(self.l_curve.fluxes)
        flux_errs_normed = np.zeros_like(self.l_curve.flux_errs)

        for quarter in np.unique(self.l_curve.quarters):
            mask = self.l_curve.quarters == quarter

            # Compute polynomial fit
            poly = np.polyfit(self.l_curve.times[mask],
                              self.l_curve.fluxes[mask], poly_order)
            z = np.poly1d(poly)

            median_flux = np.nanmedian(self.l_curve.fluxes[mask])

            # Subtract fit and median normalize
            fluxes_detrended[mask] = (self.l_curve.fluxes[mask] -
                                      z(self.l_curve.times[mask])) \
                / median_flux

            flux_errs_normed[mask] = self.l_curve.flux_errs[mask] / median_flux

        self.l_curve.fluxes_detrended = np.copy(fluxes_detrended)
        self.l_curve.fluxes = np.copy(fluxes_detrended)

        self.l_curve.flux_errs_normed = np.copy(flux_errs_normed)
        self.l_curve.flux_errs = np.copy(flux_errs_normed)

    def phase_fold(self, period_fold=None):
        """
        Phase fold the light curve at given period.
        Uses the orbital period by default.

        Parameters
        ----------
        period_fold : float, optional
            Specify a different period at which to fold.

        Returns
        -------
        phase : numpy.ndarray
            The orbital phase.
        """
        if self.params.in_kebc:
            # Subtract offset for Kepler mission days
            bjd_0 = self.params.bjd_0 - 54833
        else:
            bjd_0 = self.params.bjd_0

        if period_fold is None:
            period_fold = self.params.p_orb

        phase = ((self.l_curve.times - bjd_0) % period_fold) / period_fold

        return phase

    def interpolate_over_eclipse(self, window=1.):
        """
        Linearly interpolate over the eclipses.

        Parameters
        ----------
        window : float, optional
            The fraction of the eclipse width to interpolate over.
        """
        phase = self.phase_fold()

        window /= 2

        mask = ((phase > self.params.width_pri * window) &
                (phase < 1 - self.params.width_pri * window)) & \
            ((phase > self.params.sep + self.params.width_sec * window) |
             (phase < self.params.sep - self.params.width_sec * window))

        fluxes_interp = np.copy(self.l_curve.fluxes)
        fluxes_interp[~mask] = np.interp(self.l_curve.times[~mask],
                                         self.l_curve.times[mask],
                                         self.l_curve.fluxes[mask])

        self.l_curve.fluxes_interp = np.copy(fluxes_interp)
        self.l_curve.fluxes = np.copy(fluxes_interp)

    def run_periodogram(self, oversampling=5):
        """
        Compute a periodogram using gatspy.LombScargleFast.

        Parameters
        ----------
        oversampling : int, optional
            The oversampling factor for the periodogram.
        """
        model = LombScargleFast().fit(self.l_curve.times,
                                      self.l_curve.fluxes,
                                      self.l_curve.flux_errs)

        self.periods, self.powers = model.periodogram_auto(oversampling=oversampling)

    def run_acf(self):
        """
        Compute the autocorrelation function.
        """
        self.lags, self.acf = interpacf.interpolated_acf(self.l_curve.times,
                                                         self.l_curve.fluxes,
                                                         cadences=self.l_curve.cadences)

    def find_acf_peaks(self):
        """
        Find the peaks in the autocorrelation function.
        """
        self.peak_max, self.all_peaks = interpacf.dominant_period(self.lags,
                                                                  self.acf)
    
    def phase_evolution_plot(self, t_min = 0, t_max = 10000, period_fold = None):
        """
        Plot a phase folded light curve color coded by time.

        Parameters
        ----------
        t_min, t_max : float, optional
            The minimum and maximum observation times to plot. The defaults
            will include all data.
        """
        if period_fold is None:
            period_fold = self.params.p_orb
        self.detrend_and_normalize()
        phase = self.phase_fold(period_fold = period_fold)
        mask = (self.l_curve.times > t_min) & (self.l_curve.times < t_max)
        phase = phase[mask]
        fluxes = self.l_curve.fluxes[mask]
        times = self.l_curve.times[mask]

        plt.scatter(phase, fluxes, s=1, c=times, edgecolors="None",
                    cmap="viridis")
        plt.xlabel('Phase')
        plt.ylabel('Relative Flux')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Days')
        plt.show()
