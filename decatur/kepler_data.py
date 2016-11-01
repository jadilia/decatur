#!/usr/bin/env python
# encoding: utf-8
"""
Query the KEBC and load Kepler light curves.
"""

from __future__ import print_function, division, absolute_import

import kplr
import numpy as np
import paramiko
import pymysql
import socket

from . import exceptions
from . import utils
from .config import db_params


def select_kics(catalog_file='kebc.csv', period_min=0., period_max=None):
    """
    Return KIC IDs based on system parameters.

    Parameters
    ----------
    catalog_file : string, optional
        Name of catalog file contained in decatur/data
    period_min : float, optional
        Minimum orbital period in days.
    period_max : float, optional
        Maximum orbital period in days.

    Returns
    -------
    kics : numpy.ndarray
        KIC IDs matching search criteria.

    Raises
    ------.
    CatalogMatchError
        If the query returns no KIC IDs.
    """
    # Load the catalog DataFrame.
    df = utils.load_catalog(catalog_file)

    # Construct the query string with minimum period cutoff.
    query_string = 'period > {}'.format(period_min)

    if period_max is not None:
        # Add the maximum period cutoff
        query_string = '{} & period < {}'.format(query_string, period_max)

    # Query the catalog and return KIC IDs.
    kics = df.query(query_string)['KIC'].values.astype(int)

    if len(kics) == 0:
        raise exceptions.CatalogMatchError('No EB catalog entries matching criteria.')

    return kics.astype(int)


def __dbconnect(db_name):
    """
    Log into a database using MySQLdb. Written by Ethan Kruse.

    Parameters
    ----------
    db_name : string
        Database name

    Returns
    -------
    dbconnect : Connect
        MySQLdb connector.

    Raises
    ------
    DatabaseSetupError
        If the environment variables for the database are not defined
    """
    if None in db_params.values():
        raise exceptions.DatabaseSetupError('Environment variables for the '
                                            'database are not defined.')

    return pymysql.connect(host=db_params['host'], user=db_params['user'],
                           passwd=db_params['password'], db=db_name,
                           connect_timeout=1)


def loadlc(kic, use_pdc=True, long_cadence=True, from_db=True,
           db_name='Kepler', fetch=True):
    """
    Load Kepler data from a local database. Written by Ethan Kruse.

    Parameters
    ----------
    kic : int
        Kepler Input Catalog number for the target.
    use_pdc : bool, optional
        Defaults to True. If True, use the PDCSAP data instead of the raw SAP.
    long_cadence : bool, optional
        Whether to select long or short cadence. Defaults to True, or LC data.
    from_db : bool, optional
        Default loads data from the MySQL database.
        Set to False to load data from MAST using the kplr package.
    db_name : str, optional
        Database name.
    fetch : bool, optional
        If `from_db` == False, set to `fetch == False` to not download data.

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

    Raises
    ------
    NoLightCurvesError
        If there are no light curves for the given KIC ID.

    """
    if from_db:
        table_name = 'source'

        if long_cadence:
            lc_flag = 'LCFLAG > 0'
        else:
            lc_flag = 'LCFLAG = 0'

        if use_pdc:
            flux_str = 'pdcsap_flux, pdcsap_flux_err '
        else:
            flux_str = 'sap_flux, sap_flux_err '

        host_name = socket.gethostname()
        # Check if local host is on same domain as database host
        if db_params['domain'] in host_name:
            count = 0
            got_it = False
            # Try multiple times in case of sporadic database timeouts
            while count < 5 and not got_it:
                try:
                    db = __dbconnect(db_name)
                    cursor = db.cursor()

                    to_ex = 'SELECT cadenceno, quarter, sap_quality, time, {} ' \
                            'FROM {} WHERE keplerid = %s AND {};'\
                        .format(flux_str, table_name, lc_flag)

                    cursor.execute(to_ex, (int(kic),))
                    results = cursor.fetchall()
                    cadences = np.array([x[0] for x in results], dtype=np.int32)
                    quarters = np.array([x[1] for x in results], dtype=np.int32)
                    flags = np.array([x[2] for x in results], dtype=np.int32)
                    times = np.array([x[3] for x in results], dtype=np.float64)
                    fluxes = np.array([x[4] for x in results], dtype=np.float32)
                    flux_errs = np.array([x[5] for x in results], dtype=np.float32)
                    cursor.close()
                    db.close()

                    # For some reason some results are coming back with
                    # arrays of length 0.
                    if len(times) > 0:
                        got_it = True

                    count += 1
                except pymysql.OperationalError:
                    print('mysqldb connection failed on attempt {0} of {1}.\n'
                          'Trying again.'.format(count + 1, 5))
                    count += 1
        else:
            # Run query through an SSH tunnel
            query_str = "'SELECT cadenceno, quarter, sap_quality, time, {0} " \
                        "FROM {3} " \
                        "WHERE keplerid = {2} AND {1};'".format(flux_str, lc_flag, int(kic), table_name)

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(db_params['tunnel_host'], username=db_params['tunnel_user'])

            command_str = 'mysql -h {} -u {} -D {} --password={} -e {}'\
                .format(db_params['host'], db_params['user'], db_name,
                        db_params['password'], query_str)

            stdin, stdout, stderr = ssh.exec_command(command_str)
            results = stdout.read().splitlines()
            results = results[1:]

            cadences = np.array([int(x.split('\t')[0]) for x in results],
                                dtype=np.int32)
            quarters = np.array([int(x.split('\t')[1]) for x in results],
                                dtype=np.int32)
            flags = np.array([int(x.split('\t')[2]) for x in results],
                             dtype=np.int32)
            times = np.array([float(x.split('\t')[3]) for x in results],
                             dtype=np.float64)
            fluxes = np.array([float(x.split('\t')[4]) for x in results],
                              dtype=np.float32)
            flux_errs = np.array([float(x.split('\t')[5]) for x in results],
                                 dtype=np.float32)
            ssh.close()

    else:
        client = kplr.API()

        light_curves = client.light_curves(kepler_id=kic, fetch=fetch,
                                           short_cadence=~long_cadence)

        times, fluxes, flux_errs = [], [], []
        flags, cadences, quarters = [], [], []

        for lc in light_curves:
            with lc.open() as ff:
                hdu_data = ff[1].data

                times = np.append(times, hdu_data["time"])
                flags = np.append(flags, hdu_data["sap_quality"])
                cadences = np.append(cadences, hdu_data["cadenceno"])

                quarter = np.repeat(int(ff[0].header['quarter']),
                                    len(hdu_data))
                quarters = np.append(quarters, quarter)

                if use_pdc:
                    fluxes = np.append(fluxes, hdu_data["pdcsap_flux"])
                    flux_errs = np.append(flux_errs,
                                          hdu_data["pdcsap_flux_err"])
                else:
                    fluxes = np.append(fluxes, hdu_data["sap_flux"])
                    flux_errs = np.append(flux_errs, hdu_data["sap_flux_err"])

        # Remove NaNs
        good_data = np.isfinite(fluxes)
        times = times[good_data]
        fluxes = fluxes[good_data]
        flux_errs = flux_errs[good_data]
        flags = flags[good_data]
        cadences = cadences[good_data]
        quarters = quarters[good_data]

    if len(times) == 0:
        raise exceptions.NoLightCurvesError('No light curves found for KIC {}'.format(kic))

    # Guarantee the light curve is in sequential order
    # %timeit says that doing the ordering in Python is faster than
    #  including an 'ORDER BY time' flag in the MySQL search.
    # I have no idea why, but I'll keep doing the ordering here.
    order = np.argsort(times)
    times = times[order]
    fluxes = fluxes[order]
    flux_errs = flux_errs[order]
    flags = flags[order]
    cadences = cadences[order]
    quarters = quarters[order]

    return times, fluxes, flux_errs, cadences, quarters, flags
