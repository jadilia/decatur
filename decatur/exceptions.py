#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import


class CatalogMatchError(Exception):
    """
    Raised when there are no matching entries in the EB catalog.
    """
    pass


class DatabaseSetupError(Exception):
    """
    Raised if the light curve database environment variables are not defined.
    """
    pass


class NoLightCurvesError(Exception):
    """
    Raised if there are no light curves for given target in the database.
    """
    pass
