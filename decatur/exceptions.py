#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import


class CatalogMatchError(Exception):
    """
    Raised when there are no matching entries in the EB catalog.
    """
    def __init__(self, message):
        self.message = message


class DatabaseSetupError(Exception):
    """
    Raised if the light curve database environment variables are not defined.
    """
    def __init__(self, message):
        self.message = message


class NoLightCurvesError(Exception):
    """
    Raised if there are no light curves for given target in the database.
    """
    def __init__(self, message):
        self.message = message
