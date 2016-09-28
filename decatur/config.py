#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import os

# Load database parameters from environment variables
host = os.getenv('DB_HOST')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
db_params = {'host': host, 'user': user, 'password': password}
