#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import os

# Data directory
data_dir = os.getenv('DECATUR_DATA_DIR')
if data_dir is None:
    data_dir = '.'

# Repository data directory
repo_data_dir = catalog_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                            'data'))

# Load database parameters from environment variables
host = os.getenv('DB_HOST')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
domain = os.getenv('DB_DOMAIN')
tunnel_host = os.getenv('TUNNEL_HOST')
tunnel_user = os.getenv('TUNNEL_USER')
db_params = {'host': host, 'user': user, 'password': password,
             'domain': domain, 'tunnel_host': tunnel_host,
             'tunnel_user': tunnel_user}
