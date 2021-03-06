from __future__ import absolute_import, division, print_function

import os
import platform
import sys

os.environ['MPLBACKEND'] = 'TkAgg'

import h5py
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import decatur
from decatur import config, utils

h5 = h5py.File('{}/inspection_data.h5'.format(decatur.config.repo_data_dir))
df = pd.DataFrame({'KIC': h5['kic'][:], 'corr': h5['corr/corr'][:]})

kebc = utils.load_catalog()
merge = pd.merge(kebc, df, on='KIC')

keep = (merge['pwidth'] < 0.1) & (merge['corr'] < 0.95)

merge['KIC'][keep].to_csv('{}/kic_keep.csv'.format(config.repo_data_dir),
                          index=False)

# Setup to run on my laptop
if platform.system() == 'Darwin':
    from_db = False
    zoom_pan = 0.15
else:
    from_db = True
    zoom_pan = 0.05

gadget = decatur.inspector.InspectorGadget('periodograms.20160929.h5',
                                           'acfs.20161108.h5',
                                           kic_list=merge['KIC'][keep].values,
                                           from_db=from_db, zoom_pan=zoom_pan)
gadget.gogo_gadget()
