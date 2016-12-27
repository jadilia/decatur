from __future__ import absolute_import, division, print_function

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import decatur


# EB with starspots
eb_spots = decatur.eclipsing_binary.EclipsingBinary.from_kic(7129465,
                                                             from_db=False)
eb_spots.normalize()

plt.figure(1)
plt.plot(eb_spots.l_curve.times, eb_spots.l_curve.fluxes)

# EB with ellipsoidal variations
eb_ellip = decatur.eclipsing_binary.EclipsingBinary.from_kic(6213131,
                                                             from_db=False)
eb_ellip.normalize()

plt.figure(2)
plt.plot(eb_ellip.l_curve.times, eb_ellip.l_curve.fluxes)

# EB with no periodic out-of-eclipse variability
eb_flat = decatur.eclipsing_binary.EclipsingBinary.from_kic(5553624,
                                                            from_db=False)
eb_flat.normalize()

plt.figure(3)
plt.plot(eb_flat.l_curve.times, eb_flat.l_curve.fluxes)
plt.show()

# Inspect the light curves
# gadget = decatur.inspector.InspectorGadget('rotation_periods.20161025.pkl',
#                                            'periodograms.20160929.h5',
#                                            results_file='test.pkl',
#                                            from_db=False)
#
# gadget.gogo_gadget()
