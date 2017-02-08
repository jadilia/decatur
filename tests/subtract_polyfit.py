from __future__ import absolute_import, division, print_function

import os
import sys

import interpacf
import matplotlib.pyplot as plt
from gatspy.periodic import LombScargleFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import decatur

kic = 11724091
returns = decatur.pf_subtract.load_kebc_lc(kic)
times1, fluxes1, phases1, _, pf_phases, pf_fluxes = returns
times2, fluxes2 = decatur.pf_subtract.pf_subracted_lc(kic)
times3, fluxes3 = decatur.pf_subtract.pf_subracted_lc(kic,
                                                      interp_eclipses=True,
                                                      window=1.0)

plt.scatter(phases1, fluxes1, s=0.01, color='k')
plt.plot(pf_phases, pf_fluxes, lw=0.5)
plt.xlabel('Phase')
plt.ylabel('Normalized Flux')

plt.figure(2)
plt.step(times1, fluxes1, lw=0.5, label='original')
plt.step(times2, fluxes2 + 1, lw=0.5, label='minus polyfit')
plt.step(times3, fluxes3 + 1, lw=0.5, label='eclipses removed')
plt.xlabel('Time (days)')
plt.ylabel('Normalized Flux')
plt.xlim(times1[0], times1[0] + 10)
plt.legend(loc='lower left')

model1 = LombScargleFast().fit(times1, fluxes1)
periods1, powers1 = model1.periodogram_auto(oversampling=5)

model3 = LombScargleFast().fit(times3, fluxes3)
periods3, powers3 = model3.periodogram_auto(oversampling=5)

plt.figure(3)
plt.step(periods1, powers1, lw=0.5)
plt.step(periods3, powers3, lw=0.5)
plt.xlabel('Period (days)')
plt.ylabel('Normalized power')
plt.xlim(0, 10)

plt.show()

# plt.figure(4)
# lags3, acf3 = interpacf.interpolated_acf(times3, fluxes3)
# plt.step(lags3, acf3 / acf3.max(), lw=0.5)
# plt.xlabel('Lag (days)')
# plt.ylabel('ACF')
# plt.xlim(0, 10)
#
#
# bjd_0 = 0
# # period_fold = 0.9771
# # period_fold = 1.0346
# period_fold = 0.9299
#
# pp = ((times3 - bjd_0) % period_fold) / period_fold
#
# plt.figure(5)
# plt.scatter(pp, fluxes3, s=0.1, color='k')
# plt.xlabel('Phase')
# plt.ylabel('Normalized flux')
#
# plt.show()
