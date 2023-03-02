import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr

box = "L1000N1800"
flm = sp.flamingo(box)

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

log_EM_density = sp.log_gradients(rad_mid, flm.EM_median)
log_SZ_density = sp.log_gradients(rad_mid, flm.SZ_median)
log_WL_density = sp.log_gradients(rad_mid, flm.WL_median)

Rsp_EM = dr.standard(rad_mid, log_EM_density)
Rsp_SZ = dr.standard(rad_mid, log_SZ_density)
Rsp_WL = dr.standard(rad_mid, log_WL_density)

# plt.figure()
# plt.scatter(Rsp_WL, Rsp_EM, edgecolor="k")
# plt.show()
