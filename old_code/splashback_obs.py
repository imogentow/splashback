"""
Looks at splashback properties from observable profiles. 
So far only looks at MACSIS data.
Looks at stacked profiles of different observables.
Compares splashback radii of observables to each other and to 3D dark matter.
"""

import splashback as sp
import numpy as np
import matplotlib.pyplot as plt

macsis = sp.macsis()

# Stacked profiles
DM_avg = sp.stack_data(macsis.DM_median)
EM_avg = sp.stack_data(macsis.EM_median)
SD_avg = sp.stack_data(macsis.SD_median)
SZ_avg = sp.stack_data(macsis.SZ_median)

N_bins = 40

log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

plt.figure()
plt.semilogx(rad_mid, DM_avg, c="k", label="Surface density")
plt.semilogx(rad_mid, EM_avg, c="r", label="EM")
plt.semilogx(rad_mid, SD_avg, c="gold", label="Stellar density")
plt.semilogx(rad_mid, SZ_avg, c="cyan", label="SZ")
plt.xlabel("$R_{SP,DM}$")
plt.ylabel("$R_{SP,gas}$")
plt.legend()
#plt.savefig("R_SP_DM_v_gas.png", dpi=300)
plt.show()

# Compare splashback radii
macsis.calculate_Rsp_2D()

RSP_DM_3D = sp.R_SP_finding(rad_mid, macsis.DM_density_3D)

plt.scatter(np.hstack((RSP_DM_3D, RSP_DM_3D, RSP_DM_3D)), macsis.SP_DM_median)
plt.show()

plt.figure()
plt.scatter(macsis.SP_DM_median, macsis.SP_EM_median)
ylim = plt.gca().get_ylim()
plt.show()

plt.figure()
plt.scatter(macsis.SP_DM_median, macsis.SP_SZ_median)
plt.ylim(ylim)
plt.show()

plt.figure()
plt.scatter(macsis.SP_EM_median, macsis.SP_SZ_median)
plt.ylim(ylim)
plt.show()
