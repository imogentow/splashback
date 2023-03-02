import splashback as sp
import numpy as np
import matplotlib.pyplot as plt

mcs = sp.macsis()

N_clusters = 10 #1170

N_rad = 40
log_radii = np.linspace(-1, 0.6, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

for i in range(N_clusters):
    plt.figure()
    plt.semilogx(rad_mid, mcs.SZ_median[i,:],
                 color="k")
    plt.xlabel(r"$r/R_{200m}$")
    plt.ylabel("$d \log y / d \log r$")
    plt.show()