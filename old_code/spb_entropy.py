import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
from scipy import interpolate

PATH = "Analysis_files/Splashback/"
entropy = np.genfromtxt(PATH + "entropy_profiles_macsis.csv", delimiter=",")

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins+1)
radii = 10**log_radii
rad_mid = 10**((log_radii[1:] + log_radii[:-1]) / 2)
window = 15
order = 4

N_clusters = 390
smooth_K = np.zeros((N_clusters, N_bins))
for i in range(50):
    dlnK_dlnr = np.gradient(np.log(entropy[i,:]), np.log(rad_mid))
    smooth_K[i,:] = sp.savitzky_golay(dlnK_dlnr, window, order)
    
    # plt.figure()
    # plt.semilogx(rad_mid, smooth_K)
    # plt.title(i)
    # plt.show()

interp =interpolate.interp1d(rad_mid, smooth_K[14,:], kind="cubic")
x_interp = np.linspace(0.15, 4.7, 100)
K_interp = interp(x_interp)

plt.figure()
#plt.semilogx(rad_mid, smooth_K, color="k", linewidth=5)
plt.semilogx(x_interp, K_interp, color="k", linewidth=5)
ylim = plt.gca().get_ylim()
low_RSP = 0.838859188
high_RSP = 1.203017
plt.fill_betweenx(ylim, low_RSP, high_RSP, alpha=0.5, color="grey")
plt.ylim(ylim)
plt.xlabel("$r/R_{200m}$")
plt.ylabel("$d \log K / d \log r$")
plt.xlim((0.15,4))
plt.subplots_adjust(left=0.2, bottom=0.2)
#plt.savefig("entropy_profile_macsis_14.png", dpi=300)
plt.show()
