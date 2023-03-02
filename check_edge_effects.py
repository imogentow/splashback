import numpy as np
import matplotlib.pyplot as plt
import splashback as sp

box = "L1000N1800" #"L2800N5040"
flm = sp.flamingo(box)

N_bins_short = 45
log_radii = np.linspace(-1, 0.7, N_bins_short)
rad_mid_short = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

#N_bins_long = 51
dr = 1.7/(N_bins_short-1)
log_radii = np.arange(-1, 0.9, dr)
rad_mid_long = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

log_DM_density = sp.log_gradients(rad_mid_short, flm.DM_density_3D)
log_gas_density = sp.log_gradients(rad_mid_short, flm.gas_density_3D)

path = "splashback_data/flamingo/"
test_DM = np.genfromtxt(path + "test_3D_DM_density.csv", delimiter=",")
test_gas = np.genfromtxt(path + "test_3D_gas_density.csv", delimiter=",")

log_test_DM = sp.log_gradients(rad_mid_long, test_DM)
log_test_gas = sp.log_gradients(rad_mid_long, test_gas)

index1 = 47
index2 = 0


plt.semilogx(rad_mid_short, log_DM_density[index1,:], 
             label="DM", color="k")
plt.semilogx(rad_mid_long, log_test_DM[index2,:], 
             label="DM", color="k", linestyle="--")
xlim = plt.gca().get_xlim()
plt.plot(xlim, (-3,-3), c="grey", linestyle="--")
plt.xlim(xlim)
#plt.legend()
plt.xlabel("$r/R_{200m}$")
plt.ylabel("$d \log y / d \log r$")
plt.show()

plt.semilogx(rad_mid_short, log_gas_density[index1,:],
             label="Gas", color="b")
plt.semilogx(rad_mid_long, log_test_gas[index2,:],
             label="Gas", color="b", linestyle="--")
xlim = plt.gca().get_xlim()
plt.plot(xlim, (-3,-3), c="grey", linestyle="--")
plt.xlim(xlim)
#plt.legend()
plt.xlabel("$r/R_{200m}$")
plt.ylabel("$d \log y / d \log r$")
plt.show()