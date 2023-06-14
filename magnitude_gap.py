import splashback as sp
import numpy as np
import matplotlib.pyplot as plt

box = "L1000N3600"
flm = sp.flamingo(box, "HF")
flm.read_properties()
# mass_mask = np.where(flm.M200m > 10**15)[0]
magnitudes = np.genfromtxt(flm.path + "_galaxy_magnitudes.csv", delimiter=",")
# flm.DM_density_3D = flm.DM_density_3D[mass_mask]
# flm.gas_density_3D = flm.gas_density_3D[mass_mask]
# flm.accretion = flm.accretion[mass_mask]
sorted_magnitudes = np.sort(magnitudes)
mag_bcg = sorted_magnitudes[:,0]
mag_fourth = sorted_magnitudes[:,2]
flm.gap = mag_fourth - mag_bcg

N_bins = 5
gap_bins = np.linspace(0, 3, N_bins+1)
sp.stack_and_find_3D(flm, "gap", gap_bins)

plt.figure()
cm = plt.cm.copper(np.linspace(0,1,N_bins))
for i in range(N_bins):
    label = str(np.round(gap_bins[i],2)) \
            + "$< \\rm{M14} <$" \
            + str(np.round(gap_bins[i+1],2))
    plt.semilogx(flm.rad_mid, flm.gap_log_DM[i,:], color=cm[i],
                 linewidth=0.8, label=label)
plt.legend()
plt.xlabel("$r/R_{\\rm{200m}}$")
plt.ylabel("$d \log \\rho_{\\rm{DM}} / d \log r$")
plt.show()

print(len(flm.gap[np.isfinite(flm.gap)]))

plt.hist(flm.gap, bins=50, range=(0,4))
plt.show()

plt.figure()
plt.scatter(flm.gap, magnitudes[:,1], color="r", s=2)
plt.scatter(flm.gap, magnitudes[:,2], color="b", s=2)
plt.xlabel("M14")
plt.ylabel("r-band magnitude")
plt.show()

mask = np.where(np.isfinite(flm.accretion) & np.isfinite(flm.gap))[0]

print(np.corrcoef(flm.accretion[mask], flm.gap[mask]))

plt.scatter(flm.accretion, flm.gap)
plt.xlabel("$\Gamma$")
plt.ylabel("M14")
plt.show()