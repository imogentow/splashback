import splashback as sp
import numpy as np
import matplotlib.pyplot as plt

box = "L2800N5040"
flm = sp.flamingo(box, "HF")
flm.read_properties()
# mass_mask = np.where(flm.M200m < 10**14.5)[0]
magnitudes = np.genfromtxt(flm.path + "_galaxy_magnitudes.csv", delimiter=",")#[mass_mask]
gap2 = magnitudes[:,1] - magnitudes[:,0] 
gap4 = (magnitudes[:,2] - magnitudes[:,0])
flm.gap = gap4

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
plt.scatter(gap4, magnitudes[:,0], color="r", s=2)
plt.scatter(gap4, magnitudes[:,2], color="b", s=2)
plt.xlabel("M14")
plt.ylabel("r-band magnitude")
plt.show()

mask2 = np.where(np.isfinite(flm.accretion) & np.isfinite(gap2))[0]
mask4 = np.where(np.isfinite(flm.accretion) & np.isfinite(gap4))[0]

print(np.corrcoef(flm.accretion[mask2], gap2[mask2]))
print(np.corrcoef(flm.accretion[mask4], gap4[mask4]))

plt.scatter(flm.accretion, gap4)
plt.show()