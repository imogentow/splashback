import numpy as np
import splashback as sp
import matplotlib.pyplot as plt

box = "L1000N1800"
plt.style.use("mnras.mplstyle")
    
flm = sp.flamingo(box, "HF")
flm.read_2D_properties()
flm.read_properties()
flm.read_magnitude_gap(twodim=True)

# Expand arrays so they correspond to each projection
M200m_xyz = np.hstack((flm.M200m, flm.M200m, flm.M200m))
accretion_xyz = np.hstack((flm.accretion, flm.accretion, flm.accretion))
energy_xyz = np.hstack((flm.energy, flm.energy, flm.energy))

properties_all = np.vstack((accretion_xyz, M200m_xyz, energy_xyz, 
                            flm.gap, flm.concentration,
                            flm.symmetry, flm.alignment, flm.centroid))

list_good1 = np.intersect1d(
    np.intersect1d(np.intersect1d(np.where(np.isfinite(properties_all[0,:]))[0], 
                                  np.where(np.isfinite(properties_all[1,:]))[0]),
    np.intersect1d(np.where(np.isfinite(properties_all[2,:]))[0], np.where(np.isfinite(properties_all[3,:]))[0])),
    np.where(np.isfinite(properties_all[4,:]))[0])
list_good2 = np.intersect1d(
    np.intersect1d(np.where(np.isfinite(properties_all[5,:]))[0], 
                                  np.where(np.isfinite(properties_all[6,:]))[0]),
    np.where(np.isfinite(properties_all[7,:]))[0])
list_good = np.intersect1d(list_good1, list_good2)

properties_all = properties_all[:,list_good]

correlations_p = np.corrcoef(properties_all)

labels = ["$\Gamma$", "$M_{\\rm{200m}}$", "$X_{\\rm{E}}$",
          "$\\rm{M14}$", "$c$", "$s$", "$a$",
          r"$\log\langle w \rangle$"]

fig, ax = plt.subplots(1,1)
cb = ax.matshow(correlations_p, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticks(np.arange(0, 8))
ax.set_yticks(np.arange(0, 8))
ax.set_xticklabels(labels=labels, rotation=90)
ax.set_yticklabels(labels=labels)
c = "darkslategrey"
lw = 2
ls = (0,(1,1))
ax.plot((2.5,7.5), (2.5,2.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((2.5,2.5), (2.5,7.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((7.45,7.45), (2.5,7.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((2.5,7.5), (7.45,7.45), color=c, linewidth=lw, linestyle=ls)
plt.subplots_adjust(left=0.15, top=0.85)
# cbaxes = fig.add_axes([0.94, 0.04, 0.03, 0.75]) 
# cbar = fig.colorbar(cb, cax=cbaxes)
plt.colorbar(cb, label=r"$\rho_{\rm{r}}$")
plt.savefig("splashback_data/flamingo/plots/correlations.png", dpi=300)
plt.show()