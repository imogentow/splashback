import numpy as np
import splashback as sp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

box = "L1000N1800"
plt.style.use("mnras.mplstyle")
    
flm = sp.flamingo(box, "HF")
flm.read_2D_properties()
flm.read_properties()

flm.sphericity_gas = np.genfromtxt(flm.path + "_sphericity_gas.csv",
                                delimiter=",")
flm.sphericity_DM = np.genfromtxt(flm.path + "_sphericity_DM.csv",
                                delimiter=",")

M200m_xyz = np.hstack((flm.M200m, flm.M200m, flm.M200m))
accretion_xyz = np.hstack((flm.accretion, flm.accretion, flm.accretion))
energy_xyz = np.hstack((flm.energy, flm.energy, flm.energy))
Sgas_xyz = np.hstack((flm.sphericity_gas, flm.sphericity_gas, flm.sphericity_gas))
SDM_xyz = np.hstack((flm.sphericity_DM, flm.sphericity_DM, flm.sphericity_DM))

properties_all = np.vstack((accretion_xyz, M200m_xyz, energy_xyz,
                            Sgas_xyz, SDM_xyz, flm.concentration,
                            flm.symmetry, flm.alignment, flm.centroid))

accretion_finite = np.isfinite(accretion_xyz)
properties_finite = properties_all[:,accretion_finite]
correlations_accretion = np.corrcoef(properties_finite)[:,0]

Sgas_finite = np.isfinite(Sgas_xyz)
properties_finite = properties_all[:,Sgas_finite]
correlations_Sgas = np.corrcoef(properties_finite)[:,3]

SDM_finite = np.isfinite(SDM_xyz)
properties_finite = properties_all[:,SDM_finite]
correlations_SDM = np.corrcoef(properties_finite)[:,4]

correlations_p =np.corrcoef(properties_all)

correlations_p[:,3] = correlations_Sgas
correlations_p[3,:] = correlations_Sgas
correlations_p[:,4] = correlations_SDM
correlations_p[4,:] = correlations_SDM
correlations_p[:,0] = correlations_accretion
correlations_p[0,:] = correlations_accretion

# correlations_s = spearmanr(properties_all, axis=1, nan_policy='omit').correlation
labels = [0, "$\Gamma$", "$M_{\\rm{200m}}$", "$E_{\\rm{kin}}/E_{\\rm{therm}}$",
          "$S_{\\rm{gas}}$", "$S_{\\rm{DM}}$", "$c$", "$s$", "$a$",
          r"$\log\langle w \rangle$"]

fig, ax = plt.subplots(1,1)
cb = ax.matshow(correlations_p, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticklabels(labels=labels, rotation=90)
ax.set_yticklabels(labels=labels)
c = "darkslategrey"
lw = 2
ls = (0,(1,1))
ax.plot((4.5,8.5), (4.5,4.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((4.5,4.5), (4.5,8.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((8.45,8.45), (4.5,8.5), color=c, linewidth=lw, linestyle=ls)
ax.plot((4.5,8.5), (8.45,8.45), color=c, linewidth=lw, linestyle=ls)
# cbaxes = fig.add_axes([0.94, 0.04, 0.03, 0.75]) 
# cbar = fig.colorbar(cb, cax=cbaxes)
plt.colorbar(cb, label=r"$\rho_{\rm{r}}$")
plt.savefig("splashback_data/flamingo/plots/correlations.png", dpi=300)
plt.show()

# fig, ax = plt.subplots(1,1)
# c = ax.matshow(correlations_s, cmap='RdBu', vmin=-1, vmax=1)
# plt.colorbar(c)
# ax.set_xticklabels(labels=labels, rotation=90)
# ax.set_yticklabels(labels=labels)
# c = "darkslategrey"
# lw = 2
# ls = (0,(1,1))
# ax.plot((4.5,8.5), (4.5,4.5), color=c, linewidth=lw, linestyle=ls)
# ax.plot((4.5,4.5), (4.5,8.5), color=c, linewidth=lw, linestyle=ls)
# ax.plot((8.45,8.45), (4.5,8.5), color=c, linewidth=lw, linestyle=ls)
# ax.plot((4.5,8.5), (8.45,8.45), color=c, linewidth=lw, linestyle=ls)
# plt.show()
