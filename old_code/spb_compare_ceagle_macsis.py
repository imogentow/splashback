import numpy as np
import matplotlib.pyplot as plt
import splashback as sp

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


path = "splashback_data/macsis/"

N_macsis = 390

N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

mcs = sp.macsis()

# Calculate splashback radius

mcs.SP_radius_DM_3D = sp.R_SP_finding(rad_mid, mcs.DM_density_3D)
mcs.SP_radius_gas_3D = sp.R_SP_finding(rad_mid, mcs.gas_density_3D)
mcs.SP_radius_star_3D = sp.R_SP_finding(rad_mid, mcs.star_density_3D)

DM_3D_RSP_xyz = np.hstack((mcs.SP_radius_DM_3D, mcs.SP_radius_DM_3D, mcs.SP_radius_DM_3D))

# Read in splashback data from C-EAGLE and MACSIS observables
RSP_DM_CE = np.genfromtxt("splashback_data/R_SP_ceagle_3D_DM.csv", delimiter=",")
RSP_gas_CE = np.genfromtxt("splashback_data/R_SP_ceagle_3D_gas.csv", delimiter=",")

plt.figure()
plt.scatter(mcs.SP_radius_DM_3D, mcs.SP_radius_gas_3D, s=100, edgecolor="k", 
            color="cornflowerblue", label="MACSIS", zorder=2)
plt.scatter(RSP_DM_CE, RSP_gas_CE, s=100, edgecolor="k", color="red",
            label="C-EAGLE", zorder=3, marker="^")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, ylim, color="k", linestyle="--", zorder=1)
plt.legend()
plt.xlabel("$R_{\\rm{SP,DM}} / R_{200m}$")
plt.ylabel("$R_{\\rm{SP,gas}} / R_{200m}$")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xscale('log')
plt.yscale('log')
plt.subplots_adjust(left=0.2, bottom=0.2)
#plt.savefig("Rsp_gas_DM_macsis_ceagle.png", dpi=300)
plt.show()