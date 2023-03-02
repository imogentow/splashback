import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import macsis_convert_halo_IDs as hIDs

mass_fraction_cut = 5
radii_cut = 2

path = "splashback_data/macsis/"
filename = "clusters_" + str(mass_fraction_cut) + "_percent_" + str(radii_cut) + "R200m.csv"
# cluster_list = np.genfromtxt(path + filename, delimiter=",")
cluster_list = hIDs.convert(path+filename)
cluster_list = np.array(cluster_list, dtype=int)

other_list = np.delete(np.arange(390), cluster_list)

print(len(cluster_list))

mcs = sp.macsis()

DM_density_3D_near = sp.stack_data(mcs.DM_density_3D[cluster_list,:])
DM_density_3D_far = sp.stack_data(mcs.DM_density_3D[other_list,:])

plt.semilogx(mcs.rad_mid, DM_density_3D_near, color="k", label="Nearby groups")
plt.semilogx(mcs.rad_mid, DM_density_3D_far, color="r", label="No nearby groups")
plt.legend()
plt.xlabel("$r/R_{200m}$")
plt.ylabel("$ d \log \\rho_{DM} / d \log r$")
plt.title("Exclude clusters of " + str(mass_fraction_cut) + "% mass within " 
          + str(radii_cut) + "R$_{200m}$")
filename = "DM_3D_profiles_FoF_" + str(mass_fraction_cut) + "_percent_" + str(radii_cut) + "R200m.png"
plt.savefig(filename, dpi=300)
plt.show()

# for i in cluster_list:
#     plt.figure()
#     plt.semilogx(mcs.rad_mid, mcs.DM_density_3D[int(i),:], color="k")
#     plt.xlabel("$r/R_{200m}$")
#     plt.ylabel("$d \log \\rho / d \log r$")
#     plt.show()