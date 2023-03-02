import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr

plt.style.use("mnras.mplstyle")

path = "splashback_data/flamingo"
box = "L1000N1800"

R_sp = np.genfromtxt(path + "/" + box + "_R_sp_evolution.csv",
                      delimiter=",") #r/R200m
R200m = np.genfromtxt(path + "/" + box + "_R200m_evolution.csv",
                      delimiter=",") #comoving Mpc
M200m = np.genfromtxt(path + "/" + box + "_M200m_evolution.csv",
                      delimiter=",") #Msol (comoving?)

R_sp[np.where(R_sp == 0)] = np.nan
R200m[np.where(R200m == 0)] = np.nan
M200m[np.where(M200m == 0)] = np.nan
N_clusters = R_sp.shape[0]

index_dif = 2
local_accretion = np.log10(M200m[:,:-index_dif]) - np.log10(M200m[:,index_dif:]) #-1*np.diff(np.log10(M200m)) #mass gained since last snapshot
R_sp_change = np.diff(R_sp)[:,:-(index_dif-1)]
plt.scatter(local_accretion, R_sp_change, edgecolor="k")
plt.xlim((-1,1))
plt.show()

snapshots = np.arange(77, 57, -1)

R_sp_Mpc = R_sp * R200m
cm = plt.cm.get_cmap('rainbow_r')
for i in range(30):
    plt.figure()
    plt.plot(M200m[i,:], R_sp[i,:],
             color="k", alpha=0.4)
    plt.scatter(M200m[i,:], R_sp[i,:],
                edgecolor="k", c=snapshots, cmap=cm)
    plt.xscale('log')
    plt.xlabel("$M_{200m}$")
    plt.ylabel("$R_{\\rm{SP}}/R_{200m}$")
    plt.title(i)
    plt.show()
    
#R_sp = R_sp * R200m 

# plt.figure()
# cm = plt.cm.get_cmap('rainbow_r')
# for i in range(100):
#     plt.scatter(local_accretion[i,:], R_sp[i,:-2], linewidth=1, 
#                 c=snapshots[:-index_dif], alpha=0.6, cmap=cm)
    
# plt.colorbar()
# plt.xlabel("$ d \log M_{200m}$")
# plt.ylabel("$R_{SP} / R_{200m}$")
# plt.xlim((-0.05,0.1))
# # plt.savefig("splashback_data/flamingo/plots/splashback_local_accretion.png",
# #             dpi=300)
# plt.show()

# plt.figure()
# for i in range(10):
#     plt.plot(snapshots, R_sp[i,:], linewidth=1)
    
# plt.xlabel("$ d \log M_{200m}$")
# plt.ylabel("$R_{SP} / R_{200m}$")
# #plt.xlim((-0.05,0.1))
# # plt.savefig("splashback_data/flamingo/plots/splashback_local_accretion.png",
# #             dpi=300)
# plt.show()

#########################
# ID = 10
# log_density_test = np.genfromtxt(path + "/log_density_evolution." + str(ID)
#                                   + ".csv", delimiter=",")
# N_z = log_density_test.shape[0]
# N_bins = log_density_test.shape[1]
# log_radii = np.linspace(-1, 0.7, N_bins + 1)
# rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

# RSP_test = dr.depth_cut(rad_mid, log_density_test)

# for i in range(N_z):
#     plt.semilogx(rad_mid, log_density_test[i,:], color="b")
#     ylim = plt.gca().get_ylim()
#     plt.plot((RSP_test[i], RSP_test[i]), ylim, color="grey", linestyle="--")
#     if ylim[1] > 2:
#         ylim = (ylim[0], 2)
#     plt.ylim(ylim)
#     plt.title("z=" + str(i*0.05))
#     plt.show()
    
# plt.plot(RSP_test)
# plt.title(ID)
# plt.show()

# plt.plot(M200m[ID,:])
# plt.title(ID)
# plt.show()

# plt.plot(M200m[ID,:], RSP_test)
# plt.title(ID)
# plt.show()