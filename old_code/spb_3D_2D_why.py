import numpy as np
import matplotlib.pyplot as plt
import splashback
from scipy.stats import spearmanr

sp = splashback.macsis()

N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins2 = 45
log_radii2 = np.linspace(-1, 0.7, N_bins2+1)
rad_mid2 = (10**log_radii2[1:] + 10**log_radii2[:-1]) / 2

path = "splashback_data/macsis/"

#Read DMO data
DMO_3D = np.genfromtxt(path + "density_DMO_macsis_3D.csv", delimiter=",")
sp.DMO_3D = splashback.log_gradients(rad_mid2, DMO_3D)

DMO_2D_x = np.genfromtxt(path + "log_DMO_grad_profiles_x_macsis_median_all.csv", 
                         delimiter=",")
DMO_2D_y = np.genfromtxt(path + "log_DMO_grad_profiles_y_macsis_median_all.csv", 
                         delimiter=",")
DMO_2D_z = np.genfromtxt(path + "log_DMO_grad_profiles_z_macsis_median_all.csv", 
                         delimiter=",")
sp.DMO_2D = np.vstack((DMO_2D_x, DMO_2D_y, DMO_2D_z))

#Read hydro DM only surface density
DM_x = np.genfromtxt(path + "log_DM_only_grad_profiles_x_macsis_median_all.csv",
                     delimiter=",")
DM_y = np.genfromtxt(path + "log_DM_only_grad_profiles_y_macsis_median_all.csv",
                     delimiter=",")
DM_z = np.genfromtxt(path + "log_DM_only_grad_profiles_z_macsis_median_all.csv",
                     delimiter=",")
sp.DM_2D_xyz = np.vstack((DM_x, DM_y, DM_z))

#Calculate splashback radii
sp.RSP_3D_DMO = splashback.R_SP_finding(rad_mid2, sp.DMO_3D)
sp.RSP_2D_DMO = splashback.R_SP_finding(rad_mid, sp.DMO_2D)

sp.RSP_2D_DM = splashback.R_SP_finding(rad_mid, sp.DM_2D_xyz)
sp.calculate_Rsp_2D()
sp.RSP_3D = splashback.R_SP_finding(rad_mid, sp.DM_density_3D)

DMO_3D_xyz = np.hstack((sp.RSP_3D_DMO, sp.RSP_3D_DMO, sp.RSP_3D_DMO))
hydro_3D_xyz = np.hstack((sp.RSP_3D, sp.RSP_3D, sp.RSP_3D))

#Compare values
#"True" splashback v "observable"
correlation_3D_WL = spearmanr(hydro_3D_xyz, sp.RSP_DM_median).correlation
plt.scatter(hydro_3D_xyz, sp.RSP_DM_median, edgecolor="k")
plt.xlabel(r"$R_{\rm{SP, 3D}}$")
plt.ylabel(r"$R_{\rm{SP, WL}}$")
plt.title(r"Hydro, $r_{s}=$" + str(round(correlation_3D_WL,2)))
plt.show()

#"True" splashback v surface density only using DM
correlation_3D_DM = spearmanr(hydro_3D_xyz, sp.RSP_2D_DM).correlation
plt.scatter(hydro_3D_xyz, sp.RSP_2D_DM, edgecolor="k")
plt.xlabel(r"$R_{\rm{SP, 3D}}$")
plt.ylabel(r"$R_{\rm{SP, DM}}$")
plt.title(r"Hydro, $r_{s}=$" + str(round(correlation_3D_DM,2)))
plt.show()

#3D v 2D dark matter densities using DMO simulation data
correlation_DMO = spearmanr(DMO_3D_xyz, sp.RSP_2D_DMO).correlation
plt.scatter(DMO_3D_xyz, sp.RSP_2D_DMO, edgecolor="k")
plt.xlabel(r"$R_{\rm{SP, 3D}}$")
plt.ylabel(r"$R_{\rm{SP, 2D}}$")
plt.title(r"DMO, $r_{s}=$" + str(round(correlation_DMO,2)))
plt.show()


#Look at substructure stuff
substructure_xyz = np.hstack((sp.substructure, sp.substructure, sp.substructure)) 

low_bound = np.percentile(sp.substructure, 25)
mid_bound = np.percentile(sp.substructure, 50)
high_bound = np.percentile(sp.substructure, 75)

sub_1 = np.where(substructure_xyz <= low_bound)[0]
sub_2 = np.where((substructure_xyz > low_bound) & (substructure_xyz <= mid_bound))[0]
sub_3 = np.where((substructure_xyz > mid_bound) & (substructure_xyz <= high_bound))[0]
sub_4 = np.where(substructure_xyz > high_bound)[0]


plt.scatter(hydro_3D_xyz[sub_1], sp.RSP_DM_median[sub_1], 
            edgecolor="k",
            color = "r",
            label=r"Low $f_{\rm{sub}}$")
plt.scatter(hydro_3D_xyz[sub_2], sp.RSP_DM_median[sub_2], 
            edgecolor="k",
            color = "b",
            label=r"Mid-low $f_{\rm{sub}}$")
plt.scatter(hydro_3D_xyz[sub_3], sp.RSP_DM_median[sub_3], 
            edgecolor="k",
            color = "y",
            label=r"Mid-high $f_{\rm{sub}}$")
plt.scatter(hydro_3D_xyz[sub_4], sp.RSP_DM_median[sub_4], 
            edgecolor="k",
            color = "g",
            label=r"High $f_{\rm{sub}}$")
plt.xlabel(r"$R_{\rm{SP, 3D}}$")
plt.ylabel(r"$R_{\rm{SP, WL}}$")
#plt.legend()
plt.title(r"Hydro, $r_{s}=$" + str(round(correlation_3D_WL,2)))
plt.show()

#Stack different substructure fractions
DM_density_lazy = np.vstack((sp.DM_density_3D, sp.DM_density_3D, sp.DM_density_3D))
hydro_3D_sub1 = splashback.stack_data(DM_density_lazy[sub_1])
hydro_3D_sub2 = splashback.stack_data(DM_density_lazy[sub_2])
hydro_3D_sub3 = splashback.stack_data(DM_density_lazy[sub_3])
hydro_3D_sub4 = splashback.stack_data(DM_density_lazy[sub_4])


DM_median_sub1 = splashback.stack_data(sp.DM_median[sub_1])
DM_median_sub2 = splashback.stack_data(sp.DM_median[sub_2])
DM_median_sub3 = splashback.stack_data(sp.DM_median[sub_3])
DM_median_sub4 = splashback.stack_data(sp.DM_median[sub_4])

RSP_hydro_3D_subs = splashback.R_SP_finding(rad_mid, 
                                            np.vstack((hydro_3D_sub1, 
                                                       hydro_3D_sub2, 
                                                       hydro_3D_sub3, 
                                                       hydro_3D_sub3)))
RSP_DM_median_subs = splashback.R_SP_finding(rad_mid,
                                             np.vstack((DM_median_sub1, 
                                                        DM_median_sub2, 
                                                        DM_median_sub3,
                                                        DM_median_sub4)))

plt.scatter(RSP_hydro_3D_subs, RSP_DM_median_subs)
plt.show()

#Stack different numbers of substructure bins
N_sub_bins = 20
order_sub = np.argsort(sp.substructure)
order_sub_xyz = np.argsort(np.hstack((sp.substructure, sp.substructure, sp.substructure)))

DM_3D_stacked_profiles = np.zeros((N_sub_bins, N_bins))
WL_2D_stacked_profiles = np.zeros((N_sub_bins, N_bins))
substructure_splits = np.array_split(order_sub, N_sub_bins)
for i in range(N_sub_bins):
    DM_3D_stacked_profiles[i,:] = splashback.stack_data(sp.DM_density_3D[substructure_splits[i],:])
    WL_2D_stacked_profiles[i,:] = splashback.stack_data(sp.DM_median[substructure_splits[i],:])

RSP_DM_stacked = splashback.R_SP_finding(rad_mid, DM_3D_stacked_profiles)
RSP_WL_stacked = splashback.R_SP_finding(rad_mid, WL_2D_stacked_profiles)

plt.scatter(RSP_DM_stacked, RSP_WL_stacked)
plt.show()

print(spearmanr(RSP_DM_stacked, RSP_WL_stacked).correlation)



