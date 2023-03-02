import numpy as np
import matplotlib.pyplot as plt
import splashback 
from scipy import stats
from scipy import interpolate

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

lw = 5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

def mean_median_plots(mean, median, title="none"):
    plt.figure()
    plt.scatter(mean, median, edgecolor="k", color="r")
    plt.xlabel("$R_{SP, mean}$")
    plt.ylabel("$R_{SP, median}$")
    if title != "none":
        plt.title(title)    
    plt.show()
    
    
def compare_observables(obs_1, obs_2, label1="none", label2="none"):
    x_sub = "_{SP," + label1 + "}"
    y_sub = "_{SP," + label2 + "}"
    plt.figure()
    plt.scatter(obs_1, obs_2, edgecolor="k", color="b")
    plt.xlabel(f"$R{x_sub}$")
    plt.ylabel(f"$R{y_sub}$")
    plt.show()

N_eagle = 30

N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2


sp = splashback.splashback()

"""Read observable profiles"""
DM_mean_x, DM_med_x, EM_mean_x, EM_med_x, SD_mean_x, SD_med_x, SZ_mean_x, SZ_med_x = sp.CEAGLE_observables("x")
DM_mean_y, DM_med_y, EM_mean_y, EM_med_y, SD_mean_y, SD_med_y, SZ_mean_y, SZ_med_y = sp.CEAGLE_observables("y")
DM_mean_z, DM_med_z, EM_mean_z, EM_med_z, SD_mean_z, SD_med_z, SZ_mean_z, SZ_med_z = sp.CEAGLE_observables("z")

DMO_x = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_x_ceagle_all.csv", delimiter=",")
DMO_y = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_y_ceagle_all.csv", delimiter=",")
DMO_z = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_z_ceagle_all.csv", delimiter=",")

sp.DM_mean_CE = np.vstack((DM_mean_x, DM_mean_y, DM_mean_z))
sp.DM_median_CE = np.vstack((DM_med_x, DM_med_y, DM_med_z))
    
sp.EM_mean_CE = np.vstack((EM_mean_x, EM_mean_y, EM_mean_z))
sp.EM_median_CE = np.vstack((EM_med_x, EM_med_y, EM_med_z))
   
sp.SD_mean_CE = np.vstack((SD_mean_x, SD_mean_y, SD_mean_z))
sp.SD_median_CE = np.vstack((SD_med_x, SD_med_y, SD_med_z))
  
sp.SZ_mean_CE = np.vstack((SZ_mean_x, SZ_mean_y, SZ_mean_z))
sp.SZ_median_CE = np.vstack((SZ_med_x, SZ_med_y, SZ_med_z))

DMO_all = np.vstack((DMO_x, DMO_y, DMO_z))

"""Read cluster information"""
M_200m = np.genfromtxt("Analysis_files/Splashback/M_200m_ceagle.csv", delimiter=",")
accretion = np.genfromtxt("Analysis_files/Splashback/accretion_rates_ceagle.csv",
                                delimiter=",")
energy_ratio = np.genfromtxt("Analysis_files/Splashback/energy_ratios_C_EAGLE.csv",
                             delimiter=",")

mass_bound = 10**4.65
high_mass = np.where(M_200m > mass_bound)[0]
low_mass = np.where(M_200m <= mass_bound)[0]

accretion_bound = 1.55
high_a = np.where(accretion > accretion_bound)[0]
low_a = np.where(accretion <= accretion_bound)[0]

energy_bound = 0.43
high_E = np.where(energy_ratio > energy_bound)[0]
low_E = np.where(energy_ratio <= energy_bound)[0]

"""Mean v median"""
# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(3,9), sharex=True)
# axes[0].semilogx(rad_mid, sp.DM_median_CE[25,:], linestyle="--", color="k", label="Median")
# axes[0].semilogx(rad_mid, sp.DM_mean_CE[25,:], color="k", label="Mean")
# axes[0].set_ylabel("$d \log \\rho_{DM} / d \log r$")
# axes[0].legend()

# axes[1].semilogx(rad_mid, sp.EM_median_CE[25,:], linestyle="--", color="gold")
# axes[1].semilogx(rad_mid, sp.EM_mean_CE[25,:], color="gold")
# axes[1].set_ylabel("$d \log \\rho_{EM} / d \log r$")
# axes[1].set_ylim((-10,7))

# axes[3].semilogx(rad_mid, sp.SD_median_CE[25,:], linestyle="--", color="r")
# axes[3].semilogx(rad_mid, sp.SD_mean[25,:], color="r")
# axes[3].set_ylabel("$d \log \\rho_{stars} / d \log r$")

# axes[2].semilogx(rad_mid, sp.SZ_median_CE[25,:], linestyle="--", color="c")
# axes[2].semilogx(rad_mid, sp.SZ_mean_CE[25,:], color="c")
# axes[2].set_ylabel("$d \log \\rho_{SZ} / d \log r$")

# axes[3].set_xlabel("$r/R_{200m}$")
# plt.subplots_adjust(left=0.2)
# #plt.savefig("ceagle_mean_median_obs.png", dpi=300)
# plt.show()

stack_DM_median = splashback.stack_data(sp.DM_median_CE[high_mass,:])
stack_DM_mean = splashback.stack_data(sp.DM_mean_CE[high_mass,:])
stack_EM_median = splashback.stack_data(sp.EM_median_CE[high_mass,:])
stack_EM_mean = splashback.stack_data(sp.EM_mean_CE[high_mass,:])
stack_SZ_median = splashback.stack_data(sp.SZ_median_CE[high_mass,:])
stack_SZ_mean = splashback.stack_data(sp.SZ_mean_CE[high_mass,:])

stack_DM_3D = splashback.stack_data(sp.DM_density_CE)
stack_gas_3D = splashback.stack_data(sp.gas_density_CE)
stack_DMO = splashback.stack_data(DMO_all)

x_interp = np.linspace(0.15, 3.5, 200)
interp_DM_median = interpolate.interp1d(rad_mid, stack_DM_median, kind="cubic")
interp_DM_mean = interpolate.interp1d(rad_mid, stack_DM_mean, kind="cubic")
interp_EM_median = interpolate.interp1d(rad_mid, stack_EM_median, kind="cubic")
interp_EM_mean = interpolate.interp1d(rad_mid, stack_EM_mean, kind="cubic")
interp_SZ_median = interpolate.interp1d(rad_mid, stack_SZ_median, kind="cubic")
interp_SZ_mean = interpolate.interp1d(rad_mid, stack_SZ_mean, kind="cubic")
interp_DM_3D = interpolate.interp1d(rad_mid, stack_DM_3D, kind="cubic")
interp_gas_3D = interpolate.interp1d(rad_mid, stack_gas_3D, kind="cubic")
interp_DMO = interpolate.interp1d(rad_mid, stack_DMO, kind="cubic")




# plt.figure(figsize=(8,7))
# plt.semilogx(x_interp, interp_DM_median(x_interp), color="k", linestyle=(0, (0.5,1)), linewidth=lw, label="Median")
# plt.semilogx(x_interp, interp_DM_mean(x_interp), color="k", label="DM", linewidth=lw)
# plt.semilogx(x_interp, interp_EM_median(x_interp), color="r", linestyle=(0, (0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_EM_mean(x_interp), color="r", label="EM", linewidth=lw)
# plt.semilogx(x_interp, interp_SZ_median(x_interp), color="orange", linestyle=(0, (0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_SZ_mean(x_interp), color="orange", label="SZ", linewidth=lw)
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \Sigma / d \log r$")
# plt.xlim((0.15,2))
# plt.legend()
# plt.subplots_adjust(left=0.1, bottom=0.1)
# #plt.savefig("mean_median_stacked_CE.png", dpi=300)
# plt.show()

RSP_DM_mean = np.zeros(3*N_eagle)
RSP_DM_median = np.zeros(3*N_eagle)
    
RSP_EM_mean = np.zeros(3*N_eagle)
RSP_EM_median = np.zeros(3*N_eagle)
    
RSP_SZ_mean = np.zeros(3*N_eagle)
RSP_SZ_median = np.zeros(3*N_eagle)
    
RSP_stellar_mean = np.zeros(3*N_eagle)
RSP_stellar_median = np.zeros(3*N_eagle)

fs = (12,8)

"""Calculate observable splashback radii"""
for i in range(N_eagle*3):
    #RSP_DM_mean[i] = sp.R_SP_finding(rad_mid, sp.DM_mean_CE[i,:])
    RSP_DM_median[i] = sp.R_SP_finding(rad_mid, sp.DM_median_CE[i,:])
        
    #RSP_EM_mean[i] = sp.R_SP_finding(rad_mid, sp.EM_mean_CE[i,:])
    RSP_EM_median[i] = sp.R_SP_finding(rad_mid, sp.EM_median_CE[i,:])
      
    #RSP_SZ_mean[i] = sp.R_SP_finding(rad_mid, sp.SZ_mean_CE[i,:])
    RSP_SZ_median[i] = sp.R_SP_finding(rad_mid, sp.SZ_median_CE[i,:])
        
    #RSP_stellar_mean[i] = sp.R_SP_finding(rad_mid, sp.SD_mean_CE[i,:])
    RSP_stellar_median[i] = sp.R_SP_finding(rad_mid, sp.SD_median_CE[i,:])

    # RSP_DM_median[i] = sp.RSP_finding_new(rad_mid, sp.DM_median_CE[i,:])
    # RSP_EM_median[i] = sp.RSP_finding_new(rad_mid, sp.EM_median_CE[i,:])
    # RSP_SZ_median[i] = sp.RSP_finding_new(rad_mid, sp.SZ_median_CE[i,:])
    # RSP_stellar_median[i] = sp.RSP_finding_new(rad_mid, sp.SD_median_CE[i,:])
    
    
"""Determine splashback radius for 3D profiles"""
sp.SP_radius_DM_CE = np.zeros(N_eagle)
sp.SP_radius_gas_CE = np.zeros(N_eagle)
sp.SP_radius_star_CE = np.zeros(N_eagle)
    
for i in range(N_eagle):
    sp.SP_radius_DM_CE[i] = sp.R_SP_finding(rad_mid, sp.DM_density_CE[i,:])
    sp.SP_radius_gas_CE[i] = sp.R_SP_finding(rad_mid, sp.gas_density_CE[i,:])
    sp.SP_radius_star_CE[i] = sp.R_SP_finding(rad_mid, sp.star_density_CE[i,:])
    
    # grad_plots(DM_density_CE[i,:], gas_density_CE[i,:], star_density_CE[i,:], 
    #             np.array([SP_radius_DM_CE[i], SP_radius_gas_CE[i], SP_radius_star_CE[i]]),
    #             title=str(i))
    
#np.savetxt("R_SP_ceagle_3D_gas.csv", sp.SP_radius_gas_CE, delimiter=",")
RSP_DM_xyz = np.vstack((sp.SP_radius_DM_CE, sp.SP_radius_DM_CE, sp.SP_radius_DM_CE))

lw = 2
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,7), sharex=True, 
                        gridspec_kw={'hspace' : 0, 'wspace' : 0})
axs[0].semilogx(rad_mid, sp.DM_density_CE[3,:],
                color="k", linewidth=lw)
ylim = axs[0].get_ylim()
axs[0].semilogx((sp.SP_radius_DM_CE[3],sp.SP_radius_DM_CE[3]), ylim,
                color="k", linestyle="--", linewidth=lw)
axs[0].set_ylim(ylim)
axs[1].semilogx(rad_mid, sp.EM_median_CE[3,:], linewidth=lw,
                color="orange")
axs[1].semilogx(rad_mid, sp.EM_median_CE[3+30,:], linewidth=lw,
                color="blue")
axs[1].semilogx(rad_mid, sp.EM_median_CE[3+60,:], linewidth=lw,
                color="grey")
ylim = axs[1].get_ylim()
axs[1].semilogx((RSP_EM_median[3], RSP_EM_median[3]), ylim,
                linewidth=lw, linestyle="--", color="orange")
axs[1].semilogx((RSP_EM_median[3+30], RSP_EM_median[3+30]), ylim,
                linewidth=lw, linestyle="--", color="blue")
axs[1].semilogx((RSP_EM_median[3+60], RSP_EM_median[3+60]), ylim,
                linewidth=lw, linestyle="--", color="grey")
axs[1].set_ylim((-7,ylim[1]))
axs[0].set_ylabel("$d \log \\rho_{DM} / d \log r$")
axs[1].set_ylabel("$d \log EM / d \log r$")
plt.xlim((0.1, 3))
plt.xlabel("$r/R_{200m}$")
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig("CE_03_example.png", dpi=300)
plt.show()
    
size=100
# plt.figure()
# plt.scatter(sp.SP_radius_DM_CE, sp.SP_radius_gas_CE, edgecolor="k", color="b", s=size)
# plt.show()


plt.figure()
plt.scatter(sp.SP_radius_DM_CE, RSP_EM_median[:30], edgecolor="k", 
            color="orange", s=size, marker="^")
plt.scatter(sp.SP_radius_DM_CE, RSP_EM_median[30:60], edgecolor="k", 
            color="blue", s=size, marker="^")
plt.scatter(sp.SP_radius_DM_CE, RSP_EM_median[60:], edgecolor="k", 
            color="grey", s=size, marker="^")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, ylim, color="k", linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("$R_{\\rm{SP, DM}} / R_{\\rm{200m}}$")
plt.ylabel("$R_{\\rm{SP, EM}} / R_{\\rm{200m}}$")
plt.xscale('log')
plt.yscale('log')
plt.subplots_adjust(left=0.2, bottom=0.2)
#plt.savefig("Rsp_EM_DM_ceagle.png", dpi=300)
plt.show()

# plt.figure()
# plt.scatter(sp.SP_radius_DM_CE, RSP_SZ_median[:30], edgecolor="k", 
#             color="orange", s=size)
# plt.scatter(sp.SP_radius_DM_CE, RSP_SZ_median[30:60], edgecolor="k", 
#             color="blue", s=size)
# plt.scatter(sp.SP_radius_DM_CE, RSP_SZ_median[60:], edgecolor="k", 
#             color="grey", s=size)
# plt.show()

# plt.figure()
# plt.scatter(RSP_SZ_median[:30], RSP_EM_median[:30], edgecolor="k", 
#             color="orange", s=size)
# plt.scatter( RSP_SZ_median[30:60], RSP_EM_median[30:60], edgecolor="k", 
#             color="blue", s=size)
# plt.scatter(RSP_SZ_median[60:], RSP_EM_median[60:], edgecolor="k", 
#             color="grey", s=size)
# plt.show()


# high_RSP = np.nanpercentile(sp.SP_radius_DM_CE, 75)
# low_RSP = np.nanpercentile(sp.SP_radius_DM_CE, 25)



# plt.figure(figsize=(6,5))
# plt.semilogx(x_interp, interp_DM_3D(x_interp), color="k", label="3D dark matter",
#              linewidth=lw, linestyle=(0, (0.5,1,1.5,1)))
# plt.semilogx(x_interp, interp_DM_median(x_interp), color="k", label="Surface density", 
#              linestyle=(0,(0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_DMO(x_interp), color="k", label="2D dark matter",
#              linewidth=lw)
# ylim = plt.gca().get_ylim()
# plt.fill_betweenx(ylim, low_RSP, high_RSP, alpha=0.5, color="grey")
# plt.ylim(ylim)
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \Sigma/ d \log r$")
# plt.xlim((0.15,2))
# plt.legend()
# plt.subplots_adjust(left=0.18, bottom=0.15)
# #plt.savefig("DM_2D_v_3D_CE.png", dpi=300)
# plt.show()


# plt.figure(figsize=(6,5))
# plt.semilogx(x_interp, interp_gas_3D(x_interp), color="mediumblue", label="3D gas", linewidth=lw)
# plt.semilogx(x_interp, interp_EM_median(x_interp), color="mediumblue", label="EM", 
#              linestyle=(0,(0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_SZ_median(x_interp), color="mediumblue", label="SZ", 
#              linestyle=(0,(0.5,1, 1.5, 1)), linewidth=lw)
# ylim = plt.gca().get_ylim()
# plt.fill_betweenx(ylim, low_RSP, high_RSP, alpha=0.5, color="grey")
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \Sigma / d \log r$")
# plt.ylim(ylim)
# plt.xlim((0.15,2))
# plt.legend()
# plt.subplots_adjust(left=0.18, bottom=0.15)
# #plt.savefig("gas_property_profiles_2D_3D.png", dpi=300)
# plt.show()


# plt.figure(figsize=(8,7))
# plt.semilogx(x_interp, interp_DM_median(x_interp), color="k", linestyle=(0, (0.5,1)), linewidth=lw, label="Median")
# plt.semilogx(x_interp, interp_DM_mean(x_interp), color="k", label="Surface density", linewidth=lw)
# plt.semilogx(x_interp, interp_EM_median(x_interp), color="r", linestyle=(0, (0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_EM_mean(x_interp), color="r", label="Emission measure", linewidth=lw)
# plt.semilogx(x_interp, interp_SZ_median(x_interp), color="orange", linestyle=(0, (0.5,1)), linewidth=lw)
# plt.semilogx(x_interp, interp_SZ_mean(x_interp), color="orange", label="SZ", linewidth=lw)
# ylim = plt.gca().get_ylim()
# plt.fill_betweenx(ylim, low_RSP, high_RSP, alpha=0.5, color="grey")
# plt.ylim(ylim)
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \Sigma / d \log r$")
# plt.xlim((0.15,2))
# plt.legend()
# plt.subplots_adjust(left=0.1, bottom=0.1)
# #plt.savefig("mean_median_stacked_CE.png", dpi=300)
# plt.show()

    
"""Example plot showing well aligned splashback radii from observable profiles."""
# plt.figure(figsize=fs)
# plt.semilogx(rad_mid, sp.DM_median_CE[31,:], color="k", label="Surface density",
#              linewidth=2)
# plt.semilogx(rad_mid, sp.EM_median_CE[31,:], color="gold", label="Emission measure",
#              linewidth=2)
# plt.semilogx(rad_mid, sp.SZ_median_CE[31,:], color="c", label="SZ effect",
#              linewidth=2)
# plt.semilogx(rad_mid, sp.SD_median_CE[31,:], color="r", label="Stellar density",
#              linewidth=2)
# axes = plt.gca()
# ylim = axes.get_ylim()
# plt.semilogx([RSP_DM_median[31], RSP_DM_median[31]], ylim, color="k",
#              linewidth=2)
# plt.semilogx([RSP_EM_median[31], RSP_EM_median[31]], ylim, color="gold",
#              linewidth=2)
# plt.semilogx([RSP_SZ_median[31], RSP_SZ_median[31]], ylim, color="c",
#              linewidth=2)
# plt.semilogx([RSP_stellar_median[31], RSP_stellar_median[31]], ylim, color="r",
#              linewidth=2)
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# plt.ylim(ylim)
# plt.savefig("CE_01_obs_profs.png", dpi=300)
# plt.show()
    

"""Mean v median observable plots"""
# mean_median_plots(RSP_DM_mean, RSP_DM_median, title="Surface density")
# mean_median_plots(RSP_EM_mean, RSP_EM_median, title="Emission measure")
# mean_median_plots(RSP_SZ_mean, RSP_SZ_median, title="SZ")
# mean_median_plots(RSP_stellar_mean, RSP_stellar_median, title="Stellar density")


"""Comparing observables"""
# compare_observables(RSP_DM_median, RSP_EM_median, label1="DM", label2="EM")   
# compare_observables(RSP_DM_median, RSP_SZ_median, label1="DM", label2="SZ")   
# compare_observables(RSP_DM_median, RSP_stellar_median, label1="DM", label2="Stellar")   

# compare_observables(RSP_EM_median, RSP_SZ_median, label1="EM", label2="SZ")   
# compare_observables(RSP_EM_median, RSP_stellar_median, label1="EM", label2="Stellar")   

# compare_observables(RSP_stellar_median, RSP_SZ_median, label1="Stellar", label2="SZ")   
    

    
"""Example plot showing well matching splashback radii from 3D density profiles"""
# plt.figure(figsize=fs)
# plt.semilogx(rad_mid, sp.DM_density_CE[1,:], label="Dark matter", color="k",
#              linewidth=2)
# plt.semilogx(rad_mid, sp.gas_density_CE[1,:], label="Gas", color="gold",
#              linewidth=2)
# plt.semilogx(rad_mid, sp.star_density_CE[1,:], label="Stars", color="r",
#              linewidth=2)
# axes = plt.gca()
# ylim = axes.get_ylim()
# plt.semilogx([sp.SP_radius_DM_CE[1], sp.SP_radius_DM_CE[1]], ylim, color="k",
#              linewidth=2)
# plt.semilogx([sp.SP_radius_gas_CE[1], sp.SP_radius_gas_CE[1]], ylim, color="gold",
#              linewidth=2)
# plt.semilogx([sp.SP_radius_star_CE[1], sp.SP_radius_star_CE[1]], ylim, color="r",
#              linewidth=2)
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# plt.ylim(ylim)
# plt.savefig("CE_01_3D_profs.png", dpi=300)
# plt.show()
    
    
"""Stacked data profiles"""
# DM_stacked = splashback.stack_data(sp.DM_median_CE)
# EM_stacked = splashback.stack_data(sp.EM_median_CE)  
# SZ_stacked = splashback.stack_data(sp.SZ_median_CE)  
# SD_stacked = splashback.stack_data(sp.SD_median_CE)  
    
# plt.figure()
# plt.semilogx(rad_mid, DM_stacked, color="k", label="DM")
# plt.semilogx(rad_mid, EM_stacked, color="gold", label="EM")
# plt.semilogx(rad_mid, SZ_stacked, color="c", label="SZ")
# plt.semilogx(rad_mid, SD_stacked, color="r", label="Stars")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# plt.show()

# DM_stacked_3D = splashback.stack_data(sp.DM_density_CE)
# gas_stacked_3D = splashback.stack_data(sp.gas_density_CE)
# stars_stacked_3D = splashback.stack_data(sp.star_density_CE)

# plt.figure()
# plt.semilogx(rad_mid, DM_stacked_3D, color="k", label="DM")
# plt.semilogx(rad_mid, gas_stacked_3D, color="gold", label="Gas")
# plt.semilogx(rad_mid, stars_stacked_3D, color="r", label="Stars")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# plt.show()


"""Histograms of splashback radii values"""

# mean, std, skew = splashback.RSP_histogram(np.vstack((RSP_DM_mean, RSP_EM_mean, RSP_stellar_mean, RSP_SZ_mean)),
#                                            ["DM", "EM", "Stars", "SZ"], title="2D mean profiles, z = 0", N_bins=10)
    
# print("RADIUS DISTRIBUTIONS: OBSERVABLE PROFILES")
# print("MEAN")
# print("DARK MATTER")
# print(mean[0], std[0], skew[0])

# print("EMISSION MEASURE")
# print(mean[1], std[1], skew[1])
    
# print("SUNYAEV ZEL'DOVICH")
# print(mean[2], std[2], skew[2])

# print("STELLAR DENSITY")
# print(mean[3], std[3], skew[3])


# mean, std, skew = splashback.RSP_histogram(np.vstack((RSP_DM_median, RSP_EM_median, RSP_stellar_median, RSP_SZ_median)),
#                                            ["DM", "EM", "Stars", "SZ"], title="2D median profiles, z = 0", N_bins=10)
    
# print("RADIUS DISTRIBUTIONS: OBSERVABLE PROFILES")
# print("MEDIAN")
# print("DARK MATTER")
# print(mean[0], std[0], skew[0])

# print("EMISSION MEASURE")
# print(mean[1], std[1], skew[1])
    
# print("SUNYAEV ZEL'DOVICH")
# print(mean[2], std[2], skew[2])

# print("STELLAR DENSITY")
# print(mean[3], std[3], skew[3])



# mean, std, skew = splashback.RSP_histogram(np.vstack((sp.SP_radius_DM_CE, sp.SP_radius_gas_CE, sp.SP_radius_star_CE)),
#                                            ["DM", "Gas", "Stars"], title="3D profiles, z = 0", N_bins=10)
    
# print("RADIUS DISTRIBUTIONS: OBSERVABLE PROFILES")
# print("DARK MATTER")
# print(mean[0], std[0], skew[0])

# print("GAS")
# print(mean[1], std[1], skew[1])
    
# print("STELLAR DENSITY")
# print(mean[2], std[2], skew[2])


"""Compare splashback radii determined from different types of matter"""

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,8), gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
# axes[0].scatter(sp.SP_radius_DM_CE, sp.SP_radius_gas_CE, color="cornflowerblue",
#                 edgecolor="k")
# axes[0].set_xlabel("$R_{SP,\mathrm{DM}}$")
# axes[0].set_ylabel("$R_{SP,\mathrm{gas}}$")

# axes[1].scatter(sp.SP_radius_gas_CE, sp.SP_radius_star_CE, color="cornflowerblue",
#                 edgecolor="k")
# axes[1].set_xlabel("$R_{SP,\mathrm{gas}}$")
# axes[1].set_ylabel("$R_{SP,\mathrm{star}}$")

# axes[2].scatter(sp.SP_radius_star_CE, sp.SP_radius_DM_CE, color="cornflowerblue",
#                 edgecolor="k")
# axes[2].set_ylabel("$R_{SP,\mathrm{DM}}$")
# axes[2].set_xlabel("$R_{SP,\mathrm{star}}$")
# plt.subplots_adjust(left=0.2)
# #plt.savefig("macsis_3D_RSP_compare.png", dpi=300)
# plt.show()


"""Compare splashback radii determined from different observable profiles"""
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,9), gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
# axes[0].scatter(RSP_DM_median, RSP_EM_median, color="cornflowerblue",
#                 edgecolor="k")
# axes[0].set_xlabel("$R_{SP,\mathrm{DM}}$")
# axes[0].set_ylabel("$R_{SP,\mathrm{EM}}$")
# ylim = axes[0].get_ylim()

# star_mask = np.where(RSP_stellar_median > ylim[0])[0]

# axes[1].scatter(RSP_EM_median[star_mask], RSP_stellar_median[star_mask], color="cornflowerblue",
#                 edgecolor="k")
# axes[1].set_xlabel("$R_{SP,\mathrm{EM}}$")
# axes[1].set_ylabel("$R_{SP,\mathrm{star}}$")

# axes[2].scatter(RSP_stellar_median[star_mask], RSP_DM_median[star_mask], color="cornflowerblue",
#                 edgecolor="k")
# axes[2].set_ylabel("$R_{SP,\mathrm{DM}}$")
# axes[2].set_xlabel("$R_{SP,\mathrm{star}}$")
# plt.subplots_adjust(left=0.2)
# #plt.savefig("macsis_obs_RSP_compare.png", dpi=300)
# plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,9), gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
# axes[0].scatter(RSP_SZ_median, RSP_EM_median, color="cornflowerblue",
#                 edgecolor="k")
# axes[0].set_xlabel("$R_{SP,\mathrm{SZ}}$")
# axes[0].set_ylabel("$R_{SP,\mathrm{EM}}$")
# ylim = axes[0].get_ylim()

# star_mask = np.where(RSP_stellar_median > ylim[0])[0]

# axes[1].scatter(RSP_SZ_median[star_mask], RSP_stellar_median[star_mask], color="cornflowerblue",
#                 edgecolor="k")
# axes[1].set_xlabel("$R_{SP,\mathrm{SZ}}$")
# axes[1].set_ylabel("$R_{SP,\mathrm{star}}$")

# axes[2].scatter(RSP_SZ_median, RSP_DM_median, color="cornflowerblue",
#                 edgecolor="k")
# axes[2].set_ylabel("$R_{SP,\mathrm{DM}}$")
# axes[2].set_xlabel("$R_{SP,\mathrm{SZ}}$")
# plt.subplots_adjust(left=0.2)
# #plt.savefig("macsis_obs_RSP_compare.png", dpi=300)
# plt.show()


DM = np.array([0, 3, 4, 11, 20,23,27,28,29,34,38,42,43,46,50,52,54,57,64,66,72,88])
stars = np.array([0,2,4,5,12,15,22,27,28,29,34,35,41,51,53,54,57,58,65,66,67,68,73,80,84,88])
both = np.array([0,4,27,28,29,34,54,57,66,88])
single = np.array([3,11,20,23,38,42,43,46,50,52,64,72, 2,5,12,15,22,35,41,51,53,58,65,67,68,73,80,84])
either = np.hstack((both, single))
neither = np.arange(0,90,1)
neither = np.delete(neither, both)

DM_only_x = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_x_ceagle_all.csv", delimiter=",")
DM_only_y = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_y_ceagle_all.csv", delimiter=",")
DM_only_z = np.genfromtxt("Analysis_files/Splashback/log_DM_only_grad_profiles_z_ceagle_all.csv", delimiter=",")
DM_only_all = np.vstack((DM_only_x, DM_only_y, DM_only_z))
# RSP_DM_only = np.zeros(90)
# for i in range(90):
#     RSP_DM_only[i] = sp.R_SP_finding(rad_mid, DM_only_all[i,:])

# plt.figure(figsize=(4,4))
# plt.plot([0,2.0], [0,2.0], linestyle="--", color="k")
# plt.scatter(RSP_DM_median, RSP_stellar_median, color="cornflowerblue",
#             edgecolor="k", s=75, alpha=0.6)
# plt.xlabel("$R_{SP,\mathrm{DM}}$")
# plt.ylabel("$R_{SP,\mathrm{star}}$")
# plt.xlim((0,2.0))
# plt.ylim((0,2.0))
# #plt.savefig("star_dm_obs_RSP.png", dpi=300)
# plt.show()

# top = np.where((RSP_stellar_median > 1.27))[0]
# bottom = np.where((RSP_stellar_median < 0.6) & (RSP_DM_only > 0.75))[0]

# plt.figure(figsize=(4,4))
# plt.plot([0,2.0], [0,2.0], linestyle="--", color="k")
# plt.scatter(RSP_DM_only, RSP_stellar_median, color="cornflowerblue",
#             edgecolor="k", s=75, alpha=0.6)
# plt.xlabel("$R_{SP,\mathrm{DMO}}$")
# plt.ylabel("$R_{SP,\mathrm{star}}$")
# plt.xlim((0,2.0))
# plt.ylim((0,2.0))
# #plt.savefig("star_dm_obs_RSP.png", dpi=300)
# plt.show()

# for i in top:
#     plt.figure()
#     plt.semilogx(rad_mid, DM_only_all[i,:], color="k")
#     plt.semilogx(rad_mid, sp.SD_median_CE[i,:], color="r")
#     ylim = plt.gca().get_ylim()
#     plt.semilogx([RSP_DM_only[i], RSP_DM_only[i]], ylim, linestyle="--", color="k", label="DM")
#     plt.semilogx([RSP_stellar_median[i], RSP_stellar_median[i]], ylim, linestyle="--", 
#                  color="r", label="Stars")
#     plt.legend()
#     plt.title(i)
#     plt.show()

# RSP_DM_3D = np.hstack((sp.SP_radius_DM_CE, sp.SP_radius_DM_CE, sp.SP_radius_DM_CE))

# lobf = lambda m, x, c : m*x + c

# DM = stats.linregress(RSP_DM_3D[neither], RSP_DM_median[neither])
# EM = stats.linregress(RSP_DM_3D[neither], RSP_EM_median[neither])
# SZ = stats.linregress(RSP_DM_3D[neither], RSP_SZ_median[neither])
# stellar = stats.linregress(RSP_DM_3D[neither], RSP_stellar_median[neither])
# print(np.mean(RSP_DM_median[neither] / RSP_DM_3D[neither]))
# print(np.mean(RSP_EM_median[neither] / RSP_DM_3D[neither]))
# print(np.mean(RSP_SZ_median[neither] / RSP_DM_3D[neither]))
# print(np.mean(RSP_stellar_median[neither] / RSP_DM_3D[neither]))

# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4,14), gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
# axes[0].scatter(RSP_DM_3D[neither], RSP_DM_median[neither], color="k",
#                 edgecolor="k")
# axes[0].set_ylabel("$R_{SP,\mathrm{DM}}$")
# ylim = axes[0].get_ylim()
# xlim = axes[0].get_xlim()
# ylobf = (lobf(DM.slope, xlim[0], DM.intercept), lobf(DM.slope, xlim[1], DM.intercept)) 
# axes[0].plot(xlim, ylobf, linestyle="--", color="k")
# axes[0].plot(xlim, ylim, color="k")
# axes[0].set_xlim(xlim)
# axes[0].set_ylim(ylim)

# axes[1].scatter(RSP_DM_3D[neither], RSP_EM_median[neither], color="gold",
#                 edgecolor="k")
# axes[1].set_ylabel("$R_{SP,\mathrm{EM}}$")
# ylim = axes[1].get_ylim()
# xlim = axes[1].get_xlim()
# ylobf = (lobf(EM.slope, xlim[0], EM.intercept), lobf(EM.slope, xlim[1], EM.intercept)) 
# axes[1].plot(xlim, ylobf, linestyle="--", color="darkkhaki")
# axes[1].plot(xlim, ylim, color="k")
# axes[1].set_xlim(xlim)
# axes[1].set_ylim(ylim)

# axes[2].scatter(RSP_DM_3D[neither], RSP_SZ_median[neither], color="c",
#                 edgecolor="k")
# axes[2].set_ylabel("$R_{SP,\mathrm{SZ}}$")
# ylim = axes[2].get_ylim()
# xlim = axes[2].get_xlim()
# ylobf = (lobf(SZ.slope, xlim[0], SZ.intercept), lobf(SZ.slope, xlim[1], SZ.intercept)) 
# axes[2].plot(xlim, ylobf, linestyle="--", color="lightseagreen")
# axes[2].plot(xlim, ylim, color="k")
# axes[2].set_xlim(xlim)
# axes[2].set_ylim(ylim)

# axes[3].scatter(RSP_DM_3D[neither], RSP_stellar_median[neither], color="r",
#                 edgecolor="k")
# axes[3].set_ylabel("$R_{SP,\mathrm{star}}$")
# axes[3].set_xlabel("$R_{SP}$")
# ylim = axes[3].get_ylim()
# xlim = axes[3].get_xlim()
# ylobf = (lobf(stellar.slope, xlim[0], stellar.intercept), lobf(stellar.slope, xlim[1], stellar.intercept)) 
# axes[3].plot(xlim, ylobf, linestyle="--", color="firebrick")
# axes[3].plot(xlim, ylim, color="k")
# axes[3].set_xlim(xlim)
# axes[3].set_ylim(ylim)

# plt.subplots_adjust(left=0.4)
# plt.savefig("ceagle_obs_DM_3D.png", dpi=300)
# plt.show()