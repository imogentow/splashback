import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import splashback as sp
from scipy import interpolate
from scipy.stats import spearmanr

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

morph = np.genfromtxt("../morphology_rel_xyz.csv", delimiter=",")
rel = np.where(morph == 1)[0]
per = np.where(morph == 0)[0]

def mean_median_obs(data):
    """Compares splashback radii determined from mean and median profiles""" 
    plt.figure()
    plt.scatter(data.SP_DM_mean, data.SP_DM_median, edgecolor="k", color="k", label="DM")
    plt.scatter(data.SP_EM_mean, data.SP_EM_median, edgecolor="k", color="r", label="EM")
    plt.scatter(data.SP_star_mean, data.SP_star_median, edgecolor="k", color="gold", label="Stars")
    plt.plot([0.4, 1.7], [0.4, 1.7], linestyle="--", color="k")
    plt.xlabel("$R_{SP, mean}$")
    plt.ylabel("$R_{SP, median}$")
    plt.legend()
    plt.ylim((0.35,1.7))
    plt.show()
    
    plt.figure()
    plt.scatter(data.SP_DM_mean, data.SP_EM_mean, edgecolor="k", color="r")
    plt.xlabel("$R_{SP, DM}$")
    plt.ylabel("$R_{SP, EM}$")
    plt.show()
    
    plt.figure()
    plt.scatter(data.SP_DM_mean, data.SP_star_mean, edgecolor="k", color="r")
    plt.xlabel("$R_{SP, DM}$")
    plt.ylabel("$R_{SP, Star}$")
    plt.show()
        
    plt.figure()
    plt.scatter(data.SP_star_mean, data.SP_EM_mean, edgecolor="k", color="r")
    plt.xlabel("$R_{SP, Star}$")
    plt.ylabel("$R_{SP, EM}$")
    plt.show()
        
        
    plt.figure()
    plt.scatter(data.SP_DM_median, data.SP_EM_median, edgecolor="k", color="b")
    plt.xlabel("$R_{SP, DM}$")
    plt.ylabel("$R_{SP, EM}$")
    plt.show()
        
    plt.figure()
    plt.scatter(data.SP_DM_median, data.SP_star_median, edgecolor="k", color="b")
    plt.xlabel("$R_{SP, DM}$")
    plt.ylabel("$R_{SP, Star}$")
    plt.ylim((0.35,1.7))
    plt.show()
        
    plt.figure()
    plt.scatter(data.SP_star_median, data.SP_EM_median, edgecolor="k", color="b")
    plt.xlabel("$R_{SP, Star}$")
    plt.ylabel("$R_{SP, EM}$")
    plt.xlim((0.35,1.7))
    plt.show()

path = "splashback_data/macsis/"

N_macsis = 390

N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins2 = 45
log_radii2 = np.linspace(-1, 0.7, N_bins2+1)
radii_bins = 10 ** log_radii2
radii_200m = (radii_bins[:-1] + radii_bins[1:]) / 2

rad_bins = np.linspace(0.2,1.8, 20)

mcs = sp.macsis()

###Calculate splashback radius

mcs.calculate_Rsp_2D()

mcs.SP_DM_3D = sp.R_SP_finding(rad_mid, mcs.DM_density_3D)
mcs.SP_gas_3D = sp.R_SP_finding(rad_mid, mcs.gas_density_3D)
mcs.SP_stars_3D = sp.R_SP_finding(rad_mid, mcs.star_density_3D)

def DM_profile_example(data, index=86):
    """Plot of example DM log grad profile"""
    x_interp = np.linspace(0.2,3.8, 200)
    interp = interpolate.interp1d(rad_mid, 
                                  data.DM_density_3D[index,:],
                                  kind="cubic")
    y_interp = interp(x_interp)
    
    plt.figure(figsize=(5,4))
    plt.semilogx(x_interp, y_interp, color="k")
    ylim = plt.gca().get_ylim()
    plt.semilogx((data.SP_DM_3D[86], data.SP_DM_3D[86]), 
                 ylim, color="k", linestyle="--")
    
    plt.ylim(ylim)
    plt.xlabel("$r/R_{200m}$")
    plt.ylabel("$d \log \\rho_{\mathrm{DM}} / d \log r$")
    plt.subplots_adjust(left=0.2, bottom=0.2)
    # plt.savefig("DM_splashback_profile_example_86.png", dpi=300)
    plt.show()


def compare_R_SP_3D(data):
    """ Compare splashback radii determined from different types of matter
    using 3D profiles."""
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,8), 
                             gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
    
    axes[0].scatter(data.SP_DM_3D, data.SP_gas_3D, 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[0].set_xlabel("$R_{SP,\mathrm{DM}}$")
    axes[0].set_ylabel("$R_{SP,\mathrm{gas}}$")
    
    axes[1].scatter(data.SP_gas_3D, data.SP_stars_3D, 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[1].set_xlabel("$R_{SP,\mathrm{gas}}$")
    axes[1].set_ylabel("$R_{SP,\mathrm{star}}$")
    
    axes[2].scatter(data.SP_stars_3D, data.SP_DM_3D, 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[2].set_ylabel("$R_{SP,\mathrm{DM}}$")
    axes[2].set_xlabel("$R_{SP,\mathrm{star}}$")
    
    plt.subplots_adjust(left=0.2)
    # plt.savefig("macsis_3D_RSP_compare.png", dpi=300)
    plt.show()


def compare_R_SP_2D(data):
    """Compare splashback radii determined from different observable profiles."""
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,8), 
                             gridspec_kw={'hspace' : 0.3, 'wspace' : 0})
    axes[0].scatter(data.SP_DM_median, data.SP_EM_median, 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[0].set_xlabel("$R_{SP,\mathrm{DM}}$")
    axes[0].set_ylabel("$R_{SP,\mathrm{EM}}$")
    ylim = axes[0].get_ylim()
    
    star_mask = np.where(data.SP_SD_median > ylim[0])[0]
    
    axes[1].scatter(data.SP_EM_median[star_mask], 
                    data.SP_SD_median[star_mask], 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[1].set_xlabel("$R_{SP,\mathrm{EM}}$")
    axes[1].set_ylabel("$R_{SP,\mathrm{star}}$")
    
    axes[2].scatter(data.SP_SD_median[star_mask], 
                    data.SP_DM_median[star_mask], 
                    color="cornflowerblue",
                    edgecolor="k")
    axes[2].set_ylabel("$R_{SP,\mathrm{DM}}$")
    axes[2].set_xlabel("$R_{SP,\mathrm{star}}$")
    
    plt.subplots_adjust(left=0.2)
    # plt.savefig("macsis_obs_RSP_compare.png", dpi=300)
    plt.show()
    

def compare_RSP_2D_3D(data):
    """Compare the splashback obtained using 3D density profiles to splashback 
    radius obtained from potential observables"""
    data_3D_xyz = np.hstack((data.SP_DM_3D, data.SP_DM_3D, data.SP_DM_3D))

    plt.figure()
    plt.scatter(data_3D_xyz, data.SP_DM_median)
    plt.xlabel("$R_{SP,3D}$")
    plt.ylabel("$R_{SP,2D}$")
    plt.show()
    
    print(spearmanr(data.SP_DM_median, data_3D_xyz).correlation)


def compare_R_SP_good_bad(data):
    """Compare Rsp values while looking at quality of Rsp identification. 
    Need to change if you want to compare other things than gas and DM"""
    data.define_bad_halos()
    
    plt.figure()
    plt.scatter(data.SP_DM_3D[data.good_DM_gas], 
                data.SP_gas_3D[data.good_DM_gas], 
                edgecolor="k", color="red", marker="o", 
                label="Well matched")
    plt.scatter(data.SP_DM_3D[data.poor_DM_gas], 
                data.SP_gas_3D[data.poor_DM_gas], 
                edgecolor="k", color="none", marker="s", 
                label="Poor matched")
    plt.scatter(data.SP_DM_3D[data.good_DM_poor_gas], 
                data.SP_gas_3D[data.good_DM_poor_gas],
                edgecolor="k", color="none", marker="o", 
                label="Well matched DM, poor gas")
    plt.scatter(data.SP_DM_3D[data.poor_DM_good_gas],
                data.SP_gas_3D[data.poor_DM_good_gas],
                edgecolor="k", color="blue", marker="s", 
                label="Well matched gas, poor DM")
    
    plt.plot(np.arange(3), np.arange(3), linestyle="--", color="k")
    
    plt.xlim((0.45,2.05))
    plt.ylim((0.45, 2.05))
    plt.xlabel("$R_{SP,DM}$")
    plt.ylabel("$R_{SP,gas}$")
    plt.legend()
    # plt.savefig("R_SP_DM_v_gas.png", dpi=300)
    plt.show()
    

def compare_mean_median_profiles(data, index=86):
    """Compare mean and median profiles.
    Index=86 for macsis, chnage for ceagle"""
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,8), sharex=True)
    axes[0].semilogx(rad_mid, data.DM_median[index,:], 
                     linestyle="--", 
                     color="k", 
                     label="Median")
    axes[0].semilogx(rad_mid, data.DM_mean[index,:], 
                     color="k", 
                     label="Mean")
    axes[0].set_ylabel("$d \log \\rho_{DM} / d \log r$")
    axes[0].legend()
    
    axes[1].semilogx(rad_mid, data.EM_median[index,:], linestyle="--", color="gold")
    axes[1].semilogx(rad_mid, data.EM_mean[index,:], color="gold")
    axes[1].set_ylabel("$d \log \\rho_{EM} / d \log r$")
    axes[1].set_ylim((-7,3))
    
    axes[2].semilogx(rad_mid, data.SD_median[index,:], linestyle="--", color="r")
    axes[2].semilogx(rad_mid, data.SD_mean[index,:], color="r")
    axes[2].set_ylabel("$d \log \\rho_{\star} / d \log r$")
    axes[2].set_xlabel("$r/R_{200m}$")
    plt.subplots_adjust(left=0.2)
    # plt.savefig("macsis_mean_median_obs.png", dpi=300)
    plt.show()


def compare_3D_density_profiles(data, index=86):
    """Compare different observable profiles, DM, EM, stellar density and SZ.
    Index set to 86 for a good example from MACSIS."""
    
    plt.figure()
    plt.semilogx(rad_mid, data.DM_median[index,:], 
                 color="k", label="DM")
    plt.semilogx(rad_mid, data.EM_median[index,:], 
                 color="gold", label="EM")
    plt.semilogx(rad_mid, data.SD_median[index,:], 
                 color="r", label="Stars")
    plt.semilogx(rad_mid, data.SZ_median[index,:], 
                 color="cyan", label="SZ")
    
    axes = plt.gca()
    ylim = axes.get_ylim()
    plt.semilogx([data.SP_DM_median[index], data.SP_DM_median[index]], 
                 ylim, color="k")
    plt.semilogx([data.SP_EM_median[index], data.SP_EM_median[index]], 
                 ylim, color="gold")
    plt.semilogx([data.SP_SD_median[index], data.SP_SD_median[index]], 
                 ylim, color="r")
    plt.semilogx([data.SP_SZ_median[index], data.SP_SZ_median[index]], 
                 ylim, color="cyan")
    
    plt.legend()
    plt.xlabel("$r/R_{200m}$")
    plt.ylabel("$d \log \\rho / d \log r$")
    plt.ylim(ylim)
    plt.show()


def compare_observable_profiles(data, index=86):
    """Compare different 3D density profiles, DM, gas and stars.
    Index set to 86 for a good example from MACSIS."""
    
    plt.figure()
    plt.semilogx(rad_mid, data.DM_density_3D[index,:], color="k", label="DM")
    plt.semilogx(rad_mid, data.gas_density_3D[index,:], color="gold", label="EM")
    plt.semilogx(rad_mid, data.star_density_3D[index,:], color="r", label="Stars")
    
    axes = plt.gca()
    ylim = axes.get_ylim()
    plt.semilogx([data.SP_DM_3D[index], data.SP_DM_3D[index]], ylim, color="k")
    plt.semilogx([data.SP_gas_3D[index], data.SP_gas_3D[index]], ylim, color="gold")
    plt.semilogx([data.SP_stars_3D[index], data.SP_stars_3D[index]], ylim, color="r")
    
    plt.legend()
    plt.xlabel("$r/R_{200m}$")
    plt.ylabel("$d \log \\rho / d \log r$")
    plt.ylim(ylim)
    plt.show()


def R_SP_histograms(data, subset="median"):
    """Creates histogram of splashback radius values. 
    subset - decides which property to look at; median, mean, 3D, z05"""
    
    if subset == "median":
        DM = data.SP_DM_median
        gas = data.SP_EM_median
        stars = data.SP_SD_median
        SZ = data.SP_SZ_median
        labels = ["DM", "EM", "Stellar", "SZ"]
    
    elif subset == "mean":
        DM = data.SP_DM_mean
        gas = data.SP_EM_mean
        stars = data.SP_SD_mean
        SZ = data.SP_SZ_mean
        labels = ["DM", "EM", "Stellar", "SZ"]
        
    elif subset == "3D":
        DM = data.SP_DM_3D
        gas = data.SP_gas_3D
        stars = data.SP_stars_3D
        labels = ["DM", "Gas", "Stellar"]
        
    elif subset == "z05":
        DM = data.SP_DM_z05
        gas = data.SP_gas_z05
        stars = data.SP_stars_z05
        labels = ["DM", "Gas", "Stellar"]
        
    else:
        print("Check names")
        return
    
    bins = np.linspace(0.4,1.8, 20)
    plt.figure()
    plt.hist(DM, bins=bins, 
             histtype="stepfilled",
             color="grey",
             alpha=0.6,
             label=labels[0])
    plt.hist(gas, bins=bins, 
             histtype="stepfilled",
             color="mediumblue",
             alpha=0.6,
             label=labels[1])
    plt.hist(stars, bins=bins,
             histtype="step",
             hatch="///",
             color="r",
             label=labels[2])
    if subset == "median" or subset == "mean":
        plt.hist(SZ, bins=bins,
                 histtype="step",
                 hatch="\\\\",
                 color="mediumseagreen",
                 label=labels[3])
    plt.show()
    
    DM_mean = np.nanmean(DM)
    DM_median = np.nanmedian(DM)
    DM_skew = skew(DM)
    
    gas_mean = np.nanmean(gas)
    gas_median = np.nanmedian(gas)
    gas_skew = skew(gas)
    
    stars_mean = np.nanmean(stars)
    stars_median = np.nanmedian(stars)
    stars_skew = skew(stars)
    
    print("Dark matter:" + str(DM_mean) + ", " + str(DM_median) + ", " + str(DM_skew))
    print("Gas:" + str(gas_mean) + ", " + str(gas_median) + ", " + str(gas_skew))
    print("Stellar density:" + str(stars_mean) + ", " + str(stars_median) + ", " + str(stars_skew))
    
    if subset == "median" or subset == "mean":
        SZ_mean = np.nanmean(SZ)
        SZ_median = np.nanmedian(SZ)
        SZ_skew = skew(SZ)
        
        print("SZ effect:" + str(SZ_mean) + ", " + str(SZ_median) + ", " + str(SZ_skew))
        
    
compare_RSP_2D_3D(mcs)


"""Stacked data profiles"""
# DM_stacked = splashback.stack_data(sp.DM_median)
# EM_stacked = splashback.stack_data(sp.EM_median)   
# SD_stacked = splashback.stack_data(sp.SD_median)  
# DM_stacked_3D = splashback.stack_data(sp.DM_density_3D)
    
# plt.figure()
# plt.semilogx(rad_mid, DM_stacked, color="k", label="Surface Density")
# plt.semilogx(rad_mid, EM_stacked, color="gold", label="EM")
# plt.semilogx(rad_mid, SD_stacked, color="r", label="Stars")
# plt.semilogx(rad_mid, DM_only_stack, color="k", linestyle="--", label="DM")
# plt.semilogx(rad_mid, DM_stacked_3D, color="k", linestyle=":", label="3D")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# #plt.savefig("stacked_obs_macsis_flattening.png", dpi=300)
# plt.show()


# gas_stacked_3D = splashback.stack_data(sp.gas_density_3D)
# stars_stacked_3D = splashback.stack_data(sp.star_density_3D)

# plt.figure()
# plt.semilogx(rad_mid, DM_stacked_3D, color="k", label="DM")
# plt.semilogx(rad_mid, gas_stacked_3D, color="gold", label="Gas")
# plt.semilogx(rad_mid, stars_stacked_3D, color="r", label="Stars")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# #plt.savefig("stacked_3D_macsis.png", dpi=300)
# plt.show()

"""Regular v perturbed clusters"""
# DM_stacked_rel = splashback.stack_data(sp.DM_median[rel,:])
# EM_stacked_rel = splashback.stack_data(sp.EM_median[rel,:])   
# SD_stacked_rel = splashback.stack_data(sp.SD_median[rel,:])  

# DM_stacked_per = splashback.stack_data(sp.DM_median[per,:])
# EM_stacked_per = splashback.stack_data(sp.EM_median[per,:])   
# SD_stacked_per = splashback.stack_data(sp.SD_median[per,:]) 

# plt.figure()
# plt.semilogx(rad_mid, DM_stacked_rel, color="k", label="Surface Density")
# plt.semilogx(rad_mid, EM_stacked_rel, color="gold", label="EM")
# plt.semilogx(rad_mid, SD_stacked_rel, color="r", label="Stars")

# plt.semilogx(rad_mid, DM_stacked_per, color="k", linestyle="--", label="Perturbed")
# plt.semilogx(rad_mid, EM_stacked_per, color="gold", linestyle="--")
# plt.semilogx(rad_mid, SD_stacked_per, color="r", linestyle="--")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# #plt.savefig("relaxed_perturbed_macsis_stacked_2D.png", dpi=300)
# plt.show()

# sp.read_MACSIS_z1()

# DM_stacked_z1 = splashback.stack_data(sp.DM_median_z1)
# EM_stacked_z1 = splashback.stack_data(sp.EM_median_z1)
# SD_stacked_z1 = splashback.stack_data(sp.SD_median_z1)

# plt.figure()
# plt.semilogx(rad_mid, DM_stacked, color="k", label="Surface density")
# plt.semilogx(rad_mid, DM_stacked_z1, color="k", label="z=1.0", linestyle="--")
# plt.semilogx(rad_mid, EM_stacked, color="gold", label="EM")
# plt.semilogx(rad_mid, EM_stacked_z1, color="gold", linestyle="--")
# plt.semilogx(rad_mid, SD_stacked, color="r", label="Stellar density")
# plt.semilogx(rad_mid, SD_stacked_z1, color="r", linestyle="--")
# plt.legend()
# plt.xlabel("$r/R_{200m}$")
# plt.ylabel("$d \log \\rho / d \log r$")
# plt.savefig("obs_profs_z0_z1.png", dpi=300)
# plt.show()



