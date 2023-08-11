import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L1000N1800"

def dmo_stacking(data, split, split_bins, bootstrap="none"):
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)

    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    stacked_data = np.zeros((N_bins, data.N_rad))
    print("")
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        print(len(bin_mask))
        stacked_data[i,:] = sp.stack_data(data.DM_density_3D[not_nan,:][bin_mask,:])
            
    log = sp.log_gradients(data.rad_mid, stacked_data)
    R_SP_DM, second, depth_DM, second_depth = dr.depth_cut(data.rad_mid, log, 
                                                           cut=-2.5,
                                                           depth_value="y",
                                                           second_caustic="y")
    setattr(data, split+ "_density_DM", stacked_data)
    setattr(data, split+ "_log_DM", log)
    setattr(data, "R_DM_"+split, R_SP_DM)
    setattr(data, "2_DM_"+split, second)
    setattr(data, "depth_DM_"+split, depth_DM)
    
    if bootstrap != "none":
        Rsp_error = sp.bootstrap_errors(data, data.DM_density_3D, split, split_data, split_bins)
        setattr(data, "error_R_" + split + "_DM", Rsp_error[0,:])
    

def second_caustic(data, split):
    second_all = getattr(data, "2_DM_"+split)
    R_sp = getattr(data, "R_DM_"+split)
    
    second_mask = np.where(np.isfinite(second_all))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if R_sp[index] < second_all[index]:
            larger = second_all[index]
            smaller = R_sp[index]
            R_sp[index] = larger
            second_all[index] = smaller
    setattr(data, "R_DM_"+split, R_sp)
    setattr(data, "2_DM_"+split, second_all)
    
    
def plot_profiles(flm, dmo, bins):
    """
    Compares all profiles for three different runs and three standard stacking
    criteria.

    Parameters
    ----------
    flm_HF : obj
        "Standard" run data.
    flm_HWA : obj
        First alternative run.
    flm_HSA : obj
        Second alternative run.
    bins : numpy array
        Bins used to stack profiles for all three stacking criteria. 
        (3,N_bins) In order: accretion rate, mass, energy ratio.
    quantity : str, optional
        Plot either gas or DM profiles. The default is "DM".

    Returns
    -------
    None.

    """
    bin_type = np.array(["accretion", "mass"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$"])
    N_bins = bins.shape[1] -1
    
    fig, ax = plt.subplots(nrows=2, ncols=2,
                           figsize=(4,4),
                           sharex=True, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm1 = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm2 = plt.cm.winter(np.linspace(0,1,N_bins))
    lw = 0.8
    location = 'upper left'
    for i in range(0, N_bins):
        for j in range(2):
            if i == 0:
                label =  str(np.round(bins[j,0],2)) + "$<$" + labels[j] \
                    + "$<$"  + str(np.round(bins[j,1],2))
            elif i == N_bins-1:
                label = labels[j] + "$>$" + str(np.round(bins[j,i],2))
            else:
                label = str(np.round(bins[j,i],2)) \
                    + "$<$" + labels[j] + "$<$" \
                    + str(np.round(bins[j,i+1],2))
            if j == 0:
                cm = cm1
            elif j==1:
                cm = cm2
            ax[0,j].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_DM")[i,:], 
                             color=cm[i], linewidth=lw, label=label)
            ax[1,j].semilogx(dmo.rad_mid, getattr(dmo, bin_type[j] + "_log_DM")[i,:], 
                             color=cm[i], linewidth=lw)
            ax[0,j].legend(loc=location)
    ylim = (-4.2, 0.9)
    ax[0,0].set_ylim(ylim)
    ax[0,0].text(0.05, 0.05, "HYDRO", transform=ax[0,0].transAxes)
    ax[1,0].text(0.05, 0.05, "DMO", transform=ax[1,0].transAxes)
    #plt.subplots_adjust(left=0.5)
    #plt.subplots_adjust(bottom=0.1)
    plt.text(0.5, 0.04, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    plt.text(0.0, 0.45, r"$d \log \rho_{\rm{DM}} / d \log r$", 
             transform=fig.transFigure, rotation="vertical")
    filename = "splashback_data/flamingo/plots/dmo_v_hydro_profiles.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
def compare_rsp(flm, dmo, bins):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(flm.R_DM_accretion, dmo.R_DM_accretion,
               edgecolor="k", marker="o", color="red", s=75,
               label="$\Gamma$")
    ax.scatter(flm.R_DM_mass, dmo.R_DM_mass,
               edgecolor="k", marker="^", color="blue", s=75,
               label="$M_{\\rm{200m}}$")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.xlabel(r"$R_{\rm{SP,hydro}}/R_{\rm{200m}}$")
    plt.ylabel(r"$R_{\rm{SP,DMO}}/R_{\rm{200m}}$")
    plt.legend()
    # plt.savefig("splashback_data/flamingo/plots/dmo_v_hydro_rsp.png", dpi=300)
    plt.show()
    
def plot_difference(flm, dmo):
    delta_accretion = (flm.R_DM_accretion - dmo.R_DM_accretion) 
    acc_error = np.sqrt(flm.error_R_accretion_DM**2 + dmo.error_R_accretion_DM**2)
    acc_error = np.sqrt((acc_error/delta_accretion)**2 + (flm.error_R_accretion_DM/ flm.R_DM_accretion)**2)
    delta_accretion = delta_accretion / flm.R_DM_accretion
    acc_error = acc_error * delta_accretion
    print(acc_error)
    
    delta_mass = (flm.R_DM_mass - dmo.R_DM_mass) 
    mass_error = np.sqrt(flm.error_R_mass_DM**2 + dmo.error_R_mass_DM**2)
    mass_error = np.sqrt((mass_error/delta_mass)**2 + (flm.error_R_mass_DM/ flm.R_DM_mass)**2)
    delta_mass = delta_mass / flm.R_DM_mass
    mass_error = mass_error * delta_mass
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(4,3))
    ax[0].errorbar(mass_mid, delta_mass, yerr=mass_error,
                   fmt="o",
                   color="cyan",
                   capsize=3)
    ax[0].set_xscale('log')
    ax[0].set_xlabel("$M_{\\rm{200m}}$")
    ax[0].set_ylabel("$(R_{\\rm{SP,hydro}} - R_{\\rm{SP, DMO}}) / R_{\\rm{SP,hydro}}$")
    
    ax[1].errorbar(accretion_mid, delta_accretion, yerr=acc_error,
                   fmt="o",
                   color="r",
                   capsize=3)
    ax[1].set_xlabel("$\Gamma$")
    # filename = "splashback_data/flamingo/plots/dmo_differences.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)
    # ax2 = ax1.twiny() #shares y axis
    # ax1.scatter(mass_mid, delta_mass, 
    #             color="cyan")
    # ax1.set_xscale('log')
    # ax1.set_xlabel("$M_{\\rm{200m}}$")
    # ax1.set_ylabel("$(R_{\\rm{SP,hydro}} - R_{\\rm{SP, DMO}}) / R_{\\rm{SP,hydro}}$")
    
    # ax2.scatter(accretion_mid, delta_accretion, 
    #             color="r")
    # ax2.set_xlabel("$\Gamma$")
    # plt.show()

dmo = sp.flamingo(box, "DMO")
dmo.read_properties()
# dmo.read_low_mass()
flm = sp.flamingo(box, "HF")
flm.read_properties()
# flm.read_low_mass()

N_bins = 10
mass_bins = np.linspace(14, 15.4, N_bins+1)
accretion_bins = np.linspace(0, 4.2, N_bins+1)
mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
bins = np.vstack((accretion_bins, mass_bins))

sp.stack_and_find_3D(flm, "accretion", accretion_bins, bootstrap="y")
sp.stack_and_find_3D(flm, "mass", mass_bins, bootstrap="y")
dmo_stacking(dmo, "accretion", accretion_bins, bootstrap="y")
dmo_stacking(dmo, "mass", mass_bins, bootstrap="y")
second_caustic(dmo, "accretion")
second_caustic(flm, "accretion")

# plot_profiles(flm, dmo, bins)
#compare_rsp(flm, dmo, bins)
plot_difference(flm, dmo)