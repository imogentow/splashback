import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L2800N5040"

def dmo_stacking(data, split, split_bins, bootstrap=False):
    """
    Stacks dark matter only profiles.

    Parameters
    ----------
    data : obj
        Simulation data class.
    split : string
        Name of criteria to use for stacking.
    split_bins : array
        Bin edges for stacking criteria.
    bootstrap : bool, optional
        Whether to calculate sampling errors on the location of the minima.
        The default is False.

    Returns
    -------
    None.

    """
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)

    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    stacked_data = np.zeros((N_bins, data.N_rad))
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
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
    
    if bootstrap:
        Rsp_error, _ = sp.bootstrap_errors(data, data.DM_density_3D, split, split_data, split_bins)
        setattr(data, "error_R_DM_"+split, Rsp_error[0,:])
    

def second_caustic(data, split):
    """
    Identifies which minima is the second caustic and splashback radius in
    dark matter profiles.

    Parameters
    ----------
    data : obj
        Simulation data.
    split : str
        Name of criteria used for profile stacking.

    Returns
    -------
    None.

    """
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
    
    
def compare_rsp(flm, dmo):
    """
    Compares locations of minima in dark matter only and hydro runs.

    Parameters
    ----------
    flm : obj
        Hydro simulation data.
    dmo : obj
        Dark matter only simulation data.

    Returns
    -------
    None.

    """
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
    """
    Plots differences between the minima in hydro and dark matter only 
    simulations.

    Parameters
    ----------
    flm : obj
        Hydro simulation data.
    dmo : TYPE
        Dark matter only simulation data.

    Returns
    -------
    None.

    """
    delta_accretion = (flm.R_DM_accretion - dmo.R_DM_accretion)/ flm.R_DM_accretion
    error1 = dmo.R_DM_accretion / flm.R_DM_accretion
    error2 = dmo.error_R_DM_accretion / dmo.R_DM_accretion
    error3 = flm.error_R_DM_accretion / flm.R_DM_accretion
    acc_error = error1 * np.sqrt(error2**2 + error3**2)
    weights_acc = 1 / acc_error**2
    mean_acc = np.nansum(delta_accretion * weights_acc) / np.nansum(weights_acc)
    
    delta_mass = (flm.R_DM_mass - dmo.R_DM_mass) / flm.R_DM_mass
    error1 = dmo.R_DM_mass / flm.R_DM_mass
    error2 = dmo.error_R_DM_mass / dmo.R_DM_mass
    error3 = flm.error_R_DM_mass / flm.R_DM_mass
    mass_error = error1 * np.sqrt(error2**2 + error3**2)
    weights_mass = 1 / mass_error**2
    mean_mass = np.nansum(delta_mass * weights_mass) / np.nansum(weights_mass)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(3.3,2))
    ax[0].errorbar(mass_mid, delta_mass, yerr=mass_error,
                   fmt="o",
                   color="cyan",
                   capsize=2,
                   markersize=2)
    ax[0].set_xscale('log')
    xlim = ax[0].get_xlim()
    ax[0].semilogx(xlim, (mean_mass, mean_mass),
                    linestyle="--", color="grey",
                    label="Weighted mean")
    ax[0].semilogx(xlim, (0,0),
                    color="k", label="Zero")
    ax[0].set_xlim(xlim)
    ax[0].set_xlabel("$M_{\\rm{200m}} / \\rm{M_{\odot}}$")
    ax[0].set_ylabel("$(R_{\\rm{SP,hydro}} - R_{\\rm{SP, DMO}}) / R_{\\rm{SP,hydro}}$")
    ax[0].legend()
    
    ax[1].errorbar(accretion_mid, delta_accretion, yerr=acc_error,
                   fmt="o",
                   color="r",
                   capsize=2,
                   markersize=2)
    xlim = ax[1].get_xlim()
    ax[1].plot(xlim, (mean_acc, mean_acc), 
                linestyle="--", color="grey")
    ax[1].plot(xlim, (0,0),
                    color="k")
    ax[1].set_xlim(xlim)
    ax[1].set_xlabel("$\Gamma$")
    plt.subplots_adjust(left=0.15, right=0.99, bottom=0.17,top=0.99)
    filename = "splashback_data/flamingo/plots/dmo_differences.png"
    plt.savefig(filename, dpi=300)
    plt.show()

dmo = sp.flamingo(box, "DMO")
dmo.read_properties()
flm = sp.flamingo(box, "HF")
flm.read_properties()

N_bins = 10
mass_bins = np.linspace(14, 15.4, N_bins+1)
accretion_bins = np.linspace(0, 4.2, N_bins+1)
mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
bins = np.vstack((accretion_bins, mass_bins))

sp.stack_and_find_3D(flm, "accretion", accretion_bins, bootstrap=True)
sp.stack_and_find_3D(flm, "mass", mass_bins, bootstrap=True)
dmo_stacking(dmo, "accretion", accretion_bins, bootstrap=True)
dmo_stacking(dmo, "mass", mass_bins, bootstrap=True)
second_caustic(dmo, "accretion")
second_caustic(flm, "accretion")

# plot_profiles(flm, dmo, bins)
#compare_rsp(flm, dmo, bins)
plot_difference(flm, dmo)