import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L1000N1800"    

def bin_profiles(d, accretion_bins, mass_bins, energy_bins):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. Also calculates a best fit gradient, assuming that 
    the line goes through the origin.
    
    Inputs.
    d: obj, run of choice
    accretion_bins: array giving edge values of bins of accretion rate
    mass_bins: array giving edge values of bins of mass
    energy_bins: array giving edge values of bins of energy ratio.
    """

    sp.stack_and_find_3D(d, "accretion", accretion_bins)
    d.grad_accretion = np.sqrt(np.nanmean((d.R_gas_accretion/d.R_DM_accretion)**2))
    sp.stack_and_find_3D(d, "mass", mass_bins)
    d.grad_mass = np.sqrt(np.nanmean((d.R_gas_mass/d.R_DM_mass)**2))
    sp.stack_and_find_3D(d, "energy", energy_bins)
    d.grad_energy = np.sqrt(np.nanmean((d.R_gas_energy/d.R_DM_energy)**2))
    

def plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, bins, 
                               plot="_log_DM", stack_type="accretion"):
    """
    Plots dlogrho/dlogr profiles for different runs for same stacking criteria.

    Parameters
    ----------
    flm_HF : obj
        "Standard" run data.
    flm_HWA : obj
        First alternative run.
    flm_HSA : obj
        Second alternative run.
    bins : numpy array
        Bins used to stack profiles.
    plot : str, optional
        Plot DM or gas profiles. The default is "_log_DM". 
        Alternative is "_log_gas"
    stack_type : TYPE, optional
        Which stacking criteria to use for the profiles. 
        The default is "accretion".

    Returns
    -------
    None.

    """
    if stack_type=="accretion":
        symbol = "$\Gamma$"
    elif stack_type=="mass":
        symbol="$\log_{10} M_{\\rm{200m}}$"
    elif stack_type=="energy":
        symbol="$E_{\\rm{kin}} /E_{\\rm{therm}}$"
    
    ylim = (-4,0.2)
    N_bins = len(bins) - 1
    fig, ax = plt.subplots(nrows=1, ncols=3, 
                           figsize=(6,2.5), 
                           sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm_HF = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm_HWA = plt.cm.winter(np.linspace(0,1,N_bins))
    cm_HSA = plt.cm.copper(np.linspace(0,1,N_bins))
    lw = 0.8
    for i in range(0, N_bins):
        if i == 0:
            label = symbol + "$<$" + str(np.round(bins[1],2))
        elif i == N_bins-1:
            label = symbol + "$>$" + str(np.round(bins[i],2))
        else:
            label = str(np.round(bins[i],2)) \
            + "$<$" + symbol + "$<$" \
            + str(np.round(bins[i+1],2))
        ax[1].semilogx(rad_mid, getattr(flm_HF, stack_type+plot)[i,:], 
                       color=cm_HF[i], linewidth=lw,
                       label=label)
        ax[0].semilogx(rad_mid, getattr(flm_HWA, stack_type+plot)[i,:], 
                       color=cm_HWA[i], linewidth=lw,
                       label=label)
        ax[2].semilogx(rad_mid, getattr(flm_HSA, stack_type+plot)[i,:], 
                       color=cm_HSA[i], linewidth=lw,
                       label=label)
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[2].set_ylim(ylim)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[1].set_xlabel("$r/R_{\\rm{200m}}$")
    if plot == "_log_DM":
        ax[0].set_ylabel("$d \log \\rho_{\\rm{DM}} / d \log r$")
    elif plot == "_log_gas":
        ax[0].set_ylabel("$d \log \\rho_{\\rm{gas}} / d \log r$")
    ax[1].text(0.05, 0.05, "HYDRO_FIDUCIAL", transform=ax[1].transAxes)
    ax[0].text(0.05, 0.05, "HYDRO_WEAK_AGN", transform=ax[0].transAxes)
    ax[2].text(0.05, 0.05, "HYDRO_STRONG_AGN", transform=ax[2].transAxes)
    # filename = "splashback_data/flamingo/plots/compare_runs_energy_DM.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    

def plot_profiles_compare_all(flm_HF, flm_HWA, flm_HSA, bins,
                              quantity="DM"):
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
    
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$E_{\\rm{kin}} / E_{\\rm{therm}}$"])
    N_bins = bins.shape[1] -1
    if quantity == "DM":
        ylim = (-4.6,0.5)
        location = "upper left"
    else:
        ylim = (-3.9,0.95)
        location = "upper center"
    fig, ax = plt.subplots(nrows=3, ncols=3, 
                           figsize=(6,6), 
                           sharex=True, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm1 = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm2 = plt.cm.winter(np.linspace(0,1,N_bins))
    cm3 = plt.cm.copper(np.linspace(0,1,N_bins))
    lw = 0.8
    for i in range(0, N_bins):
        for j in range(3):
            if i == 0:
                label = labels[j] + "$<$"  + str(np.round(bins[j,1],2))
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
            else:
                cm = cm3
            ax[0,j].semilogx(rad_mid, getattr(flm_HWA, bin_type[j] + "_log_" + quantity)[i,:], 
                             color=cm[i], linewidth=lw, label=label)
            ax[1,j].semilogx(rad_mid, getattr(flm_HF, bin_type[j] + "_log_" + quantity)[i,:], 
                             color=cm[i], linewidth=lw)
            ax[2,j].semilogx(rad_mid, getattr(flm_HSA, bin_type[j] + "_log_" + quantity)[i,:], 
                             color=cm[i], linewidth=lw)
            ax[0,j].legend(loc=location)
    for axes in ax.flatten():
        axes.set_ylim(ylim)
    ax[0,0].text(0.05, 0.05, "HYDRO_WEAK_AGN", transform=ax[0,0].transAxes)
    ax[1,0].text(0.05, 0.05, "HYDRO_FIDUCIAL", transform=ax[1,0].transAxes)
    ax[2,0].text(0.05, 0.05, "HYDRO_STRONG_AGN", transform=ax[2,0].transAxes)
    ax[2,1].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[1,0].set_ylabel(r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity))
    filename = "splashback_data/flamingo/plots/HF_compare_all_" + quantity + ".png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
def plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
                               quantity="DM"):
    """
    Plots stacked profiles for one run to compare the effect of using different
    stacking criteria to define the bins used to stack.

    Parameters
    ----------
    flm : obj
        Run data to plot
    accretion_bins : array (N_bins-1)
        Edge values of accretion rate bins used for labelling.
    mass_bins : array (N_bins-1)
        Edge values of mass bins used for labelling.
    energy_bins : array (N_bins-1)
        Edge values of energy ratio bins used for labelling.
    quantity : str, optional
        Either "gas" or "DM". Decides which density profiles to plot.
        The default is "DM".

    Returns
    -------
    None.

    """
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$E_{\\rm{kin}} / E_{\\rm{therm}}$"])
    N_bins = len(accretion_bins) - 1
    ylim = (-5,0.5)
    fig, ax = plt.subplots(nrows=3, ncols=1, 
                           figsize=(3,6), 
                           sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm1 = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm2 = plt.cm.winter(np.linspace(0,1,N_bins))
    cm3 = plt.cm.copper(np.linspace(0,1,N_bins))
    lw = 0.8
    for i in range(N_bins):
        for j in range(3):
            if i == 0:
                label = labels[j] + "$<$"  + str(np.round(bins[j,i+1],2))
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
            else:
                cm = cm3
            ax[j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[2].set_ylim(ylim)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[1].set_ylabel(r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity))
    # filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    

def plot_Rsp_scatter_one(run, mids):
    """
    Plots the scatter in the splashback radius values when comparing values 
    from DM and gas density profiles. Only plots the data from one run.
    Different symbols are used for different bin types.

    Parameters
    ----------
    run : obj
        Run data to plot.
    mids : array (3,N_bins-1)
        Array giving mid points of bins used to stack for 3 stacking criteria.

    Returns
    -------
    None.

    """
    xlim = np.array([0.7,1.4])
    ylim = (0.91,1.2)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    cm = plt.cm.get_cmap('rainbow')
    axes.plot(xlim, xlim, color="k")
    axes.plot(xlim, run.grad_accretion*xlim, linestyle="--", color="grey")
    axes.plot(xlim, run.grad_mass*xlim, linestyle=":", color="grey")
    axes.plot(xlim, run.grad_energy*xlim, linestyle="-.", color="grey")
    crange1 = axes.scatter(run.R_DM_accretion, run.R_gas_accretion, 
                           c=mids[0,:], edgecolor="k", 
                           cmap=cm, s=75, marker="o",
                           label="Accretion rate")
    crange2 = axes.scatter(run.R_DM_mass, run.R_gas_mass, 
                          c=mids[1,:], edgecolor="k", 
                          cmap=cm, s=75, marker="*",
                          label="Mass")
    crange3 = axes.scatter(run.R_DM_energy, run.R_gas_energy, 
                           c=mids[2,:], edgecolor="k", 
                           cmap=cm, s=75, marker="v",
                           label="Energy ratio")
    # plt.xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    # plt.ylabel(r"$R_{\rm{SP,gas}} / R_{\rm{200m}}$")
    axes.legend(loc='upper left')
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.xlabel("$R_{\\rm{SP,DM}}$")
    plt.ylabel("$R_{\\rm{SP,gas}}$")
    cbaxes1 = fig.add_axes([0.185, 0.68, 0.02, 0.1]) 
    cbar = fig.colorbar(crange1, cax=cbaxes1, label="$\Gamma$")
    cbaxes2 = fig.add_axes([0.185, 0.53, 0.02, 0.1]) 
    cbar = fig.colorbar(crange2, cax=cbaxes2, label="$\log M_{\\rm{200m}}$")
    cbaxes3 = fig.add_axes([0.1855, 0.38, 0.02, 0.1]) 
    cbar = fig.colorbar(crange3, cax=cbaxes3, label="$E_{\\rm{kin}}/E_{\\rm{therm}}$")
    # filename = "compare_bins_HF.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    
def plot_Rsp_scatter(flm_HF, flm_HWA, flm_HSA, mids):
    xlim = np.array([0.7,1.4])
    ylim = (0.91,1.2)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.3,7),
                            gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.get_cmap('rainbow')
    axes[0].plot(xlim, xlim, color="k")
    axes[0].plot(xlim, flm_HF.grad_accretion*xlim, linestyle="--", color="grey")
    axes[0].plot(xlim, flm_HWA.grad_accretion*xlim, linestyle=":", color="grey")
    axes[0].plot(xlim, flm_HSA.grad_accretion*xlim, linestyle="-.", color="grey")
    axes[0].scatter(flm_HF.R_DM_accretion, flm_HF.R_gas_accretion, 
                    c=mids[0,:], edgecolor="k", 
                    cmap=cm, s=75, marker="o",
                    label="HYDRO_FIDUCIAL")
    crange = axes[0].scatter(flm_HWA.R_DM_accretion, flm_HWA.R_gas_accretion, 
                             c=mids[0,:], edgecolor="k", 
                             cmap=cm, s=75, marker="*",
                             label="HYDRO_WEAK_AGN")
    axes[0].scatter(flm_HSA.R_DM_accretion, flm_HSA.R_gas_accretion, 
                    c=mids[0,:], edgecolor="k", 
                    cmap=cm, s=75, marker="v",
                    label="HYDRO_STRONG_AGN")
    # plt.xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    # plt.ylabel(r"$R_{\rm{SP,gas}} / R_{\rm{200m}}$")
    axes[0].legend(loc='upper left')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_xticklabels([])
    cbaxes = fig.add_axes([0.185, 0.81, 0.02, 0.1]) 
    cbar = fig.colorbar(crange, cax=cbaxes, label="$\Gamma$")
    # filename = "compare_runs_accretion.png"
    # plt.savefig(filename, dpi=300)
    # plt.show()
    
    ylim = (0.91,1.3)
    
    # cm = plt.cm.get_cmap('rainbow')
    axes[1].plot(xlim, flm_HF.grad_mass*xlim, linestyle="--", color="grey", label="HYDRO_FIDUCIAL")
    axes[1].plot(xlim, flm_HWA.grad_mass*xlim, linestyle=":", color="grey", label="HYDRO_WEAK_AGN")
    axes[1].plot(xlim, flm_HSA.grad_mass*xlim, linestyle="-.", color="grey", label="HYDRO_STRONG_AGN")
    axes[1].plot(xlim, xlim, color="k")
    axes[1].scatter(flm_HF.R_DM_mass, flm_HF.R_gas_mass, 
                c=mids[1,:], edgecolor="k", 
                cmap=cm, s=75, marker="o") #,
                # label="HYDRO_FIDUCIAL")
    axes[1].scatter(flm_HWA.R_DM_mass, flm_HWA.R_gas_mass, 
                c=mids[1,:], edgecolor="k", 
                cmap=cm, s=75, marker="*") #,
                # label="HYDRO_WEAK_AGN")
    crange = axes[1].scatter(flm_HSA.R_DM_mass, flm_HSA.R_gas_mass, 
                c=mids[1,:], edgecolor="k", 
                cmap=cm, s=75, marker="v") #,
                # label="HYDRO_STRONG_AGN")
    # plt.xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    axes[1].set_ylabel(r"$R_{\rm{SP,gas}} / R_{\rm{200m}}$")
    axes[1].legend(loc='upper left')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_xticklabels([])
    cbaxes = fig.add_axes([0.185, 0.505, 0.02, 0.1]) 
    cbar = plt.colorbar(crange, cax=cbaxes, label="$\log_{10}M_{\\rm{200m}}$")
    # filename = "compare_runs_mass.png"
    # plt.savefig(filename, dpi=300)
    # plt.show()
    
    # fig = plt.figure()
    # cm = plt.cm.get_cmap('rainbow')
    axes[2].plot(xlim, flm_HF.grad_energy*xlim, linestyle="--", color="grey")
    axes[2].plot(xlim, flm_HWA.grad_energy*xlim, linestyle=":", color="grey")
    axes[2].plot(xlim, flm_HSA.grad_energy*xlim, linestyle="-.", color="grey")
    axes[2].plot(xlim, xlim, color="k")
    axes[2].scatter(flm_HF.R_DM_energy, flm_HF.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="o",
                label="HYDRO_FIDUCIAL")
    axes[2].scatter(flm_HWA.R_DM_energy, flm_HWA.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="*",
                label="HYDRO_WEAK_AGN")
    crange = axes[2].scatter(flm_HSA.R_DM_energy, flm_HSA.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="v",
                label="HYDRO_STRONG_AGN")
    axes[2].set_xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    # plt.ylabel(r"$R_{\rm{SP,gas}} / R_{\rm{200m}}$")
    # ax[2].legend()
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    cbaxes = fig.add_axes([0.185, 0.25, 0.02, 0.1]) 
    cbar = plt.colorbar(crange, cax=cbaxes, label="$E_{\\rm{kin}} / E_{\\rm{therm}}$")
    # filename = "splashback_data/flamingo/plots/compare_runs_new.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    # End of big plot


def plot_Rsp_params(flm_HF, flm_HWA, flm_HSA, plot_name, mids):
    fig, ax = plt.subplots(nrows=1, ncols=3, 
                            sharey=True, figsize=(5,2))
    ax[0].scatter(mids[0,:], getattr(flm_HF, plot_name+"_accretion"), 
                  marker="o", edgecolor="k", label="HYDRO_FIDUCIAL",
                  color="g")
    ax[0].scatter(mids[0,:], getattr(flm_HWA, plot_name+"_accretion"), 
                  marker="*", edgecolor="k", label="HYDRO_WEAK_AGN",
                  color="gold")
    ax[0].scatter(mids[0,:], getattr(flm_HSA, plot_name+"_accretion"), 
                  marker="v", edgecolor="k", label="HYDRO_STRONG_AGN",
                  color="r")
    # plt.legend()
    ax[0].set_xlabel(r"$\Gamma$")
    ax[0].set_ylabel(r"$\gamma(R_{\rm{SP, gas}})$")
    # plt.show()
    # plt.figure()
    ax[1].scatter(mids[1,:], getattr(flm_HF, plot_name+"_mass"), 
                  marker="o", edgecolor="k", label="HYDRO_FIDUCIAL",
                  color="g")
    ax[1].scatter(mids[1,:], getattr(flm_HWA, plot_name+"_mass"), 
                  marker="*", edgecolor="k", label="HYDRO_WEAK_AGN",
                  color="gold")
    ax[1].scatter(mids[1,:], getattr(flm_HSA, plot_name+"_mass"), 
                  marker="v", edgecolor="k", label="HYDRO_STRONG_AGN",
                  color="r")
    
    ax[0].legend()
    ax[1].set_xlabel(r"$\log_{10} (M_{\rm{200m}} / M_{\odot})$")
    # plt.ylabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    # plt.show()
    
    # plt.figure()
    ax[2].scatter(mids[2,:], getattr(flm_HF, plot_name+"_energy"), 
                  marker="o", edgecolor="k", label="HYDRO_FIDUCIAL",
                  color="g")
    ax[2].scatter(mids[2,:], getattr(flm_HWA, plot_name+"_energy"), 
                  marker="*", edgecolor="k", label="HYDRO_WEAK_AGN",
                  color="gold")
    ax[2].scatter(mids[2,:], getattr(flm_HSA, plot_name+"_energy"), 
                  marker="v", edgecolor="k", label="HYDRO_STRONG_AGN",
                  color="r")
    # plt.legend()
    ax[2].set_xlabel(r"$E_{\rm{kin}} / E_{\rm{therm}}$")
    # plt.ylabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    # filename = "splashback_data/flamingo/plots/compare_depths_gas.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    
def plot_Rsp_gamma(flm_HF, flm_HWA, flm_HSA, plot_name, bins):
    fig, ax = plt.subplots(nrows=2, ncols=1, 
                           sharex=True, figsize=(3.3,4))
    cm = plt.cm.get_cmap('rainbow')
    
    ax[0].scatter(getattr(flm_HF, "R_DM_"+plot_name), 
                  getattr(flm_HF, "depth_DM_"+plot_name),
                  marker="o", edgecolor="k", c=bins,
                  cmap=cm, label="HYDRO_FIDUCIAL")
    ax[0].scatter(getattr(flm_HWA, "R_DM_"+plot_name), 
                  getattr(flm_HWA, "depth_DM_"+plot_name),
                  marker="*", edgecolor="k", c=bins,
                  cmap=cm, label="HYDRO_WEAK_AGN")
    ax[0].scatter(getattr(flm_HSA, "R_DM_"+plot_name), 
                  getattr(flm_HSA, "depth_DM_"+plot_name),
                  marker="v", edgecolor="k", c=bins,
                  cmap=cm, label="HYDRO_STRONG_AGN")
    
    
    ax[1].scatter(getattr(flm_HF, "R_gas_"+plot_name), 
                  getattr(flm_HF, "depth_gas_"+plot_name),
                  marker="o", edgecolor="k", c=bins,
                  cmap=cm, label="HYDRO_FIDUCIAL")
    ax[1].scatter(getattr(flm_HWA, "R_gas_"+plot_name), 
                  getattr(flm_HWA, "depth_gas_"+plot_name),
                  marker="*", edgecolor="k", c=bins,
                  cmap=cm, label="HYDRO_WEAK_AGN")
    crange = ax[1].scatter(getattr(flm_HSA, "R_gas_"+plot_name), 
                           getattr(flm_HSA, "depth_gas_"+plot_name),
                           marker="v", edgecolor="k", c=bins,
                           cmap=cm, label="HYDRO_STRONG_AGN")
    
    ax[0].legend()
    cbaxes = fig.add_axes([0.195, 0.4, 0.02, 0.1]) 
    cbar = plt.colorbar(crange, cax=cbaxes, label="$E_{\\rm{kin}} / E_{\\rm{therm}}$")
    # ax[0].set_xlabel("$R_{\\rm{SP,DM}}$")
    ax[1].set_xlabel("$R_{\\rm{SP}}$")
    ax[0].set_ylabel("$\gamma(R_{\\rm{SP,DM}})$")
    ax[1].set_ylabel("$\gamma(R_{\\rm{SP,gas}})$")
    filename = "splashback_data/flamingo/plots/compare_Rsp_gamma_energy.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    
def stack_for_profiles():
    N_bins = 4
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    bin_profiles(flm_HF, accretion_bins, mass_bins, energy_bins)
    bin_profiles(flm_HWA, accretion_bins, mass_bins, energy_bins)
    bin_profiles(flm_HSA, accretion_bins, mass_bins, energy_bins)
    
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    
    plot_profiles_compare_all(flm_HF, flm_HWA, flm_HSA, bins,
                              quantity="DM")
    plot_profiles_compare_all(flm_HF, flm_HWA, flm_HSA, bins,
                              quantity="gas")
    
    # plot_profiles_compare_bins(flm_HF, accretion_bins, mass_bins, energy_bins,
    #                             quantity="DM")

    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, accretion_bins,
    #                             stack_type="accretion", plot="_log_DM")
    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, accretion_bins,
    #                             stack_type="accretion", plot="_log_gas")
    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, mass_bins,
    #                             stack_type="mass", plot="_log_DM")
    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, mass_bins,
    #                             stack_type="mass", plot="_log_gas")
    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, energy_bins,
    #                             stack_type="energy", plot="_log_DM")
    # plot_profiles_compare_runs(flm_HF, flm_HWA, flm_HSA, energy_bins,
    #                             stack_type="energy", plot="_log_gas")
    
    
def stack_for_Rsp():
    N_bins = 10
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    bin_profiles(flm_HF, accretion_bins, mass_bins, energy_bins)
    bin_profiles(flm_HWA, accretion_bins, mass_bins, energy_bins)
    bin_profiles(flm_HSA, accretion_bins, mass_bins, energy_bins)
    
    mass_mid = np.zeros(N_bins+1)
    accretion_mid = np.zeros(N_bins+1)
    energy_mid = np.zeros(N_bins+1)

    mass_mid[:-1] = (mass_bins[:-2] + mass_bins[1:-1])/2
    accretion_mid[:-1] = (accretion_bins[:-2] + accretion_bins[1:-1])/2
    energy_mid[:-1] = (energy_bins[:-2] + energy_bins[1:-1])/2
    mass_mid[-1] = mass_bins[-2]
    accretion_mid[-1] = accretion_bins[-2]
    energy_mid[-1] = energy_bins[-2]

    mids = np.vstack((accretion_mid, mass_mid, energy_mid))
    
    # Make big plot comparing Rsp for different runs and stacking criteria.
    plot_Rsp_scatter(flm_HF, flm_HWA, flm_HSA, mids)
    
    # Compare splashback with different quantities
    # plot_Rsp_params(flm_HF, flm_HWA, flm_HSA, "depth_gas", mids)
    # plot_Rsp_gamma(flm_HF, flm_HWA, flm_HSA, "energy", energy_mid)
    
    
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

if __name__ == "__main__":
    flm_HF = sp.flamingo(box, "HF")
    flm_HF.read_properties()

    flm_HWA = sp.flamingo(box, "HWA")
    flm_HWA.read_properties()
    
    flm_HSA = sp.flamingo(box, "HSA")
    flm_HSA.read_properties()
    
    # stack_for_profiles()
    stack_for_Rsp()
    
    # index = 2
    # test_density = flm_HF.DM_density_3D[index,:]
    # test_gradient = np.gradient(np.log10(test_density), np.log10(rad_mid))
    
    # plt.figure()#figsize=(1.5,1.5))
    # plt.semilogx(rad_mid, test_gradient, color="k")
    # xlim = plt.gca().get_xlim()
    # ylim = plt.gca().get_ylim()
    # # plt.semilogx((flm_HF.R_DM_accretion[2], flm_HF.R_DM_accretion[2]), ylim,
    # #               color="blueviolet", linestyle="--")
    # plt.fill_between(xlim, (ylim[0],ylim[0]), (-3,-3), facecolor="grey", alpha=0.6)
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    # plt.ylabel("$d \log \\rho_{\\rm{DM}} / d \log r$")
    # plt.savefig("splashback_data/flamingo/plots/example_noise.png",
    #             dpi=300)
    # plt.show()
    
    
    