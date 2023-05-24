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
    
    
def plot_Rsp_scatter(run1, run2, run3, mids):
    xlim = np.array([0.7,1.4])
    ylim = (0.91,1.2)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.3,7),
                            gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.get_cmap('rainbow')
    axes[0].plot(xlim, xlim, color="k")
    axes[0].plot(xlim, run1.grad_accretion*xlim, linestyle="--", color="grey")
    axes[0].plot(xlim, run2.grad_accretion*xlim, linestyle=":", color="grey")
    axes[0].plot(xlim, run3.grad_accretion*xlim, linestyle="-.", color="grey")
    axes[0].scatter(run1.R_DM_accretion, run1.R_gas_accretion, 
                    c=mids[0,:], edgecolor="k", 
                    cmap=cm, s=75, marker="o",
                    label=run1.run_label)
    crange = axes[0].scatter(run2.R_DM_accretion, run2.R_gas_accretion, 
                             c=mids[0,:], edgecolor="k", 
                             cmap=cm, s=75, marker="*",
                             label=run2.run_label)
    axes[0].scatter(run3.R_DM_accretion, run3.R_gas_accretion, 
                    c=mids[0,:], edgecolor="k", 
                    cmap=cm, s=75, marker="v",
                    label=run3.run_label)
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
    axes[1].plot(xlim, run1.grad_mass*xlim, linestyle="--", color="grey", label=run1.run_label)
    axes[1].plot(xlim, run2.grad_mass*xlim, linestyle=":", color="grey", label=run2.run_label)
    axes[1].plot(xlim, run3.grad_mass*xlim, linestyle="-.", color="grey", label=run3.run_label)
    axes[1].plot(xlim, xlim, color="k")
    axes[1].scatter(run1.R_DM_mass, run1.R_gas_mass, 
                c=mids[1,:], edgecolor="k", 
                cmap=cm, s=75, marker="o") #,
                # label="HYDRO_FIDUCIAL")
    axes[1].scatter(run2.R_DM_mass, run2.R_gas_mass, 
                c=mids[1,:], edgecolor="k", 
                cmap=cm, s=75, marker="*") #,
                # label="HYDRO_WEAK_AGN")
    crange = axes[1].scatter(run3.R_DM_mass, run3.R_gas_mass, 
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
    axes[2].plot(xlim, run1.grad_energy*xlim, linestyle="--", color="grey")
    axes[2].plot(xlim, run2.grad_energy*xlim, linestyle=":", color="grey")
    axes[2].plot(xlim, run3.grad_energy*xlim, linestyle="-.", color="grey")
    axes[2].plot(xlim, xlim, color="k")
    axes[2].scatter(run1.R_DM_energy, run1.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="o",
                label=run1.run_label)
    axes[2].scatter(run2.R_DM_energy, run2.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="*",
                label=run2.run_label)
    crange = axes[2].scatter(run3.R_DM_energy, run3.R_gas_energy, 
                c=mids[2,:], edgecolor="k", 
                cmap=cm, s=75, marker="v",
                label=run3.run_label)
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
    
    
def plot_params_all(bins, y, labels):
    """
    Plots chosen quantity against bin mid points.
    Could plot Rsp or gamma, should work for all.

    Parameters
    ----------
    bins : numpy array (3, N_bins)
        Bin midpoints.
    y : numpy array (N_runs, N_bins)
        Quantity to plot on y-axis.
    labels : list of strings
        List of labels.
    """

    N_runs = y.shape[0]
    
    fig, ax = plt.subplots(nrows=1, ncols=3,
                           sharey=True,
                           figsize=(6,3))
    cm = plt.cm.rainbow(np.linspace(0,1,N_runs))
    for i in range(N_runs):
        ax[0].scatter(bins[0,:], y[i,:,0], edgecolor="k", c=cm[i], label=labels[i])
        ax[1].scatter(bins[1,:], y[i,:,1], edgecolor="k", c=cm[i])
        ax[2].scatter(bins[2,:], y[i,:,2], edgecolor="k", c=cm[i])
    ax[0].legend()
    ax[0].set_ylabel("$R_{\\rm{SP}}$")
    ax[0].set_xlabel("$\Gamma$")
    ax[1].set_xlabel("$M_{\\rm{200m}}$")
    ax[2].set_xlabel("$E_{\\rm{kin}} / E_{\\rm{therm}}$")
    plt.show()
    
    
def plot_all_profiles(list_of_sims, bins, quantity="DM"):
    """
    Compares all profiles for three different runs and three standard stacking
    criteria.

    Parameters
    ----------
    list_of_sims: list
        List of objects, one for each simulation run wanted to plot. Order of
        runs will be order of plotting.
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
    N_runs = len(list_of_sims)
    if quantity == "DM":
        ylim = (-4.6,0.8)
        location = "upper left"
    else:
        ylim = (-3.9,2.1)
        location = "upper center"
    fig, ax = plt.subplots(nrows=N_runs, ncols=3, 
                           figsize=(5.5,N_runs*1.8),
                           sharex=True, sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm1 = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm2 = plt.cm.winter(np.linspace(0,1,N_bins))
    cm3 = plt.cm.copper(np.linspace(0,1,N_bins))
    lw = 0.8
    for j in range(3):
        for i in range(0, N_bins):
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
            for k in range(N_runs):
                ax[k,j].semilogx(rad_mid, getattr(list_of_sims[k], bin_type[j] + "_log_" + quantity)[i,:], 
                                 color=cm[i], linewidth=lw, label=label)
                ax[k,0].text(0.05, 0.05, list_of_sims[k].run_label, transform=ax[k,0].transAxes)
        ax[0,j].legend(loc=location)
    for axes in ax.flatten():
        axes.set_ylim(ylim)
    
    ax[-1,1].set_xlabel("$r/R_{\\rm{200m}}$")
    #ax[int(N_runs/2),0].set_ylabel(r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity))
    fig.text(0.02, 0.45, r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity),
             transform=fig.transFigure, rotation='vertical')
    plt.subplots_adjust(left=0.08)
    filename = "splashback_data/flamingo/plots/HF_compare_all_" + quantity + ".png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
def stack_for_profiles(list_of_sims):
    N_bins = 4
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    N_runs = len(list_of_sims)
    for i in range(N_runs):
        bin_profiles(list_of_sims[i], accretion_bins, mass_bins, energy_bins)
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    plot_all_profiles(list_of_sims, bins, quantity="DM")
    plot_all_profiles(list_of_sims, bins, quantity="gas")
    
    # plot_profiles_compare_bins(run1, accretion_bins, mass_bins, energy_bins,
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
    
    
def stack_for_Rsp(run1, run2, run3):
    N_bins = 10
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    bin_profiles(run1, accretion_bins, mass_bins, energy_bins)
    bin_profiles(run2, accretion_bins, mass_bins, energy_bins)
    bin_profiles(run3, accretion_bins, mass_bins, energy_bins)
    
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
    plot_Rsp_scatter(run1, run2, run3, mids)
    
    # Rsp_acc = np.vstack((run1.R_DM_accretion, run2.R_DM_accretion, run3.R_DM_accretion))
    # Rsp_mass = np.vstack((run1.R_DM_mass, run2.R_DM_mass, run3.R_DM_mass))
    # Rsp_energy = np.vstack((run1.R_DM_energy, run2.R_DM_energy, run3.R_DM_energy))
    # Rsp = np.dstack((Rsp_acc, Rsp_mass, Rsp_energy))
    # labels = [run1.run_label, run2.run_label, run2.run_label]
    # plot_params_all(mids, Rsp, labels)
    
    # Compare splashback with different quantities
    # plot_Rsp_params(flm_HF, flm_HWA, flm_HSA, "depth_gas", mids)
    # plot_Rsp_gamma(flm_HF, flm_HWA, flm_HSA, "energy", energy_mid)
    
    
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

if __name__ == "__main__":
    hf = sp.flamingo(box, "HF")
    hf.read_properties()

    hwa = sp.flamingo(box, "HWA")
    hwa.read_properties()
    
    hsa = sp.flamingo(box, "HSA")
    hsa.read_properties()
    
    hta = sp.flamingo(box, "HTA")
    hta.read_properties()
    
    hj = sp.flamingo(box, "HJ")
    hj.read_properties()
    
    hsj = sp.flamingo(box, "HSJ")
    hsj.read_properties()
    
    hp = sp.flamingo(box, "HP")
    hp.read_properties()
    
    hpf = sp.flamingo(box, "HPF")
    hpf.read_properties()
    
    hpv = sp.flamingo(box, "HPV")
    hpv.read_properties()
    
    stack_for_profiles([hf, hwa, hsa, hta, hj, hsj])
    # stack_for_profiles(hf, hj, hsj)
    # stack_for_profiles(hp, hpf, hpv)
    
    # stack_for_Rsp(hwa, hf, hsa)
    # stack_for_Rsp(hf, hj, hsj)
    # stack_for_Rsp(hp, hpf, hpv)
    
    
    