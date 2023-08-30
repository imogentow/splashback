import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L1000N1800"    
axes_labels = {
    "mass": "$M_{\\rm{200m}}$",
    "accretion": "$\Gamma$",
    "energy": "$E_{\\rm{kin}}/E_{\\rm{therm}}$"}

def bin_profiles(d, accretion_bins, mass_bins, energy_bins, bootstrap="none"):
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

    sp.stack_and_find_3D(d, "accretion", accretion_bins, bootstrap=bootstrap)
    sp.stack_and_find_3D(d, "mass", mass_bins, bootstrap=bootstrap)
    sp.stack_and_find_3D(d, "energy", energy_bins, bootstrap=bootstrap)

    
def second_caustic(flm, split):
    second_all = getattr(flm, "2_DM_"+split)
    R_sp = getattr(flm, "R_DM_"+split)
    
    second_mask = np.where(np.isfinite(second_all))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if R_sp[index] < second_all[index]:
            larger = second_all[index]
            smaller = R_sp[index]
            R_sp[index] = larger
            second_all[index] = smaller
    setattr(flm, "R_DM_"+split, R_sp)
    setattr(flm, "2_DM_"+split, second_all)
    
    
def plot_Rsp_scatter(list_of_sims, mids):
    xlim = np.array([0.7,1.4])
    ylim = (0.91,1.2)
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$E_{\\rm{kin}} / E_{\\rm{therm}}$"])
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.3,7),
                            gridspec_kw={'hspace' : 0, 'wspace' : 0})
                            # sharex=True, sharey=True)
    cm = plt.cm.get_cmap('rainbow')
    axes[0].plot(xlim, xlim, color="k")
    axes[1].plot(xlim, xlim, color="k")
    axes[2].plot(xlim, xlim, color="k")
    N_runs = len(list_of_sims)
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    for i in range(3):
        for k in range(N_runs):
            crange = axes[i].scatter(getattr(list_of_sims[k], "R_DM_" + bin_type[i]),
                                     getattr(list_of_sims[k], "R_gas_"+ bin_type[i]),
                                     c=mids[i,:], edgecolor="k", 
                                     cmap=cm, s=75, marker="o",
                                     label=getattr(list_of_sims[k], "run_label"))
        cbaxes = fig.add_axes([0.185, 0.21, 0.002, 0.005]) 
        cbar = fig.colorbar(crange, cax=cbaxes, label=labels[i])
        #colour bars don't quite work but don't need this plot anymore

    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    axes[2].legend(loc='upper left')
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
    axes[1].set_ylabel(r"$R_{\rm{SP,gas}} / R_{\rm{200m}}$")
    # filename = "splashback_data/flamingo/plots/compare_runs_new.png"
    # plt.savefig(filename, dpi=300)
    plt.show()


def plot_param_correlations(list_of_sims, ax, split, mids, 
                            ls, quantity="R_DM_"):
    N_runs = len(list_of_sims)
    lw = 1
    if quantity == "R_DM_" or quantity == "depth_DM_":
        cm = plt.cm.copper(np.linspace(0,1, N_runs))
    else:
        cm = plt.cm.winter(np.linspace(0,1, N_runs))
    for i in range(N_runs):
        flm = list_of_sims[i]
        Rsp = getattr(flm, quantity+split)
        hf_R = getattr(hf, quantity+split)
        if split == "mass":
            label = flm.run_label
            ax.set_xscale('log')
        else:
            label = ""
        if i == 0:
            error = getattr(hf, "error_"+quantity+split)
            ax.errorbar(mids, Rsp/hf_R, yerr=error*np.sqrt(2),
                        color=cm[i],
                        linestyle=ls[i],
                        linewidth=lw,
                        capsize=2)

        ax.plot(mids, Rsp/hf_R, # yerr=3*errors_DM, 
                color=cm[i], 
                label=label,
                linestyle=ls[i],
                linewidth=lw)
        ax.set_xlabel(axes_labels[split])

    
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
    
    
def stack_for_Rsp(list_of_sims):
    N_bins = 10
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    N_runs = len(list_of_sims)
    for i in range(N_runs):
        bin_profiles(list_of_sims[i], accretion_bins, mass_bins, energy_bins)
    
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
    plot_Rsp_scatter(list_of_sims, mids)
    
    
def stack_for_params(list_of_sims):
    N_bins = 10
    mass_bins = np.linspace(14, 15.2, N_bins+1)
    accretion_bins = np.linspace(0, 4.2, N_bins+1)
    energy_bins = np.linspace(0.05, 0.35, N_bins+1)
    mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
    accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
    energy_mid = (energy_bins[:-1] + energy_bins[1:])/2
    
    N_runs = len(list_of_sims)
    bootstrap = True
    for i in range(N_runs):
        bin_profiles(list_of_sims[i], accretion_bins, mass_bins, energy_bins,
                     bootstrap=bootstrap)
        bootstrap = False #only true for first sim
        second_caustic(list_of_sims[i], "accretion")
    linestyles = ["-", "--", (0,(1,1)), (0,(3,1,1,1)), (0,(1,2)),
                  (0, (3, 1, 1, 1, 1, 1)), (0, (3, 2, 1, 2)), (0, (5, 1))]

    fig, axes = plt.subplots(nrows=2, ncols=3, 
                             sharey="row",
                             figsize=(7,4),
                             gridspec_kw={'hspace' : 0, 'wspace' : 0})

    plot_param_correlations(list_of_sims, axes[0,0], "accretion", accretion_mid, 
                            linestyles)
    plot_param_correlations(list_of_sims, axes[0,1], "mass", mass_mid, 
                            linestyles)
    plot_param_correlations(list_of_sims, axes[0,2], "energy", energy_mid, 
                            linestyles)
        
    plot_param_correlations(list_of_sims, axes[1,0], "accretion", accretion_mid, 
                            linestyles, quantity="R_gas_")
    plot_param_correlations(list_of_sims, axes[1,1], "mass", mass_mid, 
                            linestyles, quantity="R_gas_")
    plot_param_correlations(list_of_sims, axes[1,2], "energy", energy_mid,
                            linestyles, quantity="R_gas_")
    
    fig.text(0.05, 0.45, r"$R_{\rm{SP,model}} / R_{\rm{SP,fiducial}}$",
             transform=fig.transFigure, rotation='vertical')
    axes[0,0].text(0.68, 0.05, "Dark matter",
             transform=axes[0,0].transAxes)
    axes[1,0].text(0.88, 0.05, "Gas",
             transform=axes[1,0].transAxes)
    axes[0,1].legend(ncol=2)
    axes[1,1].legend(ncol=2)
    axes[0,0].set_ylim((0.75, 1.2))
    axes[1,0].set_ylim((0.75, 1.2))
    # filename = "splashback_data/flamingo/plots/parameter_dependence_cosmo_Rsp.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(nrows=2, ncols=3, 
                              sharey="row",
                              figsize=(7,4),
                              gridspec_kw={'hspace' : 0, 'wspace' : 0})
    plot_param_correlations(list_of_sims, axes[0,0], "accretion", accretion_mid, 
                            linestyles, quantity="depth_DM_")
    plot_param_correlations(list_of_sims, axes[0,1], "mass", mass_mid, 
                            linestyles, quantity="depth_DM_")
    plot_param_correlations(list_of_sims, axes[0,2], "energy", energy_mid, 
                            linestyles, quantity="depth_DM_")
        
    plot_param_correlations(list_of_sims, axes[1,0], "accretion", accretion_mid, 
                            linestyles, quantity="depth_gas_")
    plot_param_correlations(list_of_sims, axes[1,1], "mass", mass_mid, 
                            linestyles, quantity="depth_gas_")
    plot_param_correlations(list_of_sims, axes[1,2], "energy", energy_mid,
                            linestyles, quantity="depth_gas_")
    
    fig.text(0.05, 0.45, r"$\gamma_{\rm{SP,model}} / \gamma_{\rm{SP,fiducial}}$",
              transform=fig.transFigure, rotation='vertical')
    axes[0,0].text(0.04, 0.05, "Dark matter",
             transform=axes[0,0].transAxes)
    axes[1,0].text(0.04, 0.05, "Gas",
             transform=axes[1,0].transAxes)
    axes[0,1].legend(ncol=2)
    axes[1,1].legend(ncol=2)
    axes[0,0].set_ylim((0.75, 1.2))
    axes[1,0].set_ylim((0.75, 1.2))
    # filename = "splashback_data/flamingo/plots/parameter_dependence_cosmo_gamma.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    
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
    
    hua = sp.flamingo(box, "HUA")
    hua.read_properties()
    
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
    
    # stack_for_profiles([hf, hwa, hsa, hta, hj, hsj])
    # stack_for_profiles(hp, hpf, hpv)

    # stack_for_params([hf, hwa, hsa, hta, hua, hj, hsj])
    stack_for_params([hf, hp, hpf, hpv])