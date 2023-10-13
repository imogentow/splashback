import numpy as np
import matplotlib.pyplot as plt
import splashback as sp

plt.style.use("mnras.mplstyle")

axes_labels = {
    "mass": "$M_{\\rm{200m}}$",
    "accretion": "$\Gamma$",
    "energy": "$X_{\\rm{E}}$"}
        

def bin_profiles(data, accretion_bins, mass_bins, energy_bins, bootstrap="none"):
    """
    Takes a given run object and bins the density profiles according to
    given bin arrays. 
    
    Parameters
    ----------
    data : obj
        Simulation dataset object of choice
    accretion_bins : numpy array
        Array of bin edges used for stacking following the accretion rate.
    mass_bins : numpy array
        Array of bin edges used for stacking following the cluster mass.
    energy_bins : numpy array
        Array of bin edges used for stacking following the cluster energy 
        ratio.

    Returns
    -------
    None.

    """
    sp.stack_and_find_3D(data, "accretion", accretion_bins, bootstrap=bootstrap)
    sp.stack_and_find_3D(data, "mass", mass_bins, bootstrap=bootstrap)
    sp.stack_and_find_3D(data, "energy", energy_bins, bootstrap=bootstrap)

    
def second_caustic(flm, split):
    """
    Sorts out which minima in a profile is the splashback radius and which is
    the second caustic, replaces these values correctly.

    Parameters
    ----------
    flm : obj
        Simulation data.
    split : str
        Name of the original criteria used to stack the original profiles.

    Returns
    -------
    None.

    """
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


def plot_param_correlations(list_of_sims, ax, split, mids, 
                            ls, quantity="R_DM_"):
    """
    Plots parameter dependence onto one given panel of a figure.

    Parameters
    ----------
    list_of_sims : list
        List of simulation data objects.
    ax : obj
        Pyplot axes object.
    split : str
        Name of crtieria used to stack profiles that will be plotted on the 
        x-axis.
    mids : float
        Midpoints of bins for x-axis.
    ls : TYPE
        Linestyle for plotting.
    quantity : str, optional
        Name of quantity to plot values of. Options are "R_DM_", "depth_DM_",
        "R_gas_" and "depth_gas_". The default is "R_DM_".

    Returns
    -------
    None.

    """
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

    
def stack_for_params(list_of_sims):
    """
    Takes a variety of simulation runs and plots the dependence of 
    both the radius and depth of the splashback feature on the criteria used 
    to stack the profiles.

    Parameters
    ----------
    list_of_sims : list
        List of simulation objects.

    Returns
    -------
    None.

    """
    N_bins = 10
    mass_bins = np.linspace(14, 14.1, N_bins+1)
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
    # filename = "splashback_data/flamingo/plots/parameter_dependence_all_runs_Rsp.png"
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
    # filename = "splashback_data/flamingo/plots/parameter_dependence_all_runs_gamma.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    

if __name__ == "__main__":
    box = "L1000N1800"   
    
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

    stack_for_params([hf, hwa, hsa, hta, hua, hj, hsj])
    # stack_for_params([hf, hp, hpf, hpv])