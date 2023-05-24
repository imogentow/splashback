import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L2800N5040"

def bin_profiles(d, accretion_bins, mass_bins, energy_bins):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. 
    
    Inputs.
    d: obj, run of choice
    accretion_bins: array giving edge values of bins of accretion rate
    mass_bins: array giving edge values of bins of mass
    energy_bins: array giving edge values of bins of energy ratio.
    """
    sp.stack_and_find_3D(d, "accretion", accretion_bins)
    sp.stack_and_find_3D(d, "mass", mass_bins)
    sp.stack_and_find_3D(d, "energy", energy_bins)
    
    
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
    ylim = (-4,0.5)
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
            ax[j].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    ax[0].set_ylim(ylim)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[1].set_ylabel(r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity))
    filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    plt.savefig(filename, dpi=300)
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
    ylim = (0.87,1.2)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    cm1 = plt.cm.get_cmap('autumn')
    cm2 = plt.cm.get_cmap('winter')
    cm3 = plt.cm.get_cmap('copper')
    axes.plot(xlim, xlim, color="k")
    crange1 = axes.scatter(run.R_DM_accretion, run.R_gas_accretion, 
                           c=mids[0,:], edgecolor="k", 
                           cmap=cm1, s=75, marker="o",
                           label="Accretion rate")
    crange2 = axes.scatter(run.R_DM_mass, run.R_gas_mass, 
                          c=mids[1,:], edgecolor="k", 
                          cmap=cm2, s=75, marker="*",
                          label="Mass")
    crange3 = axes.scatter(run.R_DM_energy, run.R_gas_energy, 
                           c=mids[2,:], edgecolor="k", 
                           cmap=cm3, s=75, marker="v",
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
    filename = "compare_bins_HF.png"
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
    bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
    
    plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
                               quantity="DM")
    
def stack_for_Rsp():
    N_bins = 20
    mass_bins = np.linspace(14,
                            np.nanpercentile(np.log10(flm.M200m), 99),
                            N_bins+1)
    accretion_bins = np.linspace(0, 
                                 np.nanpercentile(flm.accretion[np.isfinite(flm.accretion)], 99), 
                                 N_bins+1)
    energy_bins = np.linspace(np.min(flm.energy), 
                              np.nanpercentile(flm.energy, 99), 
                              N_bins+1)
    bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
    flm.mass_mid = (mass_bins[:-1] + mass_bins[1:])/2
    flm.accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
    flm.energy_mid = (energy_bins[:-1] + energy_bins[1:])/2
    mids = np.vstack((flm.accretion_mid, flm.mass_mid, flm.energy_mid))

    plot_Rsp_scatter_one(flm, mids)

if __name__ == "__main__":
    flm = sp.flamingo(box, "HF")
    flm.read_properties()
    
    stack_for_profiles()
    stack_for_Rsp()