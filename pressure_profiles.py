import numpy as np
import matplotlib.pyplot as plt
import splashback as sp

plt.style.use("mnras.mplstyle")
    
def plot_profiles_3D_all(flm, accretion_bins, mass_bins, energy_bins):
    """
    Plots stacked profiles for one run to compare the effect of using different
    stacking criteria to define the bins used to stack on the gas pressure and
    entropy profiles.

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

    Returns
    -------
    None.

    """
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$X_{\\rm{E}}$"])
    N_bins = len(accretion_bins) - 1
    ylim_P = (-4.5,1.4)
    ylim_K = (-0.6,1.4)
    fig, ax = plt.subplots(nrows=3, ncols=2, 
                           figsize=(3.3,4.5), 
                           sharey="col",
                           gridspec_kw={'hspace' : 0, 'wspace' : 0.2})
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
            ax[j,0].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_P")[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
            ax[j,1].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_K")[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    ax[0,0].set_ylim(ylim_P)
    #ax[0,1].set_ylim(ylim_K)
    ax[0,0].legend()
    ax[1,0].legend()
    ax[2,0].legend()
    ax[1,0].set_ylabel(r"$d \log f / d \log r$")
    plt.subplots_adjust(bottom=0.1)
    ax[0,0].set_xticklabels([])
    ax[1,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    ax[1,1].set_xticklabels([])
    fig.text(0.48, 0.02, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    ax[0,0].text(0.05, 0.07, "$f=P$", transform=ax[0,0].transAxes)
    ax[0,1].text(0.05, 0.07, "$f=K$", transform=ax[0,1].transAxes)
    plt.subplots_adjust(bottom=0.07, top=0.99, right=0.99)
    filename = "splashback_data/flamingo/plots/HF_compare_bins_pressure_entropy.png"
    plt.savefig(filename, dpi=300)
    plt.show()

box = "L1000N1800"
flm = sp.flamingo(box, "HF")
flm.read_properties()
flm.read_pressure()
flm.read_entropy()

N_bins = 4
mass_bins = np.linspace(14, 15, N_bins+1)
mass_bins = np.append(mass_bins, 16)
accretion_bins = np.linspace(0, 4, N_bins+1)
accretion_bins = np.append(accretion_bins, 20)
energy_bins = np.linspace(0.05, 0.3, N_bins+1)
energy_bins = np.append(energy_bins, 1)

mass_mid = np.zeros(N_bins+1)
accretion_mid = np.zeros(N_bins+1)
energy_mid = np.zeros(N_bins+1)
mass_mid[:-1] = (mass_bins[:-2] + mass_bins[1:-1])/2
accretion_mid[:-1] = (accretion_bins[:-2] + accretion_bins[1:-1])/2
energy_mid[:-1] = (energy_bins[:-2] + energy_bins[1:-1])/2
mass_mid[-1] = mass_bins[-2]
accretion_mid[-1] = accretion_bins[-2]
energy_mid[-1] = energy_bins[-2]

sp.stack_and_find_3D(flm, "accretion", accretion_bins)
sp.stack_and_find_3D(flm, "mass", mass_bins)
sp.stack_and_find_3D(flm, "energy", energy_bins)

plot_profiles_3D_all(flm, accretion_bins, mass_bins, energy_bins)
