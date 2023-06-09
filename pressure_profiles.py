import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

box = "L1000N1800"

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
    if quantity == "P":
        ylim = (-5,0.1)
    elif quantity == "K":
        ylim = (-0.5,1.5)
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
    # filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    # plt.savefig(filename, dpi=300)
    plt.show()


flm = sp.flamingo(box, "HF")
flm.read_properties()
flm.read_pressure()
flm.read_entropy()
flm.read_2D()
# mass_mask = np.where(flm.M200m < 10**14.1)[0]
# flm.gas_pressure_3D = flm.gas_pressure_3D[mass_mask,:]
# flm.gas_entropy_3D = flm.gas_entropy_3D[mass_mask,:]
# flm.gas_density_3D = flm.gas_density_3D[mass_mask,:]
# flm.DM_density_3D = flm.DM_density_3D[mass_mask,:]
# flm.accretion = flm.accretion[mass_mask]
# flm.M200m = flm.M200m[mass_mask]
# flm.energy = flm.energy[mass_mask]

N_bins = 14
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

sp.stack_and_find_2D(flm, "accretion", accretion_bins)
sp.stack_and_find_2D(flm, "mass", mass_bins)
sp.stack_and_find_2D(flm, "energy", energy_bins)

cm_a = plt.cm.get_cmap('autumn')
cm_m = plt.cm.get_cmap('winter')
cm_e = plt.cm.get_cmap('copper')

plt.figure()
plt.scatter(flm.R_gas_accretion, flm.R_P_accretion, 
            edgecolor="k", c=accretion_mid, cmap=cm_a,
            marker="o")
plt.scatter(flm.R_gas_mass, flm.R_P_mass, 
            edgecolor="k", c=mass_mid, cmap=cm_m,
            marker="*")
plt.scatter(flm.R_gas_energy, flm.R_P_energy, 
            edgecolor="k", c=energy_mid, cmap=cm_e,
            marker="v")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("$R_{\\rm{SP,}\\rho_{\\rm{gas}}}$")
plt.ylabel("$R_{\\rm{SP,P}}$")
plt.show()

plt.figure()
plt.scatter(flm.R_gas_accretion, flm.R_EM_accretion, 
            edgecolor="k", c=accretion_mid, cmap=cm_a,
            marker="o")
plt.scatter(flm.R_gas_mass, flm.R_EM_mass, 
            edgecolor="k", c=mass_mid, cmap=cm_m,
            marker="*")
plt.scatter(flm.R_gas_energy, flm.R_EM_energy, 
            edgecolor="k", c=energy_mid, cmap=cm_e,
            marker="v")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("$R_{\\rm{SP,}\\rho_{\\rm{gas}}}$")
plt.ylabel("$R_{\\rm{SP,EM}}$")
plt.show()

plt.figure()
plt.scatter(flm.R_P_accretion, flm.R_SZ_accretion, 
            edgecolor="k", c=accretion_mid, cmap=cm_a,
            marker="o")
plt.scatter(flm.R_P_mass, flm.R_SZ_mass, 
            edgecolor="k", c=mass_mid, cmap=cm_m,
            marker="*")
plt.scatter(flm.R_P_energy, flm.R_SZ_energy, 
            edgecolor="k", c=energy_mid, cmap=cm_e,
            marker="v")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("$R_{\\rm{SP,P}}$")
plt.ylabel("$R_{\\rm{SP,SZ}}$")
plt.show()

plt.figure()
plt.scatter(flm.R_K_accretion, flm.R_SZ_accretion, 
            edgecolor="k", c=accretion_mid, cmap=cm_a,
            marker="o")
plt.scatter(flm.R_K_mass, flm.R_SZ_mass, 
            edgecolor="k", c=mass_mid, cmap=cm_m,
            marker="*")
plt.scatter(flm.R_K_energy, flm.R_SZ_energy, 
            edgecolor="k", c=energy_mid, cmap=cm_e,
            marker="v")
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("$R_{\\rm{SP,K}}$")
plt.ylabel("$R_{\\rm{SP,SZ}}$")
plt.show()

# plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
#                                quantity="P")
# plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
#                                quantity="K")

# mass_mask = np.where(flm.M200m > 1e15)[0]
# log_P = sp.log_gradients(flm.rad_mid, flm.gas_pressure_3D[mass_mask])
# for i in range(30):
#     plt.figure()
#     plt.semilogx(flm.rad_mid, log_P[i,:])
#     ylim_lower = plt.gca().get_ylim()[0]
#     ylim = (ylim_lower,0)
#     plt.ylim(ylim)
#     plt.title(i)
#     plt.show()