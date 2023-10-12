import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
from scipy.optimize import curve_fit

plt.style.use("mnras.mplstyle")

axes_labels = {
    "mass": "$M_{\\rm{200m}}$",
    "accretion": "$\Gamma$",
    "energy": "$X_{\\rm{E}}$"}


def bin_profiles(data, accretion_bins, mass_bins, energy_bins):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. 

    Parameters
    ----------
    data : obj
        Simulation data.
    accretion_bins : array, float
        Bin edges for stacking according to the accretion rate
    mass_bins : array, float
        Bin edges for stacking following the cluster mass
    energy_bins : array. float
        bin edges for stacking following the cluster's energy ratio

    Returns
    -------
    None.
    """

    sp.stack_and_find_3D(data, "accretion", accretion_bins, bootstrap=True)
    sp.stack_and_find_3D(data, "mass", mass_bins, bootstrap=True)
    sp.stack_and_find_3D(data, "energy", energy_bins, bootstrap=True)
    
    
def second_caustic(split):
    """
    Determines which minima is the splashback and which is the second 
    caustic and saves the values accordingly

    Parameters
    ----------
    split : string
        Name of the criteria previously used to stack the profiles.

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
    
    
def plot_profiles_3D_all(flm, accretion_bins, mass_bins, energy_bins):
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

    Returns
    -------
    None.

    """
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$X_{\\rm{E}}$"])
    N_bins = len(accretion_bins) - 1
    ylim = (-4.1,0.5)
    fig, ax = plt.subplots(nrows=3, ncols=2, 
                           figsize=(4,5), 
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
            ax[j,0].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_DM")[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
            ax[j,1].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_gas")[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    ax[0,0].set_ylim(ylim)
    ax[0,0].legend()
    ax[1,0].legend()
    ax[2,0].legend()
    ax[1,0].set_ylabel(r"$d \log \rho / d \log r$")
    plt.subplots_adjust(bottom=0.1)
    ax[0,0].set_xticklabels([])
    ax[1,0].set_xticklabels([])
    fig.text(0.48, 0.05, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    ax[0,0].text(0.05, 0.07, "$\\rho=\\rho_{\\rm{DM}}$", transform=ax[0,0].transAxes)
    ax[0,1].text(0.05, 0.07, "$\\rho=\\rho_{\\rm{gas}}$", transform=ax[0,1].transAxes)
    filename = "splashback_data/flamingo/plots/HF_compare_bins_all.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def fit_model(accretion, a, b, c, d):
    """
    Model to get splashback radius from accretion rate.

    Parameters
    ----------
    accretion : float
        Accretion rate values to use to estimate the splashback radius.
    a : float
        Fitting parameter.
    b : float
        Fitting parameter.
    c : float
        Fitting parameter.
    d : float
        Fitting parameter.

    Returns
    -------
    Rsp : float
        Predicted splashabck radius values.

    """
    omega_m = 0.307
    Rsp = a * (1+b*omega_m) * (1+ c*np.exp(-1*accretion/d))
    return Rsp


def Rsp_model(accretion, model="More", ydata=[]):
    """
    Sorts which model paramters to use/fits the model to get the parameters
    and then gets an estimate of the splashback radius.

    Parameters
    ----------
    accretion : float
        Accretion rate values to use to estimate the splashback radius
    model : str, optional
        Name of the model parameters to use in the model that predictes the 
        splashback radius. The default is "More".
    ydata : array, optional
        If model=="mine", this is necessary to pass onto the fitting function.
        The default is [].

    Returns
    -------
    Rsp : float
        Estimated splashback radius values.

    """
    more = [0.54, 0.53, 1.36, 3.04]
    oneil = [0.8, 0.26, 1.14, 1.25]
    if model == "O'Neil":
        params = oneil    
    elif model == "mine":
        popt, pcov = curve_fit(fit_model, accretion, ydata,
                               p0=oneil)
        params = popt
    else:
        params = more
    Rsp = fit_model(accretion, *params)
    return Rsp
    

def plot_param_correlations(split, ax, plot="R"):
    """
    Plots one panel of a plot with relationship between a named stacking 
    criteria and the obtained splashback radius.

    Parameters
    ----------
    split : str
        Name of the stacking criteria used to create the profile to plot.
    ax : obj
        Pyplot axes object.
    plot : str, optional
        Which value to plot on the y-axis. Either "R" for the splashback radius
        or "depth" to plot the depth of the feature. The default is "R".

    Returns
    -------
    None.

    """
    plot_DM = getattr(flm, plot+"_DM_"+split)
    plot_gas = getattr(flm, plot+"_gas_"+split)
    plot_pressure = getattr(flm, plot+"_P_"+split)
    mids = getattr(flm, split+"_mid")
    errors_DM = getattr(flm, "error_"+plot+"_DM_"+split)
    errors_gas = getattr(flm, "error_"+plot+"_gas_"+split)
    errors_pressure = getattr(flm, "error_"+plot+"_P_"+split)
    label_DM = "Dark matter density"
    label_gas = "Gas density"
    label_P = "Gas pressure"
    # plt.figure()
    if split == "mass":
        ax.set_xscale('log')
    elif split == "accretion" and plot == "R":
        Rsp_more = Rsp_model(mids)
        Rsp_oneil = Rsp_model(mids, model="O'Neil")
        Rsp_mine = Rsp_model(mids, model="mine", ydata=plot_DM)
        ax.plot(mids, Rsp_more, 
                  color="darkmagenta", label="More 2015", linestyle="--")
        ax.plot(mids, Rsp_oneil, 
                  color="darkmagenta", label="O'Neil 2021", linestyle=":")
        label_DM = "Data"
        label_gas = ""
        label_P = ""
    ax.errorbar(mids, plot_DM, yerr=3*errors_DM, 
                color="darkmagenta", label=label_DM, capsize=2)
    ax.errorbar(mids, plot_gas, yerr=3*errors_gas, 
                color="gold", label=label_gas, capsize=2)
    ax.errorbar(mids, plot_pressure, yerr=3*errors_pressure, 
                color="c", label=label_P, capsize=2)
    ax.set_xlabel(axes_labels[split])
    
    
def stack_for_profiles(flm):
    """
    Defines bins and then plots 3D density profiles.

    Parameters
    ----------
    flm : obj
        Simulation data.

    Returns
    -------
    None.

    """
    N_bins = 4
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
    
    plot_profiles_3D_all(flm, accretion_bins, mass_bins, energy_bins)
    
    
def stack_for_params():
    """
    Defines bins and then plots relationship between the different stacking 
    criteria used to stack the profiles and either the radius or depth of the 
    splashback feature.

    Returns
    -------
    None.

    """
    N_bins = 10
    mass_bins = np.linspace(14, 15.6, N_bins+1)
    accretion_bins = np.linspace(0, 4.2, N_bins+1)
    energy_bins = np.linspace(0.05, 0.35, N_bins+1)
    
    bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
    second_caustic("accretion")
    
    flm.mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
    flm.accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
    flm.energy_mid = (energy_bins[:-1] + energy_bins[1:])/2

    fig, axes = plt.subplots(nrows=1, ncols=3, 
                              sharey=True,
                              figsize=(7,2),
                              gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("mass", axes[1])
    plot_param_correlations("accretion", axes[0])
    plot_param_correlations("energy", axes[2])
    axes[0].set_ylabel("$R_{\\rm{SP}} / R_{\\rm{200m}}$")
    axes[0].legend()
    axes[2].legend()
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/parameter_dependence_R.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=3, 
                              sharey=True,
                              figsize=(7,2),
                              gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("mass", axes[1], plot="depth")
    plot_param_correlations("accretion", axes[0], plot="depth")
    plot_param_correlations("energy", axes[2], plot="depth")
    axes[0].set_ylabel("$\gamma_{\\rm{SP}}$")
    axes[0].legend()
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/parameter_dependence_gamma.png"
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    box = "L2800N5040"
    flm = sp.flamingo(box, "HF")
    flm.read_properties()
    flm.read_pressure()
    
    # stack_for_profiles(flm)
    stack_for_params()