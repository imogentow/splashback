import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import fitting as ft

plt.style.use("mnras.mplstyle")
H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G) #kg/m^3
unit_converter = 1.99e30 / (3.09e22**3)
rho_crit = rho_crit / unit_converter

axes_labels = {
    "mass": "$M_{\\rm{200m}}$",
    "accretion": "$\Gamma$",
    "energy": "$X_{\\rm{E}}$",
    "concentration": "$c$",
    "symmetry": "$s$",
    "alignment": "$a$",
    "centroid": "$\langle w \\rangle$",
    "gap": "$M14$"}
    

def bin_profiles(d, bins, list_of_names,
                 bootstrap=False,
                 print_data=False):
    """
    Takes a given run object and bins the density profiles according to
    given bin arrays. 
    
    Parameters
    ----------
    d : obj
        Simulation dataset object of choice
    bins : numpy array
        Array of bins used for stacking. (N_types, N_bins)
    list_of_names : list
        List of names of stacking criteria, given as strings

    Returns
    -------
    None.

    """
    N_stack = bins.shape[0]
    for i in range(N_stack):
        sp.stack_and_find_2D(d, list_of_names[i], 
                             bins[i,:], 
                             bootstrap=bootstrap, 
                             print_data=print_data)
        sp.stack_and_find_3D(d, list_of_names[i], 
                             bins[i,:], 
                             bootstrap=bootstrap, 
                             print_data=print_data)
        

def convert_SZ_profiles():
    """
    Original data didn't account for changing bin sizes used to measure SZ
    profiles, this fixes that.

    Returns
    -------
    None.

    """
    radii = np.logspace(-1, 0.7, 45)
    R200m = np.genfromtxt(flm.path+"_R200m.csv", delimiter=",")
    area = np.pi * (radii[1:]**2 - radii[:-1]**2) * 10 / 50 #in R200m not mpc
    area = np.outer(R200m, area)
    area = np.vstack((area, area, area))
    flm.SZ_median = flm.SZ_median / area
    
    
def plot_observables(data, bins, bin_type):
    """
    Plots projected profiles according to a given bin.

    Parameters
    ----------
    data : obj
        Simulation data.
    bins : floats
        2d array, each row lists bin edges used for previously stacking 
        profiles.
    bin_type : list, str
        Names of criteria used to stack profiles previously. In an order 
        matching order of bins.

    Returns
    -------
    None.

    """
    N_bins = bins.shape[1] - 1
    N_stack = bins.shape[0]
    
    fig, ax = plt.subplots(nrows=3, ncols=N_stack, 
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(7.5,5))
    
    cmap_bins = np.linspace(0,0.95, N_bins)
    cmaps = ["autumn", "winter", "copper", "spring", "cool"]
    quantities = ["EM", "SZ", "WL"]

    lw = 0.8
    for i in range(N_bins):
        for j in range(N_stack):
            cm = getattr(plt.cm, cmaps[j])(cmap_bins)
            for k in range(3):
                if i == 0 and k == 0:
                    label = axes_labels[bin_type[j]] + "$<$"  + str(np.round(bins[j,1],2))
                elif i == N_bins-1 and k==0:
                    label = axes_labels[bin_type[j]] + "$>$" + str(np.round(bins[j,i],2))
                elif k == 0:
                    label = str(np.round(bins[j,i],2)) \
                        + "$<$" + axes_labels[bin_type[j]] + "$<$" \
                        + str(np.round(bins[j,i+1],2))
                else:
                    label=None
                
                ax[k,j].semilogx(flm.rad_mid, getattr(flm, bin_type[j] + "_log_" + quantities[k])[i,:], 
                                 color=cm[i], linewidth=lw,
                                 label=label)

    for a in range(N_stack):
        ax[0,a].legend()
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    ylim = ax[0,0].get_ylim()
    ax[0,0].set_ylim((ylim[0],3))
    plt.text(0.5, 0.05, "$R/R_{\\rm{200m}}$", transform=fig.transFigure)
    ax[0,0].set_ylabel(r"$d \log \Sigma_{\rm{EM}} / d \log r$")
    ax[1,0].set_ylabel(r"$d \log \Sigma_{\rm{y}} / d \log r$")
    ax[2,0].set_ylabel(r"$d \log \Sigma_{\rm{WL}} / d \log r$")
    filename = "splashback_data/flamingo/plots/profiles_2D_observables.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def mass_cut(data, mass_range, quantities):
    """
    Applies a mass cut to listed quantities.

    Parameters
    ----------
    data : obj
        simulation data.
    mass_range : float
        Mass range values, cut out clusters with a mass outside this range.
    quantities : list, str
        Names of attributes in data to apply mass cut to.

    Returns
    -------
    None.

    """
    mass_range = 10**mass_range
    mass_cut = np.where((data.M200m >= mass_range[0]) & 
                        (data.M200m < mass_range[1]))[0]
    N_quant = len(quantities)
    for i in range(N_quant):
        values = getattr(data, quantities[i])
        setattr(data, quantities[i], values[mass_cut])
    
    
def stack_for_profiles():
    """
    Stacks and plots observable profiles.

    Returns
    -------
    None.

    """
    N_bins = 5
    # mass_bins = np.linspace(14, 15, N_bins)
    # mass_bins = np.append(mass_bins, 16)
    # accretion_bins = np.linspace(0, 4, N_bins)
    # accretion_bins = np.append(accretion_bins, 20)
    # energy_bins = np.linspace(0.05, 0.3, N_bins)
    # energy_bins = np.append(energy_bins, 1)
    c_bins = np.append(np.linspace(0.0, 0.4, N_bins), 1)
    s_bins = np.append(-1.5, np.append(np.linspace(0.05, 1.4, int(N_bins-1)), 2.2))
    a_bins = np.append(-1., np.append(np.linspace(0.5, 1.5, N_bins-1), 5))
    w_bins = np.append(-5, np.append(np.linspace(-2.7, -1, N_bins-1), 0))
    gap_bins = np.append(np.linspace(0,2.5, N_bins), 8)
    
    mass_restriction = np.array([14.2, 14.4])
    quantities_to_restrict = ["concentration", "symmetry", "alignment", "centroid",
                              "EM_median", "SZ_median", "WL_median", "gap",
                              "M200m", "accretion", "energy"]
    mass_cut(flm, mass_restriction, quantities_to_restrict)
    
    bins = np.vstack((c_bins, s_bins, a_bins, w_bins, gap_bins))
    list_of_bins = ["concentration", "symmetry", "alignment", "centroid", "gap"]
    # bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    # list_of_bins = ["accretion", "mass", "energy"]
    bin_profiles(flm, bins, list_of_bins)
    
    plot_observables(flm, bins, list_of_bins)
    
    
def plot_param_correlations(split, ax):
    """
    Plots parameter dependence for one property one one figure panel.
    Calculates expected parameter dependence from projected 3D dark matter
    density.

    Parameters
    ----------
    split : str
        Name of criteria used to stack profiles and the parameter plotted on
        the x-axis.
    ax : obj
        Pyplot axis object, panel to plot data on.

    Returns
    -------
    None.

    """
    Rsp_EM = getattr(flm, "R_EM_"+split)
    Rsp_SZ = getattr(flm, "R_SZ_"+split)
    Rsp_WL = getattr(flm, "R_WL_"+split)
    mids = getattr(flm, split+"_mid")
    errors_EM = getattr(flm, "error_R_EM_"+split)
    errors_SZ = getattr(flm, "error_R_SZ_"+split)
    errors_WL = getattr(flm, "error_R_WL_"+split)
    label_EM = "Emission measure"
    label_SZ = "Compton-y"
    label_WL = "Surface density"
    
    axes_labels = {
        "mass": "$M_{\\rm{200m}} / M_{\odot}$",
        "accretion": "$\Gamma$",
        "energy": "$X_{\\rm{E}}$",
        "symmetry": "$s$",
        "centroid": "$\log \langle w \\rangle$",
        "gap": "$M14$"}
    params = ft.fit_log_models(flm, split)
    projected_density = ft.project_model(flm.rad_mid, params)
    projected_model_log_DM = sp.log_gradients(flm.rad_mid, projected_density,
                                                  smooth=False)
    # Projected splashback model from 3D
    R_model = ft.find_sort_R(flm, flm.rad_mid, projected_model_log_DM, 
                ["model", split])
    # errors_model = ft.bootstrap_errors(flm, split)
    if split == "mass":
        ax.set_xscale('log')
    ax.errorbar(mids, Rsp_EM, yerr=errors_EM, 
                color="gold", label=label_EM, capsize=2)
    ax.errorbar(mids, Rsp_SZ, yerr=errors_SZ, 
                color="c", label=label_SZ, capsize=2)
    ax.errorbar(mids, Rsp_WL, yerr=errors_WL, 
                color="darkmagenta", label=label_WL, capsize=2)
    # ax.errorbar(mids, R_model, yerr=errors_model,
    #             color="k", label="Projected model", capsize=2)
    # ax.plot(mids, Rsp_EM,
    #         color="gold", label=label_EM,)
    # ax.plot(mids, Rsp_SZ, 
    #         color="c", label=label_SZ)
    # ax.plot(mids, Rsp_WL, 
    #         color="darkmagenta", label=label_WL)
    # ax.plot(mids, R_model, 
    #         color="k", label="Projected model")
    
    ax.set_xlabel(axes_labels[split])

    
def stack_for_params():
    """
    Takes simulation data and plots Rsp and as a function of the criteria
    used to stack the projected observable profiles.

    Returns
    -------
    None.

    """
    N_bins = 10
    flm.mass_bins = np.linspace(14, 15.2, N_bins+1)
    flm.accretion_bins = np.linspace(0, 4.2, N_bins+1)
    flm.energy_bins = np.linspace(0.05, 0.35, N_bins+1)
    flm.symmetry_bins = np.linspace(0.05, 1.4, N_bins+1)
    flm.centroid_bins = np.linspace(-2.7, -1, N_bins+1)
    flm.gap_bins = np.linspace(0,2.5, N_bins+1)
    
    bin_profiles(flm, np.vstack((flm.accretion_bins, flm.mass_bins, flm.energy_bins,
                                 flm.symmetry_bins, flm.centroid_bins, flm.gap_bins)), 
                 ["accretion", "mass", "energy","symmetry", "centroid", "gap"],
                 bootstrap=True)
    
    # test_bootstrap_iteration(flm, "M200m", flm.mass_bins)
    flm.R_WL_accretion, flm.second_WL_accretion = sp.second_caustic(flm.R_WL_accretion, 
                                                                    flm.second_WL_accretion)
    
    flm.mass_mid = 10**((flm.mass_bins[:-1] + flm.mass_bins[1:])/2)
    flm.accretion_mid = (flm.accretion_bins[:-1] + flm.accretion_bins[1:])/2
    flm.energy_mid = (flm.energy_bins[:-1] + flm.energy_bins[1:])/2
    flm.symmetry_mid = (flm.symmetry_bins[:-1] + flm.symmetry_bins[1:])/2
    flm.centroid_mid = (flm.centroid_bins[:-1] + flm.centroid_bins[1:])/2
    flm.gap_mid = (flm.gap_bins[:-1] + flm.gap_bins[1:])/2

    N_clusters = 15719
    flm.symmetry = flm.symmetry[:N_clusters]
    flm.centroid = flm.centroid[:N_clusters]
    flm.gap = flm.gap[:N_clusters]
    
    fig, axes = plt.subplots(nrows=1, ncols=3, 
                              sharey=True,
                              figsize=(7,2),
                              gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("mass", axes[1])
    plot_param_correlations("accretion", axes[0])
    plot_param_correlations("energy", axes[2])
    axes[0].set_ylabel("$R_{\\rm{min}} / R_{\\rm{200m}}$")
    axes[0].legend()
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/parameter_dependence_2D_new.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=3, 
                              sharey=True,
                              figsize=(7,2),
                              gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("symmetry", axes[1])
    plot_param_correlations("gap", axes[2])
    plot_param_correlations("centroid", axes[0])
    axes[0].set_ylabel("$R_{\\rm{min}} / R_{\\rm{200m}}$")
    axes[0].legend()
    axes[0].set_ylim((0.6, 1.79))
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/obs_parameter_dependence_2D_new.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def test_bootstrap_iteration(flm, split, split_bins):
    """
    Looks at samples within bootstrap resampling, uses previously identified 
    most extreme sampling and investigates the differences between the highest
    and lowest Rsp iterations.

    Parameters
    ----------
    flm : onj
        Simulation data.
    split : str
        Name of criteria used to stack profiles.
    split_bins : float
        Bin edges to use to split clusters and stack profiles within.

    Returns
    -------
    None.

    """
    split_data = np.log10(getattr(flm, split))
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    N_rad = len(flm.rad_mid)
    
    stacked_data_SZ = np.zeros((4, N_rad))
    log_sample_SZ = np.zeros((4, N_rad))
    stacked_data_DM = np.zeros((4, N_rad))
    log_sample_DM = np.zeros((4, N_rad))
    
    iteration = np.array([[91, 8], [1, 69]])
    bin_number = np.array([0, 9])
    k = 0
    for i in range(2): #vary bin choice
        for j in range(2): #upper and lower iterations
            bin_mask = np.where(bins_sort == bin_number[i]+1)[0]
            np.random.seed(iteration[i,j]*15 + bin_number[i])
            # Select random sample from bin with replacement
            sample = np.random.choice(bin_mask, 
                                      size=len(bin_mask),
                                      replace=True)
            stacked_data_SZ[k,:] = sp.stack_data(flm.SZ_median[not_nan[sample],:])
            stacked_data_DM[k,:] = sp.stack_data(flm.DM_density_3D[not_nan[sample],:])
            k += 1
    log_sample_SZ = sp.log_gradients(flm.rad_mid, stacked_data_SZ)
    log_sample_DM = sp.log_gradients(flm.rad_mid, stacked_data_DM)
    
    
    fig, axes = plt.subplots(nrows=2, ncols=2,
                              sharey="row",
                              figsize=(4,2.5),
                              gridspec_kw={'hspace' : 0, 'wspace' : 0})
    axes[0,0].semilogx(flm.rad_mid, log_sample_DM[1,:], 
                        color="purple", label="WL")
    axes[0,1].semilogx(flm.rad_mid, log_sample_DM[0,:], 
                        color="purple")
    axes[0,0].semilogx(flm.rad_mid, log_sample_SZ[1,:],
                        color="c", label="SZ")
    axes[0,1].semilogx(flm.rad_mid, log_sample_SZ[0,:], 
                        color="c")
    
    axes[1,0].semilogx(flm.rad_mid, log_sample_DM[3,:], 
                        color="purple")
    axes[1,1].semilogx(flm.rad_mid, log_sample_DM[2,:], 
                        color="purple")
    axes[1,0].semilogx(flm.rad_mid, log_sample_SZ[3,:], 
                        color="c")
    axes[1,1].semilogx(flm.rad_mid, log_sample_SZ[2,:], 
                        color="c")
    
    axes[0,0].legend()
    axes[0,0].text(0.05, 0.05, "Low $M_{\\rm{200m}}$", transform=axes[0,0].transAxes)
    axes[1,0].text(0.05, 0.05, "High $M_{\\rm{200m}}$", transform=axes[1,0].transAxes)
    axes[0,0].text(0.05, 0.90, "Low $R_{\\rm{SP}}$", transform=axes[0,0].transAxes)
    axes[0,1].text(0.05, 0.90, "High $R_{\\rm{SP}}$", transform=axes[0,1].transAxes)
    plt.text(0.5, 0.04, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    plt.text(0.0, 0.5, r"$d \log \Sigma / d \log r$", 
              transform=fig.transFigure, rotation="vertical")
    plt.show()


if __name__ == "__main__":
    box = "L1000N1800"
    
    flm = sp.flamingo(box, "HF")
    flm.read_pressure()
    flm.read_2D()
    flm.read_2D_properties()
    flm.read_properties()
    flm.read_magnitude_gap(twodim=True)
    convert_SZ_profiles()
    
    # stack_for_profiles()
    stack_for_params()
    
