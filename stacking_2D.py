import numpy as np
import matplotlib.pyplot as plt
import splashback as sp

plt.style.use("mnras.mplstyle")
    
def bin_profiles(d, bins, list_of_names,
                 bootstrap=False,
                 print_data=False):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. Also calculates a best fit gradient, assuming that 
    the line goes through the origin.
    
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
    radii = np.logspace(-1, 0.7, 45)
    R200m = np.genfromtxt(flm.path+"_R200m.csv", delimiter=",")
    area = np.pi * (radii[1:]**2 - radii[:-1]**2) * 10 / 50 #in R200m not mpc
    area = np.outer(R200m, area)
    area = np.vstack((area, area, area))
    flm.SZ_median = flm.SZ_median / area
    
    
def plot_profiles_compare_bins(flm, bins, quantity="EM"):
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
    bin_type = np.array(["accretion", "mass", "energy"])
    labels = np.array(["$\Gamma$", "$\log M_{\\rm{200m}}$", "$E_{\\rm{kin}} / E_{\\rm{therm}}$"])
    N_bins = bins.shape[1] - 1
    ylim = (-4,0.5)
    fig, ax = plt.subplots(nrows=3, ncols=1, 
                           figsize=(3,6), 
                           sharey=True,
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
            ax[j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    # ax[0].set_ylim(ylim)
    # ax[1].set_ylim(ylim)
    # ax[2].set_ylim(ylim)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[2].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[1].set_ylabel(r"$d \log \Sigma_{{\rm{{{}}}}} / d \log r$".format(quantity)) #need to change this label manually atm
    # filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    bin_type = np.array(["concentration", "symmetry", "alignment", "centroid"])
    labels = np.array(["$c$", "$s$", "$a$", r"$\langle w \rangle$"])
    N_bins = bins.shape[1] - 1
    ylim = (-4,0.5)
    fig, ax = plt.subplots(nrows=2, ncols=2, 
                           figsize=(4,4), 
                           sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm1 = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    cm2 = plt.cm.winter(np.linspace(0,1,N_bins))
    cm3 = plt.cm.copper(np.linspace(0,1,N_bins))
    lw = 0.8
    for i in range(0, N_bins):
        for j in range(4):
            if i == 0:
                label = labels[j] + "$<$"  + str(np.round(bins[j+3,1],2))
            elif i == N_bins-1:
                label = labels[j] + "$>$" + str(np.round(bins[j+3,i],2))
            else:
                label = str(np.round(bins[j+3,i],2)) \
                    + "$<$" + labels[j] + "$<$" \
                    + str(np.round(bins[j+3,i+1],2))
            if j == 0:
                cm = cm1
                ax[0,j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
            elif j==1:
                cm = cm2
                ax[0,j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
            else:
                cm = cm3
                ax[1,j-2].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantity)[i,:], 
                           color=cm[i], linewidth=lw,
                           label=label)
    # ax[0].set_ylim(ylim)
    # ax[1].set_ylim(ylim)
    # ax[2].set_ylim(ylim)
    for a in ax.flat:
        a.legend()
    plt.xlabel("$r/R_{\\rm{200m}}$")
    plt.ylabel(r"$d \log \Sigma_{{\rm{{{}}}}} / d \log r$".format(quantity)) #need to change this label manually atm
    # filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    
def check_proj_Rsp(data, mids, quantity):
    fig, ax = plt.subplots(nrows=1, ncols=3, 
                           sharey=True,
                           figsize=(6,2.5),
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.get_cmap('rainbow')
    cacc = ax[0].scatter(data.R_DM_accretion, getattr(data, "R_" + quantity +"_accretion"),
                         c=mids[0,:], 
                         edgecolor="k", 
                         cmap=cm, s=75)
    cmass = ax[1].scatter(data.R_DM_mass, getattr(data, "R_" + quantity +"_mass"),
                          c=mids[1,:], 
                          edgecolor="k", 
                          cmap=cm, s=75)
    cener = ax[2].scatter(data.R_DM_energy, getattr(data, "R_" + quantity +"_energy"),
                          c=mids[2,:], 
                          edgecolor="k", 
                          cmap=cm, s=75)
    ax[1].set_xlabel("$R_{\\rm{SP,DM}}$")
    ax[0].set_ylabel(r"$R_{{\rm{{SP,{}}}}}$".format(quantity))
    cbaxes1 = fig.add_axes([0.09, 0.8, 0.015, 0.12]) 
    cbar = fig.colorbar(cacc, cax=cbaxes1, label="$\Gamma$")
    cbaxes2 = fig.add_axes([0.39, 0.8, 0.015, 0.12]) 
    cbar = fig.colorbar(cmass, cax=cbaxes2, label="$\log M_{\\rm{200m}}$")
    cbaxes3 = fig.add_axes([0.7, 0.8, 0.015, 0.12]) 
    cbar = fig.colorbar(cener, cax=cbaxes3, label="$E_{\\rm{kin}}/E_{\\rm{therm}}$")
    plt.show()
    
    fig, ax = plt.subplots(nrows=2, ncols=2, 
                           sharey=True,
                           figsize=(4,4),
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.get_cmap('rainbow')
    cc = ax[0,0].scatter(data.R_DM_concentration, getattr(data, "R_" 
                                                         + quantity 
                                                         + "_concentration"),
                         c=mids[3,:], 
                         edgecolor="k", 
                         cmap=cm, s=75)
    cs = ax[0,1].scatter(data.R_DM_symmetry, getattr(data, "R_" 
                                                     + quantity 
                                                     + "_symmetry"),
                         c=mids[4,:], 
                         edgecolor="k", 
                         cmap=cm, s=75)
    ca = ax[1,0].scatter(data.R_DM_alignment, getattr(data, "R_" 
                                                      + quantity 
                                                      +"_alignment"),
                         c=mids[5,:], 
                         edgecolor="k", 
                         cmap=cm, s=75)
    cw = ax[1,1].scatter(data.R_DM_centroid, getattr(data, "R_" 
                                                     + quantity 
                                                     +"_centroid"),
                         c=mids[6,:], 
                         edgecolor="k", 
                         cmap=cm, s=75)
    ax[1,0].set_xlabel("$R_{\\rm{SP,DM}}$")
    ax[0,0].set_ylabel(r"$R_{{\rm{{SP,{}}}}}$".format(quantity))
    cbaxes1 = fig.add_axes([0.15, 0.86, 0.015, 0.08]) 
    cbar = fig.colorbar(cc, cax=cbaxes1, label="$c$")
    cbaxes2 = fig.add_axes([0.57, 0.86, 0.015, 0.08]) 
    cbar = fig.colorbar(cs, cax=cbaxes2, label="$s$")
    cbaxes3 = fig.add_axes([0.15, 0.42, 0.015, 0.08]) 
    cbar = fig.colorbar(ca, cax=cbaxes3, label="$a$")
    cbaxes4 = fig.add_axes([0.57, 0.42, 0.015, 0.08]) 
    cbar = fig.colorbar(cw, cax=cbaxes4, label=r"$\langle w \rangle$")
    plt.show()
    
    
def compare_morphology(data, mids):
    fig, ax = plt.subplots(nrows=3, ncols=1,
                           sharex=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(3.3,6))
    bin_type = np.array(["concentration", "symmetry", "alignment", "centroid"])
    quantities = ["EM", "SZ", "WL"]
    size=50
    for i in range(3):
       cc =  ax[i].scatter(getattr(data, "R_DM_concentration"), 
                           getattr(data, "R_" + quantities[i] +"_concentration"),
                           c=mids[0,:], cmap=plt.cm.get_cmap("autumn"),
                           edgecolor="k",
                           s=size,
                           marker="o",
                           label="$c$")
       cs = ax[i].scatter(getattr(data, "R_DM_symmetry"), 
                          getattr(data, "R_" + quantities[i] +"_symmetry"),
                          c=mids[1,:], cmap=plt.cm.get_cmap("winter"),
                          edgecolor="k",
                          s=size,
                          marker="*",
                          label="$s$")
       ca = ax[i].scatter(getattr(data, "R_DM_alignment"), 
                          getattr(data, "R_" + quantities[i] +"_alignment"),
                          c=mids[2,:], cmap=plt.cm.get_cmap("copper"),
                          edgecolor="k",
                          s=size,
                          marker="^",
                          label="$a$")
       cw = ax[i].scatter(getattr(data, "R_DM_centroid"), 
                          getattr(data, "R_" + quantities[i] +"_centroid"),
                          c=mids[3,:], cmap=plt.cm.get_cmap("spring"),
                          edgecolor="k",
                          s=size,
                          marker="P",
                          label=r"$\log \langle w \rangle$")

    cbaxes1 = fig.add_axes([0.67, 0.58, 0.02, 0.08]) 
    fig.colorbar(cc, cax=cbaxes1, label="$c$")
    cbaxes2 = fig.add_axes([0.82, 0.58, 0.02, 0.08]) 
    fig.colorbar(cs, cax=cbaxes2, label="$s$")
    cbaxes3 = fig.add_axes([0.185, 0.285, 0.02, 0.08]) 
    fig.colorbar(ca, cax=cbaxes3, label="$a$")
    cbaxes4 = fig.add_axes([0.335, 0.285, 0.02, 0.08]) 
    fig.colorbar(cw, cax=cbaxes4, label=r"$\log \langle w \rangle$")

    ax[0].legend()
    ylim = ax[1].get_ylim()
    ax[1].set_ylim((ylim[0], 1.57))
    ax[0].set_ylabel("$R_{\\rm{SP,EM}} / R_{\\rm{200m}}$")
    ax[1].set_ylabel("$R_{\\rm{SP,SZ}} / R_{\\rm{200m}}$")
    ax[2].set_ylabel("$R_{\\rm{SP,WL}} / R_{\\rm{200m}}$")
    ax[2].set_xlabel("$R_{\\rm{SP,DM}} / R_{\\rm{200m}}$")
    # filename = "splashback_data/flamingo/plots/compare_Rsp_morph.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
    
def compare_best(data, mids):
    fig, ax = plt.subplots(nrows=3, ncols=3,
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(5,4))
    bin_type = np.array(["symmetry", "centroid", "gap"])
    quantities = ["EM", "SZ", "WL"]
    size=50
    winter = plt.cm.get_cmap("winter")
    spring = plt.cm.get_cmap("spring")
    autumn = plt.cm.get_cmap("autumn")
    for i in range(3):
       cs = ax[i,0].scatter(getattr(data, "R_DM_symmetry"), 
                          getattr(data, "R_" + quantities[i] +"_symmetry"),
                          c=mids[1,:], cmap=winter,
                          edgecolor="k",
                          s=size,
                          marker="*",
                          label="$s$")

       cw = ax[i,1].scatter(getattr(data, "R_DM_centroid"), 
                          getattr(data, "R_" + quantities[i] +"_centroid"),
                          c=mids[3,:], cmap=spring,
                          edgecolor="k",
                          s=size,
                          marker="P",
                          label=r"$\log \langle w \rangle$")
       
       cg = ax[i,2].scatter(getattr(data, "R_DM_gap"), 
                          getattr(data, "R_" + quantities[i] +"_gap"),
                          c=mids[3,:], cmap=autumn,
                          edgecolor="k",
                          s=size,
                          marker="o",
                          label=r"$M14$")

    cbaxes1 = fig.add_axes([0.14, 0.78, 0.015, 0.08]) 
    fig.colorbar(cs, cax=cbaxes1, label="$s$")
    cbaxes2 = fig.add_axes([0.40, 0.78, 0.015, 0.08]) 
    fig.colorbar(cw, cax=cbaxes2, label=r"$\log \langle w \rangle$")
    cbaxes3 = fig.add_axes([0.67, 0.78, 0.015, 0.08]) 
    fig.colorbar(cg, cax=cbaxes3, label=r"$M14$")

    # ax[0,0].legend()
    # ax[0,1].legend()
    ylim = ax[1,0].get_ylim()
    ax[1,0].set_ylim((ylim[0], 1.4))
    ax[0,0].set_ylabel("$R_{\\rm{SP,EM}} / R_{\\rm{200m}}$")
    ax[1,0].set_ylabel("$R_{\\rm{SP,SZ}} / R_{\\rm{200m}}$")
    ax[2,0].set_ylabel("$R_{\\rm{SP,WL}} / R_{\\rm{200m}}$")
    
    plt.subplots_adjust(bottom=0.1)
    plt.text(0.45, 0.01, "$R_{\\rm{SP,DM}} / R_{\\rm{200m}}$", 
             transform=fig.transFigure)
    
    filename = "splashback_data/flamingo/plots/compare_Rsp_morph.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def plot_observables(data, bins):
    fig, ax = plt.subplots(nrows=3, ncols=5, 
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(8,5))
    bin_type = np.array(["concentration", "symmetry", "alignment", "centroid", "gap"])
    labels = np.array(["$c$", "$s$", "$a$", r"$ \log \langle w \rangle$", "$M14$"])
    N_bins = bins.shape[1] - 1

    cmap_bins = np.linspace(0,0.95, N_bins)
    cmaps = ["autumn", "winter", "copper", "spring", "cool"]
    quantities = ["EM", "SZ", "WL"]

    lw = 0.8
    for i in range(0, N_bins):
        for j in range(5):
            cm = getattr(plt.cm, cmaps[j])(cmap_bins)
            for k in range(3):
                if i == 0 and k == 0:
                    label = labels[j] + "$<$"  + str(np.round(bins[j,1],2))
                elif i == N_bins-1 and k==0:
                    label = labels[j] + "$>$" + str(np.round(bins[j,i],2))
                elif k == 0:
                    label = str(np.round(bins[j,i],2)) \
                        + "$<$" + labels[j] + "$<$" \
                        + str(np.round(bins[j,i+1],2))
                else:
                    label=None
                
                ax[k,j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantities[k])[i,:], 
                                 color=cm[i], linewidth=lw,
                                 label=label)

    for a in range(5):
        ax[0,a].legend()
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    ylim = ax[0,0].get_ylim()
    ax[0,0].set_ylim((ylim[0],3))
    plt.text(0.5, 0.05, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    ax[0,0].set_ylabel(r"$d \log \Sigma_{\rm{EM}} / d \log r$")
    ax[1,0].set_ylabel(r"$d \log \Sigma_{\rm{SZ}} / d \log r$")
    ax[2,0].set_ylabel(r"$d \log \Sigma_{\rm{WL}} / d \log r$")
    filename = "splashback_data/flamingo/plots/profiles_2D_observables.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def mass_cut(data, mass_range, quantities):
    mass_range = 10**mass_range
    mass_cut = np.where((data.M200m >= mass_range[0]) & 
                        (data.M200m < mass_range[1]))[0]
    N_quant = len(quantities)
    for i in range(N_quant):
        values = getattr(data, quantities[i])
        setattr(data, quantities[i], values[mass_cut])
    
    
def stack_for_profiles():
    N_bins = 5
    c_bins = np.append(np.linspace(0.0, 0.4, N_bins), 1)
    s_bins = np.append(-1.5, np.append(np.linspace(0.05, 1.4, int(N_bins-1)), 2.2))
    a_bins = np.append(-1., np.append(np.linspace(0.5, 1.5, N_bins-1), 5))
    w_bins = np.append(-5, np.append(np.linspace(-2.7, -1, N_bins-1), 0))
    gap_bins = np.append(np.linspace(0,2.5, N_bins), 8)
    
    mass_restriction = np.array([14.2, 14.4])
    quantities_to_restrict = ["concentration", "symmetry", "alignment", "centroid",
                              "EM_median", "SZ_median", "WL_median", "gap"]
    mass_cut(flm, mass_restriction, quantities_to_restrict)
    
    bins = np.vstack((c_bins, s_bins, a_bins, w_bins, gap_bins))
    list_of_bins = ["concentration", "symmetry", "alignment", "centroid", "gap"]
    bin_profiles(flm, bins, list_of_bins)
    
    plot_observables(flm, bins)
    
    
def stack_for_Rsp():
    N_bins = 10
    c_bins = np.linspace(0.0, 0.4, N_bins+1)
    s_bins = np.linspace(0.05, 1.4, N_bins+1)
    a_bins = np.linspace(0.5, 1.5, N_bins+1)
    w_bins = np.linspace(-2.7, -1, N_bins+1)
    gap_bins = np.linspace(0,2.5, N_bins+1)
    bins = np.vstack((c_bins, s_bins, a_bins, w_bins, gap_bins))
    list_of_bins = ["concentration", "symmetry", "alignment", "centroid", "gap"]
    bin_profiles(flm, bins, list_of_bins)
    
    for i in range(len(list_of_bins)):
        sp.stack_and_find_3D(flm, list_of_bins[i], bins[i,:])

    c_mid = (c_bins[:-1] + c_bins[1:])/2
    s_mid = (s_bins[:-1] + s_bins[1:])/2
    a_mid = (a_bins[:-1] + a_bins[1:])/2
    w_mid = (w_bins[:-1] + w_bins[1:])/2
    gap_mid = (gap_bins[:-1] + gap_bins[1:])/2
    mids = np.vstack((c_mid, s_mid, a_mid, w_mid, gap_mid))
    
    compare_best(flm, mids)
    
    
def plot_param_correlations(split, ax):
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
        "mass": "$M_{\\rm{200m}}$",
        "accretion": "$\Gamma$",
        "energy": "$E_{\\rm{kin}}/E_{\\rm{therm}}$",
        "symmetry": "$s$",
        "centroid": "$\log \langle w \\rangle$",
        "gap": "$M14$"}

    if split == "mass":
        ax.set_xscale('log')
    ax.errorbar(mids, Rsp_EM, yerr=errors_EM, 
                color="gold", label=label_EM, capsize=2)
    ax.errorbar(mids, Rsp_SZ, yerr=errors_SZ, 
                color="c", label=label_SZ, capsize=2)
    ax.errorbar(mids, Rsp_WL, yerr=errors_WL, 
                color="darkmagenta", label=label_WL, capsize=2)
    ax.set_xlabel(axes_labels[split])

    
def stack_for_params():
    N_bins = 10
    mass_bins = np.linspace(14, 15.6, N_bins+1)
    accretion_bins = np.linspace(0, 4.2, N_bins+1)
    energy_bins = np.linspace(0.05, 0.35, N_bins+1)
    s_bins = np.linspace(0.05, 1.4, N_bins+1)
    w_bins = np.linspace(-2.7, -1, N_bins+1)
    gap_bins = np.linspace(0,2.5, N_bins+1)
    
    bin_profiles(flm, np.vstack((accretion_bins, mass_bins, energy_bins,
                                 s_bins, w_bins, gap_bins)), 
                 ["accretion", "mass", "energy","symmetry", "centroid", "gap"],
                 bootstrap=True)
    flm.R_WL_accretion, flm.second_WL_accretion = sp.second_caustic(flm.R_WL_accretion, 
                                                                    flm.second_WL_accretion)
    
    flm.mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
    flm.accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
    flm.energy_mid = (energy_bins[:-1] + energy_bins[1:])/2
    flm.symmetry_mid = (s_bins[:-1] + s_bins[1:])/2
    flm.centroid_mid = (w_bins[:-1] + w_bins[1:])/2
    flm.gap_mid = (gap_bins[:-1] + gap_bins[1:])/2

    fig, axes = plt.subplots(nrows=1, ncols=3, 
                             sharey=True,
                             figsize=(7,2),
                             gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("mass", axes[1])
    plot_param_correlations("accretion", axes[0])
    plot_param_correlations("energy", axes[2])
    axes[0].set_ylabel("$R_{\\rm{minima}} / R_{\\rm{200m}}$")
    axes[0].legend()
    # ylim = axes[0].get_ylim()
    # ylim = (0, ylim[1])
    # axes[0].set_ylim(ylim)
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/parameter_dependence_2D.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=3, 
                             sharey=True,
                             figsize=(7,2),
                             gridspec_kw={'hspace' : 0.1, 'wspace' : 0})
    plot_param_correlations("symmetry", axes[1])
    plot_param_correlations("centroid", axes[0])
    plot_param_correlations("gap", axes[2])
    axes[0].set_ylabel("$R_{\\rm{minima}} / R_{\\rm{200m}}$")
    axes[0].legend()
    # ylim = axes[0].get_ylim()
    # ylim = (0, ylim[1])
    # axes[0].set_ylim(ylim)
    plt.subplots_adjust(bottom=0.18)
    filename = "splashback_data/flamingo/plots/obs_parameter_dependence_2D.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def bin_random(data, bins, split, N_clusters,
               print_data=False):
    if split == "mass":
        split_name = "M200m"
        split_data = np.log10(getattr(data, split_name))
    else:
        split_data = getattr(data, split)
    N_profiles = 2
        
    
    saving_strings = ["_P", "_SZ"]
    pressure = data.gas_pressure_3D
    SZ = data.SZ_median[:len(pressure)]
    stacking_data = np.dstack((pressure, SZ))
    
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], bins)
    N_bins = len(bins) - 1
    stacked_data = np.zeros((N_bins, data.N_rad, N_profiles))
    if print_data:
        print("")
        print(split)
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        sample = np.random.choice(bin_mask, size=N_clusters, replace=False)
        if print_data:
            print(len(bin_mask))
        for j in range(N_profiles):
            stacked_data[i,:,j] = sp.stack_data(stacking_data[not_nan,:,j][sample,:])
            
    for i in range(N_profiles):
        log = sp.log_gradients(data.rad_mid, stacked_data[:,:,i])
        setattr(data, split+ "_profile" + saving_strings[i], stacked_data[:,:,i]) #previously density instead of profile
        setattr(data, split+ "_log" + saving_strings[i], log)


def stack_test():
    N_bins = 4
    mass_bins = np.linspace(14, 15.2, N_bins+1)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    energy_bins = np.linspace(0.05, 0.35, N_bins+1)
    
    mass_cut(flm, np.array([14.5,15]), ["accretion", "SZ_median", "gas_pressure_3D"])
    
    bin_random(flm, accretion_bins, "accretion", 30, print_data=True) 
                  
    flm.mass_mid = 10**((mass_bins[:-1] + mass_bins[1:])/2)
    cm = plt.cm.autumn(np.linspace(0,0.95,N_bins))
    plt.figure()
    for i in range(N_bins):
        plt.semilogx(flm.rad_mid, flm.accretion_log_P[i,:],
                     color=cm[i])
    plt.show()
    plt.figure()
    for i in range(N_bins):
        plt.semilogx(flm.rad_mid, flm.accretion_log_SZ[i,:],
                     color=cm[i])
    plt.show()
    
    # plt.figure()
    # plt.plot(flm.mass_mid, flm.R_P_mass, 
    #               label="Pressure", color="b")
    # plt.plot(flm.mass_mid, flm.R_SZ_mass, 
    #               label="Compton-y", color="c")
    # plt.plot(flm.mass_mid, flm.R_gas_mass, 
    #               label="Gas density", color="r")
    # plt.plot(flm.mass_mid, flm.R_EM_mass, 
    #               label="emission measure", color="gold")
    # plt.ylabel("$R_{\\rm{minima}} / R_{\\rm{200m}}$")
    # plt.legend()
    # plt.xscale('log')
    # # plt.subplots_adjust(bottom=0.18)
    # # filename = "splashback_data/flamingo/plots/parameter_dependence_2D.png"
    # # plt.savefig(filename, dpi=300)
    # plt.show()
    
    # for index in range(10):
    #     # gas_density = sp.log_gradients(flm.rad_mid, flm.gas_density_3D[index,:])
    #     pressure = sp.log_gradients(flm.rad_mid, flm.gas_pressure_3D[index,:])
    #     # EM = sp.log_gradients(flm.rad_mid, flm.EM_median[index,:])
    #     SZ = sp.log_gradients(flm.rad_mid, flm.SZ_median[index,:])
    #     plt.figure()
    #     # plt.semilogx(flm.rad_mid, gas_density,
    #     #          color="r", label="Gas density")
    #     # plt.semilogx(flm.rad_mid, EM,
    #     #          color="gold", label="Emission measure")
    #     plt.semilogx(flm.rad_mid, pressure,
    #              color="b", label="Pressure")
    #     plt.semilogx(flm.rad_mid, SZ,
    #              color="c", label="Compton-y")
    #     plt.legend()
    #     plt.ylim((-7,1))
    #     plt.show()


N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

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
    # stack_for_Rsp()
    stack_for_params()
    # stack_test()
    
