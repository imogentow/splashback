import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr
import stacking_3D as s3D

plt.style.use("mnras.mplstyle")
    
def bin_profiles(d, bins):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. Also calculates a best fit gradient, assuming that 
    the line goes through the origin.
    
    Inputs.
    d: obj, run of choice
    bins: array giving edge values of bins of accretion rate, mass and energy
    ratio.
    """
    sp.stack_and_find_2D(d, "accretion", bins[0,:])
    sp.stack_and_find_2D(d, "mass", bins[1,:])
    sp.stack_and_find_2D(d, "energy", bins[2,:])
    sp.stack_and_find_2D(d, "concentration", bins[3,:])
    sp.stack_and_find_2D(d, "symmetry", bins[4,:])
    sp.stack_and_find_2D(d, "alignment", bins[5,:])
    sp.stack_and_find_2D(d, "centroid", bins[6,:])
    
    
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
    
    
def scatter_compare(data, mids):
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
    
    
def scatter_compare_sw(data, mids):
    fig, ax = plt.subplots(nrows=3, ncols=2,
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(3.3,4))
    bin_type = np.array(["symmetry", "centroid"])
    quantities = ["EM", "SZ", "WL"]
    size=50
    for i in range(3):
       cs = ax[i,0].scatter(getattr(data, "R_DM_symmetry"), 
                          getattr(data, "R_" + quantities[i] +"_symmetry"),
                          c=mids[1,:], cmap=plt.cm.get_cmap("winter"),
                          edgecolor="k",
                          s=size,
                          marker="*",
                          label="$s$")

       cw = ax[i,1].scatter(getattr(data, "R_DM_centroid"), 
                          getattr(data, "R_" + quantities[i] +"_centroid"),
                          c=mids[3,:], cmap=plt.cm.get_cmap("spring"),
                          edgecolor="k",
                          s=size,
                          marker="P",
                          label=r"$\log \langle w \rangle$")

    cbaxes2 = fig.add_axes([0.45, 0.7, 0.02, 0.08]) 
    fig.colorbar(cs, cax=cbaxes2, label="$s$")
    cbaxes4 = fig.add_axes([0.82, 0.7, 0.02, 0.08]) 
    fig.colorbar(cw, cax=cbaxes4, label=r"$\log \langle w \rangle$")

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
    
    
def plot_stack_morph(data, bins):
    fig, ax = plt.subplots(nrows=3, ncols=4, 
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(7,5))
    bin_type = np.array(["concentration", "symmetry", "alignment", "centroid"])
    labels = np.array(["$c$", "$s$", "$a$", r"$ \log \langle w \rangle$"])
    N_bins = bins.shape[1] - 1

    cmap_bins = np.linspace(0,0.95,N_bins)
    cmaps = ["autumn", "winter", "copper", "spring"]
    quantities = ["EM", "SZ", "WL"]

    lw = 0.8
    for i in range(0, N_bins):
        for j in range(4):
            cm = getattr(plt.cm, cmaps[j])(cmap_bins)
            for k in range(3):
                if i == 0 and k == 0:
                    label = labels[j] + "$<$"  + str(np.round(bins[j+3,1],2))
                elif i == N_bins-1 and k==0:
                    label = labels[j] + "$>$" + str(np.round(bins[j+3,i],2))
                elif k == 0:
                    label = str(np.round(bins[j+3,i],2)) \
                        + "$<$" + labels[j] + "$<$" \
                        + str(np.round(bins[j+3,i+1],2))
                else:
                    label=None
                
                ax[k,j].semilogx(rad_mid, getattr(flm, bin_type[j] + "_log_" + quantities[k])[i,:], 
                                 color=cm[i], linewidth=lw,
                                 label=label)

    for a in ax.flat:
        a.legend()
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    ylim = ax[0,0].get_ylim()
    ax[0,0].set_ylim((ylim[0],3))
    plt.text(0.5, 0.05, "$r/R_{\\rm{200m}}$", transform=fig.transFigure)
    ax[0,0].set_ylabel(r"$d \log \Sigma_{\rm{EM}} / d \log r$")
    ax[1,0].set_ylabel(r"$d \log \Sigma_{\rm{SZ}} / d \log r$")
    ax[2,0].set_ylabel(r"$d \log \Sigma_{\rm{WL}} / d \log r$")
    # filename = "splashback_data/flamingo/plots/profiles_2D_morphology.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
def stack_for_profiles():
    N_bins = 5
    mass_bins = np.linspace(14, 15, N_bins)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins)
    energy_bins = np.append(energy_bins, 1)
    c_bins = np.linspace(0.0, 0.4, N_bins)
    c_bins = np.append(c_bins, 1)
    s_bins = np.linspace(0.0, 1.5, N_bins-1) #set extra limits on both sides
    s_bins = np.append(s_bins, 2.2)
    s_bins = np.append(-1.5, s_bins)
    a_bins = np.linspace(0.4, 1.6, N_bins-1) #set extra limits on both sides
    a_bins = np.append(a_bins, 5)
    a_bins = np.append(-1., a_bins)
    w_bins = np.linspace(-3, -1, N_bins-1) #set extra limits on both sides
    w_bins = np.append(w_bins, 0)
    w_bins = np.append(-5, w_bins)
    
    bins = np.vstack((accretion_bins, mass_bins, energy_bins, c_bins,
                      s_bins, a_bins, w_bins))
    bin_profiles(flm, bins)

    # plot_profiles_compare_bins(flm, bins, quantity="EM")
    # plot_profiles_compare_bins(flm, bins, quantity="SZ")
    # plot_profiles_compare_bins(flm, bins, quantity="WL")
    
    plot_stack_morph(flm, bins)
    
    
def stack_for_Rsp():
    N_bins = 15
    mass_bins = np.linspace(14, 15, N_bins)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins)
    energy_bins = np.append(energy_bins, 1)
    c_bins = np.linspace(0.0, 0.4, N_bins)
    c_bins = np.append(c_bins, 1)
    s_bins = np.linspace(0.0, 1.5, N_bins-1) #set extra limits on both sides
    s_bins = np.append(s_bins, 2.2)
    s_bins = np.append(-1.5, s_bins)
    a_bins = np.linspace(0.4, 1.6, N_bins-1) #set extra limits on both sides
    a_bins = np.append(a_bins, 5)
    a_bins = np.append(-1., a_bins)
    w_bins = np.linspace(-3, -1, N_bins-1) #set extra limits on both sides
    w_bins = np.append(w_bins, 0)
    w_bins = np.append(-5, w_bins)
    
    bins = np.vstack((accretion_bins, mass_bins, energy_bins, c_bins,
                      s_bins, a_bins, w_bins))
    bin_profiles(flm, bins)
    s3D.bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
    sp.stack_and_find_3D(flm, "concentration", c_bins)
    sp.stack_and_find_3D(flm, "symmetry", s_bins)
    sp.stack_and_find_3D(flm, "alignment", a_bins)
    sp.stack_and_find_3D(flm, "centroid", w_bins)

    mass_mid = np.zeros(N_bins)
    accretion_mid = np.zeros(N_bins)
    energy_mid = np.zeros(N_bins)
    c_mid = np.zeros(N_bins)
    s_mid = np.zeros(N_bins)
    a_mid = np.zeros(N_bins)
    w_mid = np.zeros(N_bins)

    mass_mid[:-1] = (mass_bins[:-2] + mass_bins[1:-1])/2
    accretion_mid[:-1] = (accretion_bins[:-2] + accretion_bins[1:-1])/2
    energy_mid[:-1] = (energy_bins[:-2] + energy_bins[1:-1])/2
    c_mid[:-1] = (c_bins[:-2] + c_bins[1:-1])/2
    s_mid[1:-1] = (s_bins[1:-2] + s_bins[2:-1])/2
    a_mid[1:-1] = (a_bins[1:-2] + a_bins[2:-1])/2
    w_mid[1:-1] = (w_bins[1:-2] + w_bins[2:-1])/2
    mass_mid[-1] = mass_bins[-2]
    accretion_mid[-1] = accretion_bins[-2]
    energy_mid[-1] = energy_bins[-2]
    c_mid[-1] = c_bins[-2]
    s_mid[[0,-1]] = [s_bins[1], s_bins[-2]]
    a_mid[[0,-1]] = [a_bins[1], a_bins[-2]]
    w_mid[[0,-1]] = [w_bins[1], w_bins[-2]]
    
    mids = np.vstack((accretion_mid, mass_mid, energy_mid, c_mid,
                      s_mid, a_mid, w_mid))
    
    # check_proj_Rsp(flm, mids, "EM")
    # check_proj_Rsp(flm, mids, "SZ")
    # check_proj_Rsp(flm, mids, "WL")
    
    scatter_compare(flm, mids[3:,:])
    scatter_compare_sw(flm, mids[3:,:])
    
    # plt.scatter(c_mid, flm.R_EM_concentration)
    # plt.show()
    
    # plt.scatter(s_mid, flm.R_EM_symmetry)
    # plt.show()
    
    # plt.scatter(a_mid, flm.R_EM_alignment)
    # plt.show()
    
    # plt.scatter(w_mid, flm.R_EM_centroid)
    # plt.show()

N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

if __name__ == "__main__":
    box = "L1000N1800"
    
    flm = sp.flamingo(box, "HF")
    flm.read_2D()
    flm.read_2D_properties()
    flm.read_properties()
    
    # stack_for_profiles()
    stack_for_Rsp()
    
