import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr
from scipy.stats import binned_statistic_2d as bin2d

plt.style.use("mnras.mplstyle")
    
labels = {
    "mass": "$\log M_{\\rm{200m}}$",
    "concentration": "$c$",
    "symmetry": "$s$",
    "alignment": "$a$",
    "centroid": "$\log\langle w \\rangle$"}
    

def stack_grid_bins(data, quantity1, quantity2):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj.

    Parameters
    ----------
    data : obj
        Simulation data.
    quantity1 : str
        Name of first criteria used for stacking.
    quantity2 : str
        Name of second criteria used for stacking.

    Returns
    -------
    None.

    """

    bins1 = getattr(data, quantity1+"_bins")
    bins2 = getattr(data, quantity2+"_bins")
    if quantity1 == "mass":
        quantity1 = "M200m"
        property1 = getattr(data, quantity1)
        property1 = np.log10(property1)
    else:
        property1 = getattr(data, quantity1)
    property2 = getattr(data, quantity2)
    if len(property1) > len(property2):
        property2 = np.hstack((property2, property2, property2))

    not_nan = np.where((np.isfinite(property1)==True) &
                       (np.isfinite(property2)==True))[0]
    values1 = property1[not_nan]
    values2 = property2[not_nan]

    binning = bin2d(values1, values2, None, 'count',
                      bins=[bins1, bins2],
                      expand_binnumbers=True)
    bins_sort = binning.binnumber
    N_mass = len(bins1) - 1 
    N_acc = len(bins2) - 1
    N_rad = len(data.rad_mid)
    
    stacked_DM = np.zeros((N_mass, N_acc, N_rad))
    stacked_gas = np.zeros((N_mass, N_acc, N_rad))
    
    log_DM = np.zeros((N_mass, N_acc, N_rad))
    log_gas = np.zeros((N_mass, N_acc, N_rad))
    R_SP_DM = np.zeros((N_mass, N_acc))
    R_SP_gas = np.zeros((N_mass, N_acc))
    depth_DM = np.zeros((N_mass, N_acc))
    depth_gas = np.zeros((N_mass, N_acc))

    for i in range(N_mass):
        for j in range(N_acc):
            bin_mask = np.where((bins_sort[0,:] == i+1) &
                                (bins_sort[1,:] == j+1))[0]
            stacked_DM[i,j,:] = sp.stack_data(data.DM_density_3D[not_nan][bin_mask,:])
            stacked_gas[i,j,:] = sp.stack_data(data.gas_density_3D[not_nan][bin_mask,:])
            
        log_DM[i,:,:] = sp.log_gradients(data.rad_mid, stacked_DM[i,:,:])
        log_gas[i,:,:] = sp.log_gradients(data.rad_mid, stacked_gas[i,:,:])
        R_SP_DM[i,:], depth_DM[i,:] = dr.depth_cut(data.rad_mid, log_DM[i,:,:], 
                                                   depth_value="y")
        R_SP_gas[i,:], depth_gas[i,:] = dr.depth_cut(data.rad_mid, log_gas[i,:,:], 
                                                     cut=-1, depth_value="y")
    
    data.log_DM = log_DM
    data.log_gas = log_gas
    data.R_DM = R_SP_DM
    data.R_gas = R_SP_gas
    data.depth_DM = depth_DM
    data.depth_gas = depth_gas
    
    
def grid_plot_both(data, quantity1, quantity2):
    """
    Plots profiles in bins of two different criteria.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    quantity : TYPE
        DESCRIPTION.
    accretion_bins : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    bins1 = getattr(data, quantity1+"_bins")
    bins2 = getattr(data, quantity2+"_bins")
    N_1 = len(bins1) - 1
    N_2 = len(bins2) - 1
    fig, ax = plt.subplots(nrows=N_2, ncols=2,
                           figsize=(4,6),
                           sharex=True, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.copper(np.linspace(0,1,N_1))
    symbol = labels[quantity1]
    for i in range(N_2):
        for j in range(N_1):
            label = str(np.round(bins1[j],2)) \
            + r"$<$" + symbol + "$<$" \
            + str(np.round(bins1[j+1],2))
            ax[i,0].semilogx(data.rad_mid, data.log_DM[j,i,:], 
                             color=cm[j], linewidth=0.8,
                             label=label)
            ax[i,1].semilogx(data.rad_mid, data.log_gas[j,i,:], 
                             color=cm[j], linewidth=0.8,
                             label=label)
        plot_label = str(np.round(bins2[i],2)) \
                        + r"$< X_{\rm{E}} <$" \
                        + str(np.round(bins2[i+1],2))
        ax[i,0].text(0.05, 0.05, plot_label, transform=ax[i,0].transAxes)
    ax[0,0].text(0.85, 0.06, "$\\rho_{\\rm{DM}}$", transform=ax[0,0].transAxes)
    ax[0,1].text(0.85, 0.06, "$\\rho_{\\rm{gas}}$", transform=ax[0,1].transAxes)
    fig.text(0.02, 0.45, r"$d \log \rho / d \log r$",
             transform=fig.transFigure, rotation='vertical')
    fig.text(0.45, 0.02, "$r/R_{\\rm{200m}}$",
             transform=fig.transFigure,)
    ax[0,0].legend()
    ax[0,0].set_ylim((-4.5, 0.5))
    plt.subplots_adjust(left=0.11, bottom=0.08)
    # filename = "splashback_data/flamingo/plots/grid_stacking_gas_DM.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    

def stack_for_profiles(data):
    N_bins = 5
    data.mass_bins = np.linspace(14, 15, N_bins)
    data.accretion_bins = np.linspace(0, 4, N_bins)
    data.energy_bins = np.linspace(0, 0.4, N_bins)
 
    stack_grid_bins(data, "mass", "energy")
    grid_plot_both(data, "mass", "energy")
    

if __name__ == "__main__":
    box = "L2800N5040"
    flm_HF = sp.flamingo(box, "HF")
    flm_HF.read_properties()
    
    stack_for_profiles(flm_HF)