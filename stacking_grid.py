import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr
from scipy.stats import binned_statistic_2d as bin2d

plt.style.use("mnras.mplstyle")
    
    
def stack_grid_bins(data, mass_bins, accretion_bins):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj.
    
    Inputs.
    data: obj, run of choice
    mass_bins: mass bins to use
    accretion_bins: mass bins to use
    """
    not_nan = np.where((np.isfinite(data.M200m)==True) &
                       (np.isfinite(data.accretion)==True))[0]
    mass_values = np.log10(data.M200m[not_nan])
    accretion_values = data.accretion[not_nan]

    binning = bin2d(mass_values, accretion_values, None, 'count',
                      bins=[mass_bins, accretion_bins],
                      expand_binnumbers=True)
    bins_sort = binning.binnumber
    bin_counts = binning.statistic
    print(bin_counts)
    N_mass = len(mass_bins) - 1 
    N_acc = len(accretion_bins) - 1
    
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
            
        log_DM[i,:,:] = sp.log_gradients(rad_mid, stacked_DM[i,:,:])
        log_gas[i,:,:] = sp.log_gradients(rad_mid, stacked_gas[i,:,:])
        R_SP_DM[i,:], depth_DM[i,:] = dr.depth_cut(rad_mid, log_DM[i,:,:], 
                                                   depth_value="y")
        R_SP_gas[i,:], depth_gas[i,:] = dr.depth_cut(rad_mid, log_gas[i,:,:], 
                                                     cut=-1, depth_value="y")
    
    data.log_DM = log_DM
    data.log_gas = log_gas
    data.R_DM = R_SP_DM
    data.R_gas = R_SP_gas
    data.depth_DM = depth_DM
    data.depth_gas = depth_gas
    

def grid_plot(data, mass_bins, accretion_bins):
    N_acc = len(accretion_bins) - 1
    N_mass = len(mass_bins) - 1
    fig, ax = plt.subplots(nrows=N_acc, ncols=1,
                           figsize=(3,6),
                           sharex=True, sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm = plt.cm.copper(np.linspace(0,1,N_mass))
    for i in range(N_acc):
        for j in range(N_mass):
            if j == N_mass-1:
                label = r"$M_{\rm{200m}} >$" + str(np.round(mass_bins[j],2))
            else:
                label = str(np.round(mass_bins[j],2)) \
                + r"$< M_{\rm{200m}} <$" \
                + str(np.round(mass_bins[j+1],2))
            ax[i].semilogx(rad_mid, data.log_gas[j,i,:], 
                           color=cm[j], linewidth=0.8,
                           label=label)
        if i == N_acc-1:
            plot_label = r"$\Gamma >$" + str(np.round(accretion_bins[i],2))
        else:
            plot_label = str(np.round(accretion_bins[i],2)) \
                        + r"$< \Gamma <$" \
                        + str(np.round(accretion_bins[i+1],2))
        ax[i].text(0.05, 0.05, plot_label, transform=ax[i].transAxes)
    ax[0].legend()
    plt.show()
    


def stack_for_profiles():
    N_mass = 7
    N_acc = 4
    mass_bins = np.linspace(13, 14.8, N_mass)
    accretion_bins = np.linspace(0, 3, N_acc)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.append(accretion_bins,20)
    
    stack_grid_bins(flm_HF, mass_bins, accretion_bins)
    grid_plot(flm_HF, mass_bins, accretion_bins)
    
    plt.figure()
    for i in range(N_acc):
        plt.semilogx(10**mass_bins[:-1], flm_HF.depth_gas[:,i], marker="o")
    plt.show()
    


if __name__ == "__main__":
    box = "L1000N1800"
    
    N_rad = 44
    log_radii = np.linspace(-1, 0.7, N_rad+1)
    rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2
    
    flm_HF = sp.flamingo(box, "HF")
    flm_HF.read_properties()
    flm_HF.read_low_mass()
    
    stack_for_profiles()