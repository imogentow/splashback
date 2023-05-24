import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G) #kg/m^3
unit_converter = 1.99e30 / (3.09e22**3)
rho_crit = rho_crit / unit_converter

def bin_profiles(data, accretion_bins, mass_bins, energy_bins):
    """
    Takes a given run object and bins the density profiles according to
    3 given bin arrays. Also calculates a best fit gradient, assuming that 
    the line goes through the origin.
    
    Inputs.
    d: obj, run of choice
    accretion_bins: array giving edge values of bins of accretion rate
    mass_bins: array giving edge values of bins of mass
    energy_bins: array giving edge values of bins of energy ratio.
    """
    
    sp.stack_and_find_3D(data, "mass", mass_bins)
    sp.stack_and_find_3D(data, "energy", energy_bins)
    sp.stack_fixed(data, "accretion", accretion_bins, dim="3D")
    
    log_DM = getattr(data, "accretion_log_DM")
    log_gas = getattr(data, "accretion_log_gas")
    R_SP_DM, second_DM, depth_DM, second_depth = dr.depth_cut(data.rad_mid, 
                                                    log_DM, 
                                                    depth_value="y", 
                                                    second_caustic="y")
    R_SP_gas, depth_gas = dr.depth_cut(data.rad_mid, 
                                                   log_gas, 
                                                   cut=-2.5, 
                                                   depth_value="y")
    second_mask = np.where(np.isfinite(second_DM))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if R_SP_DM[index] < second_DM[index]:
            larger = second_DM[index]
            smaller = R_SP_DM[index]
            R_SP_DM[index] = larger
            second_DM[index] = smaller
            depth1 = depth_DM[index]
            depth2 = second_depth[index]
            depth_DM[index] = depth2
            second_depth[index] = depth1
    setattr(data, "R_DM_accretion", R_SP_DM)
    setattr(data, "R_gas_accretion", R_SP_gas)
    setattr(data, "second_DM_accretion", second_DM)
    setattr(data, "depth_DM_accretion", depth_DM)
    setattr(data, "depth_gas_accretion", depth_gas)
    

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
    bin_type = np.array(["accretion", "sphericity_gas", "sphericity_DM"])
    labels = np.array(["$\Gamma$", "$S_{\\rm{gas}}$", "$S_{\\rm{DM}}$"])
    N_bins = len(accretion_bins) - 1
    ylim = (-4.5,0.5)
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
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[2].set_ylim(ylim)
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    ax[2].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[1].set_ylabel(r"$d \log \rho_{{\rm{{{}}}}} / d \log r$".format(quantity))
    # filename = "splashback_data/flamingo/plots/HF_compare_bins.png"
    # plt.savefig(filename, dpi=300)
    plt.show()
    
def plot_correlations(sgas, sDM, accretion, quantity="R"):
    fig, ax = plt.subplots(nrows=1, ncols=3, 
                           figsize=(5,2), 
                           sharey=True, 
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    
    ax[0].scatter(accretion, getattr(flm, quantity+"_DM_accretion"),
                    marker="o", color="darkorchid", edgecolor="k",
                    label="DM")
    ax[1].scatter(sDM, getattr(flm, quantity+"_DM_sphericity_DM"),
                    marker="o", color="darkorchid", edgecolor="k")
    ax[2].scatter(sgas, getattr(flm, quantity+"_DM_sphericity_gas"),
                    marker="o", color="darkorchid", edgecolor="k")
    ax[0].scatter(accretion, getattr(flm, quantity+"_gas_accretion"),
                    marker="^", color="c", edgecolor="k",
                    label="Gas")
    ax[1].scatter(sDM, getattr(flm, quantity+"_gas_sphericity_DM"),
                    marker="^", color="c", edgecolor="k")
    ax[2].scatter(sgas, getattr(flm, quantity+"_gas_sphericity_gas"),
                    marker="^", color="c", edgecolor="k")
    
    if quantity == "depth":
        quantity = "\gamma"
    ax[0].set_ylabel("${}_{{\\rm{{SP}}}}$".format(quantity))
    ax[0].set_xlabel("$\Gamma$")
    ax[1].set_xlabel("$S_{\\rm{DM}}$")
    ax[2].set_xlabel("$S_{\\rm{gas}}$")
    ax[0].legend()
    plt.show()
    
def correlations_quartiles(stack, stack_bins, stack_mids, label):
    
    not_nan = np.where(np.isfinite(stack)==True)[0]
    bins_sort = np.digitize(stack[not_nan], stack_bins)
    N_bins = len(stack_bins) - 1
    
    stacked_data_DM = np.zeros((N_bins, N_rad))
    upper_data_DM = np.zeros((N_bins, N_rad))
    lower_data_DM = np.zeros((N_bins, N_rad))
    stacked_data_gas = np.zeros((N_bins, N_rad))
    upper_data_gas = np.zeros((N_bins, N_rad))
    lower_data_gas = np.zeros((N_bins, N_rad))
    
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        for j in range(N_rad):
            stacked_data_DM[i,j] = np.nanmedian(flm.DM_density_3D[not_nan,:][bin_mask,j])
            upper_data_DM[i,j] = np.nanpercentile(flm.DM_density_3D[not_nan,:][bin_mask,j], 75)
            lower_data_DM[i,j] = np.nanpercentile(flm.DM_density_3D[not_nan,:][bin_mask,j], 25)
            stacked_data_gas[i,j] = np.nanmedian(flm.gas_density_3D[not_nan,:][bin_mask,j])
            upper_data_gas[i,j] = np.nanpercentile(flm.gas_density_3D[not_nan,:][bin_mask,j], 75)
            lower_data_gas[i,j] = np.nanpercentile(flm.gas_density_3D[not_nan,:][bin_mask,j], 25)
            
    stacked_DM = np.dstack((lower_data_DM, stacked_data_DM, upper_data_DM))
    stacked_gas = np.dstack((lower_data_gas, stacked_data_gas, upper_data_gas))
    
    log_DM = np.zeros((N_bins, N_rad, 3))
    log_gas = np.zeros((N_bins, N_rad, 3))
    R_DM = np.zeros((3,N_bins))
    R_gas = np.zeros((3,N_bins))
    depth_DM = np.zeros((3,N_bins))
    depth_gas = np.zeros((3,N_bins))
    for i in range(3):
        log_DM[:,:,i] = sp.log_gradients(flm.rad_mid, stacked_DM[:,:,i])
        log_gas[:,:,i] = sp.log_gradients(flm.rad_mid, stacked_gas[:,:,i])
        R_DM[i,:], depth_DM[i,:] = dr.depth_cut(flm.rad_mid, log_DM[:,:,i], depth_value="y", cut=-1)  
        R_gas[i,:], depth_gas[i,:] = dr.depth_cut(flm.rad_mid, log_gas[:,:,i], depth_value="y", cut=-1)  

    R_error_bars_DM = np.sort(np.vstack((R_DM[0,:], R_DM[2,:])), axis=0) - R_DM[1,:]
    R_error_bars_DM[0,:] = -R_error_bars_DM[0,:]
    R_range_gas = np.vstack((R_gas[0,:], R_gas[2,:]))
    R_error_bars_gas = np.sort(R_range_gas, axis=0) - R_gas[1,:]
    R_error_bars_gas[0,:] = -R_error_bars_gas[0,:]
    
    y_error_bars_DM = np.vstack((depth_DM[1,:]-depth_DM[0,:], 
                                 depth_DM[2,:]-depth_DM[1,:]))
    y_error_bars_gas = np.vstack((depth_gas[1,:]-depth_gas[0,:], 
                                 depth_gas[2,:]-depth_gas[1,:]))
        
    for i in range(N_bins):
        plt.figure()
        plt.semilogx(rad_mid, log_DM[i,:,1], color="k")
        plt.fill_between(rad_mid, log_DM[i,:,0], log_DM[i,:,2])
        plt.title(i)
        plt.ylim((-5, 0.5))
        plt.show()
        
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.errorbar(stack_mids, flm.R_DM_sphericity_gas,
                   yerr=R_error_bars_DM,
                   marker="o", color="darkorchid", capsize=2)
    ax.errorbar(stack_mids, flm.R_gas_sphericity_gas,
                yerr=R_error_bars_gas,
                marker="^", color="c", capsize=2)
    plt.xlabel(label)
    plt.ylabel("$R_{\\rm{SP}}$")
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.errorbar(stack_mids, flm.depth_DM_sphericity_gas,
                   yerr=y_error_bars_DM,
                   marker="o", color="darkorchid", capsize=2)
    ax.errorbar(stack_mids, flm.depth_gas_sphericity_gas,
                yerr=y_error_bars_gas,
                marker="^", color="c", capsize=2)
    plt.xlabel(label)
    plt.ylabel("$\gamma_{\\rm{SP}}$")
    plt.show()
    


box = "L1000N1800"
run = "HF"
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2
flm = sp.flamingo(box, run)
flm.sphericity_gas = np.genfromtxt(flm.path + "_sphericity_gas.csv",
                                delimiter=",")
flm.sphericity_DM = np.genfromtxt(flm.path + "_sphericity_DM.csv",
                                delimiter=",")

# plt.hist(flm.sphericity_gas, bins=50)
# plt.show()

# plt.hist(flm.sphericity_DM, bins=50)
# plt.show()

flm.read_properties()
N_bins = 15
mass_bins = np.linspace(14.0, 15, N_bins+1)
mass_bins = np.append(mass_bins, 16)
accretion_bins = np.linspace(0, 4, N_bins+1)
accretion_bins = np.append(accretion_bins, 20)
energy_bins = np.linspace(0.1, 0.3, N_bins+1)
sgas_bins = np.linspace(0.5,0.95, N_bins+1)
sgas_bins = np.append(0, sgas_bins)
sDM_bins = np.linspace(0.3, 0.85, N_bins+1)
sDM_bins = np.append(0, sDM_bins)

sgas_mid = np.zeros(N_bins+1)
sgas_mid[0] = sgas_bins[1]
sgas_mid[1:] = (sgas_bins[2:] + sgas_bins[1:-1]) / 2
sDM_mid = np.zeros(N_bins+1)
sDM_mid[0] = sDM_bins[1]
sDM_mid[1:] = (sDM_bins[2:] + sDM_bins[1:-1]) / 2
acc_mid = np.zeros(N_bins+1)
acc_mid[-1] = accretion_bins[-2]
acc_mid[:-1] = (accretion_bins[1:-1] + accretion_bins[:-2]) / 2
  
bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
sp.stack_and_find_3D(flm, "sphericity_gas", sgas_bins)
sp.stack_and_find_3D(flm, "sphericity_DM", sDM_bins)

plot_profiles_compare_bins(flm, accretion_bins, sgas_bins, sDM_bins, 
                            quantity="DM")
plot_profiles_compare_bins(flm, accretion_bins, sgas_bins, sDM_bins, 
                            quantity="gas")

# plot_correlations(sgas_mid, sDM_mid, acc_mid)
# plot_correlations(sgas_mid, sDM_mid, acc_mid, quantity="depth")
correlations_quartiles(flm.sphericity_gas, sgas_bins, sgas_mid, "$S_{\\rm{gas}}$")
correlations_quartiles(flm.sphericity_DM, sDM_bins, sDM_mid, "$S_{\\rm{DM}}$")
# correlations_quartiles(flm.accretion, accretion_bins, acc_mid, "$\Gamma$")

# plt.figure()
# plt.semilogx(rad_mid, flm.sphericity_DM_log_DM[12,:])
# plt.semilogx(rad_mid, flm.sphericity_DM_log_DM[13,:])
# plt.semilogx(rad_mid, flm.sphericity_DM_log_DM[14,:])
# plt.show()

# plt.figure()
# plt.loglog(rad_mid, flm.sphericity_DM_density_DM[12,:]/rho_crit)
# plt.loglog(rad_mid, flm.sphericity_DM_density_DM[13,:]/rho_crit)
# plt.loglog(rad_mid, flm.sphericity_DM_density_DM[14,:]/rho_crit)
# plt.show()

# for i in range(N_bins):
#     plt.loglog(rad_mid, flm.sphericity_DM_density_DM[i,:]/rho_crit)
# plt.show()