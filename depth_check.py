import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

def read_depth_data(data, depth):
    EM = np.genfromtxt(data.path +
                       "_EM_profiles_" + depth + "r200m_all.csv",
                       delimiter=",")
    SZ = np.genfromtxt(data.path +
                       "_SZ_profiles_" + depth + "r200m_all.csv",
                       delimiter=",")
    WL = np.genfromtxt(data.path + 
                       "_WL_profiles_" + depth + "r200m_all.csv",
                       delimiter=",")
    setattr(data, "EM_median_"+depth, EM)
    setattr(data, "SZ_median_"+depth, SZ)
    setattr(data, "WL_median_"+depth, WL)

def stack_fixed_bins(data, split, split_bins, depth=""):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj.
    
    Inputs.
    data: obj, run of choice
    split: str, name of stacking criteria
    split_bins: bins to use to split stacking criteria.
    """
    if depth != "":
        depth = "_" + depth
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins)+1
    
    stacked_EM = np.zeros((N_bins, N_rad))
    stacked_SZ = np.zeros((N_bins, N_rad))
    stacked_WL = np.zeros((N_bins, N_rad))
    
    print("")
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i)[0]
        stacked_EM[i,:] = sp.stack_data(getattr(data,"EM_median"+depth)[not_nan][bin_mask,:])
        stacked_SZ[i,:] = sp.stack_data(getattr(data,"SZ_median"+depth)[not_nan][bin_mask,:])
        stacked_WL[i,:] = sp.stack_data(getattr(data,"WL_median"+depth)[not_nan][bin_mask,:])
        print(len(bin_mask))
        
    log_EM = sp.log_gradients(rad_mid, stacked_EM)
    R_SP_EM, depth_EM = dr.depth_cut(rad_mid, log_EM, 
                                     depth_value="y",
                                     cut=-5)
    log_SZ = sp.log_gradients(rad_mid, stacked_SZ)
    R_SP_SZ, depth_SZ = dr.depth_cut(rad_mid, log_SZ, 
                                     depth_value="y",
                                     cut=-1)
    log_WL = sp.log_gradients(rad_mid, stacked_WL)
    R_SP_WL, depth_WL = dr.depth_cut(rad_mid, log_WL, 
                                     depth_value="y",
                                     cut=-1)
    
    
    setattr(data, "R_EM_"+split+depth, R_SP_EM)
    setattr(data, split+"_log_EM"+depth, log_EM)
    setattr(data, "depth_EM_"+split+depth, depth_EM)
    setattr(data, "R_SZ_"+split+depth, R_SP_SZ)
    setattr(data, split+"_log_SZ"+depth, log_SZ)
    setattr(data, "depth_SZ_"+split+depth, depth_SZ)
    setattr(data, "R_WL_"+split+depth, R_SP_WL)
    setattr(data, split+"_log_WL"+depth, log_WL)
    setattr(data, "depth_WL_"+split+depth, depth_WL)

def compare_depths(data, stack_criteria, stack_bins, depths):
    N_depths = len(depths)
    mid_N = int(N_depths/2)
    d_labels = np.empty(N_depths, dtype=np.dtype('U100'))
    for i in range(N_depths):
        stack_fixed_bins(flm, stack_criteria, stack_bins, depth=depths[i])
        if depths[i] != "":
            a = "_" + depths[i]
            d_labels[i] = a
        else:
            d_labels[i] = depths[i]
            depths[i] = "$\infty$"
    depths[1] = '5'
    fig, ax = plt.subplots(nrows=N_depths, ncols=3,
                           sharey=True,
                           figsize=(5,5),
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    cm_EM = plt.cm.winter(np.linspace(0,1,N_bins+2))
    cm_SZ = plt.cm.copper(np.linspace(0,1,N_bins+2))
    cm_WL = plt.cm.autumn(np.linspace(0,1,N_bins+2))
    for j in range(N_depths):
        for i in range(N_bins+2):
            ax[j,0].semilogx(rad_mid, getattr(flm, stack_criteria + "_log_EM"+d_labels[j])[i,:],
                              color=cm_EM[i])
            ax[j,1].semilogx(rad_mid, getattr(flm, stack_criteria + "_log_SZ"+d_labels[j])[i,:],
                              color=cm_SZ[i])
            ax[j,2].semilogx(rad_mid, getattr(flm, stack_criteria + "_log_WL"+d_labels[j])[i,:],
                              color=cm_WL[i])
        label = r"Depth = {} $R_{{\rm{{200m}}}}$".format(depths[j])
        ax[j,0].text(0.05,0.93, label,
                     transform=ax[j,0].transAxes)
    ax[-1,1].set_xlabel("$r/R_{\\rm{200m}}$")
    ax[mid_N,0].set_ylabel("$d \log \Sigma / d \log r$")
    ax[0,0].text(0.05, 0.05, "$\Sigma = $EM", transform=ax[0,0].transAxes)
    ax[0,1].text(0.05, 0.05, "$\Sigma = $SZ", transform=ax[0,1].transAxes)
    ax[0,2].text(0.05, 0.05, "$\Sigma = $WL", transform=ax[0,2].transAxes)
    plt.show()

plt.style.use("mnras.mplstyle")
box = "L1000N1800"
run = "HF"

N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins = 3
mass_bins = np.linspace(14.1, 15, N_bins+1)
accretion_bins = np.linspace(1, 4, N_bins+1)
energy_bins = np.linspace(0.1, 0.3, N_bins+1)

flm = sp.flamingo(box, run)
flm.read_properties()
flm.read_2D()

read_depth_data(flm, "10")
read_depth_data(flm, "2")
depths = np.array(["", "10", "2"], dtype=np.dtype('U100'))
compare_depths(flm, "mass", mass_bins, depths)
