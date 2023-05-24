import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr
from scipy.stats import spearmanr

def bin_profiles(data, bins):
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
    N_bins = bins.shape[0]
    bin_list_names = ["accretion", "mass", "energy"]
    for i in range(N_bins):
        sp.stack_fixed(data, bin_list_names[i], bins[i,:], dim="3D")
        log_DM = getattr(data, bin_list_names[i]+"_log_DM")

        R_SP_DM, second_DM, depth_DM, second_depth = dr.depth_cut(data.rad_mid, 
                                                                  log_DM, 
                                                                  cut=-1,
                                                                  depth_value="y", 
                                                                  second_caustic="y")
        second_mask = np.where(np.isfinite(second_DM))[0]
        for j in range(len(second_mask)):
            index = second_mask[j]
            if R_SP_DM[index] < second_DM[index]:
                larger = second_DM[index]
                smaller = R_SP_DM[index]
                R_SP_DM[index] = larger
                second_DM[index] = smaller
                depth1 = depth_DM[index]
                depth2 = second_depth[index]
                depth_DM[index] = depth2
                second_depth[index] = depth1
        setattr(data, "R_DM_"+bin_list_names[i], R_SP_DM)
        setattr(data, "second_DM_"+bin_list_names[i], second_DM)
        setattr(data, "depth_DM_"+bin_list_names[i], depth_DM)
    
    
def compare_Rsp_stack(quantity):
    mids = getattr(flm, quantity+"_mid")
    R_sp = getattr(flm, "R_DM_"+quantity)
    correlation = spearmanr(mids, R_sp).correlation
    print(correlation)
    
    plt.scatter(mids, R_sp)
    plt.show()


def stack_for_profiles():
    N_bins = 4
    mass_bins = np.linspace(14, 15, N_bins+1)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins+1)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins+1)
    energy_bins = np.append(energy_bins, 1)
    
    bin_profiles(flm, accretion_bins, mass_bins, energy_bins)


def stack_for_Rsp(N_bins):
    #Fixed width bins only here, no edge bins
    mass_bins = np.linspace(13,
                            np.nanpercentile(np.log10(flm.M200m), 99.9),
                            N_bins+1)
    accretion_bins = np.linspace(0, 
                                 np.nanpercentile(flm.accretion[np.isfinite(flm.accretion)], 99), 
                                 N_bins+1)
    energy_bins = np.linspace(np.min(flm.energy), 
                              np.nanpercentile(flm.energy, 99), 
                              N_bins+1)
    bins = np.vstack((accretion_bins, mass_bins, energy_bins))
    bin_profiles(flm, bins)

    flm.mass_mid = (mass_bins[:-1] + mass_bins[1:])/2
    flm.accretion_mid = (accretion_bins[:-1] + accretion_bins[1:])/2
    flm.energy_mid = (energy_bins[:-1] + energy_bins[1:])/2
    mids = np.vstack((flm.accretion_mid, flm.mass_mid, flm.energy_mid))


box = "L1000N1800"
if __name__ == "__main__":
    flm = sp.flamingo(box, "HF")
    flm.read_properties()
    flm.read_low_mass()
    
    stack_for_Rsp(20)
    compare_Rsp_stack("accretion")
    compare_Rsp_stack("mass")
    compare_Rsp_stack("energy")