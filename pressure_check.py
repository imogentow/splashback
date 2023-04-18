import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr
import stacking_3D as s3d

plt.style.use("mnras.mplstyle")

def stack_fixed_bins(data, split, split_bins):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj.
    
    Inputs.
    data: obj, run of choice
    split: str, name of stacking criteria
    split_bins: bins to use to split stacking criteria.
    """
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins)+1
    
    stacked_pressure = np.zeros((N_bins, N_rad))
    stacked_gas = np.zeros((N_bins, N_rad))
    
    print("")
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i)[0]
        stacked_pressure[i,:] = sp.stack_data(data.gas_pressure_3D[not_nan][bin_mask,:])
        stacked_gas[i,:] = sp.stack_data(data.gas_density_3D[not_nan][bin_mask,:])
        print(len(bin_mask))
        
    log_gas = sp.log_gradients(rad_mid, stacked_gas)
    R_SP_gas, depth_gas = dr.depth_cut(rad_mid, log_gas, 
                                       cut=-1, depth_value="y")
    log_pressure = sp.log_gradients(rad_mid, stacked_pressure)
    R_SP_pressure, depth_pressure = dr.depth_cut(rad_mid, log_pressure, 
                                                 cut=-1, depth_value="y")
    
    setattr(data, "R_pressure_"+split, R_SP_pressure)
    setattr(data, split+"_log_pressure", log_pressure)
    setattr(data, "depth_pressure_"+split, depth_pressure)
    setattr(data, "R_gas_"+split, R_SP_gas)
    setattr(data, split+"_log_gas", log_gas)
    setattr(data, "depth_gas_"+split, depth_gas)


def bin_profiles(d, accretion_bins, mass_bins, energy_bins):
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
    stack_fixed_bins(d, "accretion", accretion_bins)
    stack_fixed_bins(d, "mass", mass_bins)
    stack_fixed_bins(d, "energy", energy_bins)


box = "L1000N1800"
run = "HF"
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2
flm = sp.flamingo(box, run)
flm.read_extra_3D()
flm.read_properties()

N_bins = 3
mass_bins = np.linspace(14.1, 15, N_bins+1)
accretion_bins = np.linspace(1, 4, N_bins+1)
energy_bins = np.linspace(0.1, 0.3, N_bins+1)
    
bin_profiles(flm, accretion_bins, mass_bins, energy_bins)
s3d.plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
                               quantity="pressure")
s3d.plot_profiles_compare_bins(flm, accretion_bins, mass_bins, energy_bins,
                               quantity="gas")

plt.scatter(flm.R_gas_energy, flm.R_pressure_energy)
plt.show()