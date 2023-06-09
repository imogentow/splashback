"""Read main data and do most common calculations. E.g. finding splashback
radius and stacking data"""

import numpy as np
from math import factorial
import determine_radius as dr

PATH = "splashback_data/"
       
names = {
    "HF": "HYDRO_FIDUCIAL",
    "HWA": "HYDRO_WEAK_AGN",
    "HSA": "HYDRO_STRONG_AGN",
    "HTA": "HYDRO_STRONGER_AGN",
    "HUA": "HYDRO_STRONGEST_AGN",
    "HP": "HYDRO_PLANCK",
    "HPV": "HYDRO_PLANCK_LARGE_NU_VARY",
    "HPF": "HYDRO_PLANCK_LARGE_NU_FIXED",
    "HJ": "HYDRO_JETS",
    "HSJ": "HYDRO_STRONG_JETS",
    "HSS": "HYDRO_STRONG_SUPERNOVA",
    "DMO": "DARK_MATTER_ONLY"} 
class flamingo:
    """Read flamingo data"""
    def __init__(self, box, run):
        #3D profiles
        self.box = box
        self.run = run
        self.run_label = names[self.run]
        self.path = PATH + "flamingo/" + box + "_" + run
        
        self.DM_density_3D = np.genfromtxt(self.path + "_3D_DM_density_all.csv", 
                                           delimiter=",") #1e10Msol / (r/R200m)^3
        if self.run != "DMO":
            self.gas_density_3D = np.genfromtxt(self.path + "_3D_gas_density_all.csv", 
                                               delimiter=",")
        
        self.N_rad = 44
        self.log_radii = np.linspace(-1, 0.7, self.N_rad+1)
        self.rad_mid = (10**self.log_radii[1:] + 10**self.log_radii[:-1]) / 2
    
    def read_pressure(self):
        self.gas_pressure_3D = np.genfromtxt(self.path + "_3D_gas_pressure_all.csv", 
                                           delimiter=",")
        
    def read_entropy(self):
        self.gas_entropy_3D = np.genfromtxt(self.path + "_3D_gas_entropy_all.csv", 
                                           delimiter=",")
        
    def read_2D(self):
        self.EM_median = np.genfromtxt(self.path + "_EM_profiles_all.csv",
                                       delimiter=",")
        self.SZ_median = np.genfromtxt(self.path + "_SZ_profiles_all.csv",
                                       delimiter=",")
        self.WL_median = np.genfromtxt(self.path + "_WL_profiles_all.csv",
                                       delimiter=",")
        
    def read_2D_properties(self):
        morph = np.genfromtxt(self.path + "_morphology_criteria_all.csv",
                              delimiter=",")
        self.concentration = morph[:,0]
        self.symmetry = morph[:,1]
        self.alignment = morph[:,2]
        self.centroid = np.log10(morph[:,3]) #logged to give even distribution
        
    def read_properties(self):
        accretion = np.genfromtxt(self.path 
                                  + "_accretion.csv",
                                  delimiter=",")
        accretion[np.isinf(accretion)] = np.nan
        self.accretion = accretion
        self.M200m = np.genfromtxt(self.path + "_M200m.csv",
                                   delimiter=",")
        if self.run != "DMO":
            gas_properties = np.genfromtxt(self.path + "_gas_properties_all.csv",
                                       delimiter=",")
            self.energy = gas_properties[:,2]
            self.hot_gas_fraction = gas_properties[:,1]
            self.baryon_fraction = gas_properties[:,0]
        
        
    def read_low_mass(self):
        """Read low mass data when needed to save memory"""
        DM_density_3D_low = np.genfromtxt(self.path + "_3D_DM_density_low_mass_all.csv", 
                                          delimiter=",") #1e10Msol / (r/R200m)^3
        if self.run != "DMO":
            gas_density_3D_low = np.genfromtxt(self.path + "_3D_gas_density_low_mass_all.csv", 
                                               delimiter=",")
            self.gas_density_3D = np.vstack((self.gas_density_3D, gas_density_3D_low))
            
        M200m_low = np.genfromtxt(self.path + "_M200m_low_mass.csv",
                                  delimiter=",")
        accretion_low = np.genfromtxt(self.path + "_accretion_low_mass.csv",
                                      delimiter=",")
        self.DM_density_3D = np.vstack((self.DM_density_3D, DM_density_3D_low))
        self.M200m = np.hstack((self.M200m, M200m_low))
        self.accretion = np.hstack((self.accretion, accretion_low))
        

def stack_fixed(data, split, split_bins, dim="3D"):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj describing new profiles.

    Parameters
    ----------
    data : obj
        Run of choice
    split : str
        Name of stacking criteria
    split_bins : numpy array
        Bins to use to split stacking criteria.
    dim : str, optional
        Dimension of choice to tell function which profiles to stack. 3D
        stacks gas and DM, 2D stacks EM, SZ and WL. The default is "3D".

    Returns
    -------
    None.

    """
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)
        
    if dim == "3D":
        pressure = hasattr(data, 'gas_pressure_3D')
        entropy = hasattr(data, 'gas_entropy_3D')
        if pressure and not entropy:
            stacking_data = np.dstack((data.DM_density_3D, 
                                       data.gas_density_3D,
                                       data.gas_pressure_3D))
            N_profiles = 3
            saving_strings = ["_DM", "_gas", "_P"]
        elif entropy and not pressure:
            stacking_data = np.dstack((data.DM_density_3D, 
                                       data.gas_density_3D,
                                       data.gas_entropy_3D))
            N_profiles = 3
            saving_strings = ["_DM", "_gas", "_K"]
        elif pressure and entropy:
            stacking_data = np.dstack((data.DM_density_3D, 
                                       data.gas_density_3D,
                                       data.gas_pressure_3D,
                                       data.gas_entropy_3D))
            N_profiles = 4
            saving_strings = ["_DM", "_gas", "_P", "_K"]
        else:
            stacking_data = np.dstack((data.DM_density_3D, data.gas_density_3D))
            N_profiles = 2
            saving_strings = ["_DM", "_gas"]
        
        if len(split_data) > stacking_data.shape[0]:
            stacking_data = np.vstack((stacking_data, stacking_data,
                                       stacking_data))
            
    elif dim == "2D":
        stacking_data = np.dstack((data.EM_median, data.SZ_median,
                                   data.WL_median))
        N_profiles = 3
        saving_strings = ["_EM", "_SZ", "_WL"]

    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    stacked_data = np.zeros((N_bins, data.N_rad, N_profiles))
    print("")
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        print(len(bin_mask))
        for j in range(N_profiles):
            stacked_data[i,:,j] = stack_data(stacking_data[not_nan,:,j][bin_mask,:])
            
    for i in range(N_profiles):
        log = log_gradients(data.rad_mid, stacked_data[:,:,i])
        setattr(data, split+ "_profile" + saving_strings[i], stacked_data[:,:,i]) #previously density instead of profile
        setattr(data, split+ "_log" + saving_strings[i], log)
    
    

def stack_and_find_3D(data, split, split_bins):
    """
    Stacks data using stack_3D_function. Uses new profiles to determine 
    splashback radius and depth of minima.
    
    Function is separated to allow more manipulation using the two functions.

    Parameters
    ----------
    data : obj
        Relevant dataset, profiles and quantities describing data.
        Properties must also be read in as well as relevant profiles.
    split : str
        Name of stacking criteria.
    split_bins : numpy array
        Bins to use to split stacking criteria.
    """
    pressure = hasattr(data, 'gas_pressure_3D')
    entropy = hasattr(data, 'gas_entropy_3D')
    
    stack_fixed(data, split, split_bins)
    log_DM = getattr(data, split+"_log_DM")
    log_gas = getattr(data, split+"_log_gas")
    R_SP_DM, second_DM, depth_DM, depth_second = dr.depth_cut(data.rad_mid, 
                                                              log_DM, 
                                                              cut=-2.5,
                                                              depth_value="y",
                                                              second_caustic="y")
    R_SP_gas, depth_gas = dr.depth_cut(data.rad_mid, log_gas, cut=-2.5, depth_value="y")
    if pressure:
        log_P = getattr(data, split+"_log_P")
        R_SP_P, second_P, depth_P, depth_second = dr.depth_cut(data.rad_mid, 
                                                              log_P, 
                                                              cut=-2.5,
                                                              depth_value="y",
                                                              second_caustic="y")
        setattr(data, "R_P_"+split, R_SP_P)
        setattr(data, "2_P_"+split, second_P)
        setattr(data, "depth_P_"+split, depth_P)
        
    if entropy:
        log_K = getattr(data, split+"_log_K")
        R_SP_K, second_K, depth_K, depth_second = dr.depth_cut(data.rad_mid, 
                                                              log_K, 
                                                              cut=0.5,
                                                              depth_value="y",
                                                              second_caustic="y")
        setattr(data, "R_K_"+split, R_SP_K)
        setattr(data, "2_K_"+split, second_K)
        setattr(data, "depth_K_"+split, depth_K)
    
    
    setattr(data, "R_DM_"+split, R_SP_DM)
    setattr(data, "2_DM_"+split, second_DM)
    setattr(data, "R_gas_"+split, R_SP_gas)
    setattr(data, "depth_DM_"+split, depth_DM)
    setattr(data, "depth_gas_"+split, depth_gas)
    

def stack_and_find_2D(data, split, split_bins):
    """
    Stacks data using stack_3D_function. Uses new profiles to determine 
    splashback radius and depth of minima.
    
    Function is separated to allow more manipulation using the two functions.

    Parameters
    ----------
    data : obj
        Relevant dataset, profiles and quantities describing data.
        Properties must also be read in as well as relevant profiles.
    split : str
        Name of stacking criteria.
    split_bins : numpy array
        Bins to use to split stacking criteria.

    """
    stack_fixed(data, split, split_bins, dim="2D")
    log_EM = getattr(data, split+"_log_EM")
    log_SZ = getattr(data, split+"_log_SZ")
    log_WL = getattr(data, split+"_log_WL")
    
    R_SP_EM, depth_EM = dr.depth_cut(data.rad_mid, log_EM, 
                                     depth_value="y",
                                     cut=-5)
    R_SP_SZ, depth_SZ = dr.depth_cut(data.rad_mid, log_SZ, 
                                     depth_value="y",
                                     cut=-1)
    R_SP_WL, depth_WL = dr.depth_cut(data.rad_mid, log_WL, 
                                     depth_value="y",
                                     cut=-1)
    
    setattr(data, "R_EM_"+split, R_SP_EM)
    setattr(data, "depth_EM_"+split, depth_EM)
    setattr(data, "R_SZ_"+split, R_SP_SZ)
    setattr(data, "depth_SZ_"+split, depth_SZ)
    setattr(data, "R_WL_"+split, R_SP_WL)
    setattr(data, "depth_WL_"+split, depth_WL)
    

def stack_data(array):
    """Stacks data.
    array - 2D array, each row a cluster, each column a rad bin
    
    returns 
    profile - 1D array, same number of rad bins as above"""
    N_bins = np.shape(array)[1]
    profile = np.zeros(N_bins)
    for i in range(N_bins):
        profile[i] = np.nanmedian(array[:,i])
    return profile

    
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def log_gradients(radii, array, window=19, order=4):
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    smoothed_array = np.zeros((N, len(radii)))
        
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
        dlnrho_dlnr = np.gradient(np.log10(density), np.log10(temp_radii)) 
        smoothed_array[i,finite] = savitzky_golay(dlnrho_dlnr, window_size=window,
                                              order=order)
    if N == 1:
        smoothed_array = smoothed_array.flatten()
        
    return smoothed_array
