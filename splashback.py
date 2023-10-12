"""Read main data and do most common calculations."""

import numpy as np
import determine_radius as dr
from scipy.signal import savgol_filter

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

identifiers = {
    "HF": "L1_m9",
    "HWA": "fgas+2$\sigma$",
    "HSA": "fgas-2$\sigma$",
    "HTA": "fgas-4$\sigma$",
    "HUA": "fgas-8$\sigma$",
    "HP": "Planck",
    "HPV": "PlanckNu0p24Var",
    "HPF": "PlanckNu0p24Fix",
    "HJ": "Jet",
    "HSJ": "Jet_fgas-4$\sigma$",
    "HSS": "M*-1$\sigma$",
    "DMO": "L1_m9_DMO"} #assumes 1Gpc box

class flamingo:
    def __init__(self, box, run):
        """Read in key 3D profiles and define radii used to produce profiles."""
        #3D profiles
        self.box = box
        self.run = run
        self.run_label = identifiers[self.run]
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
        
    def read_temperature(self):
        self.gas_temperature_3D = np.genfromtxt(self.path + "_3D_gas_temperature_all.csv", 
                                                delimiter=",")
        
    def read_velocity(self):
        self.gas_velocity_3D = -1*np.genfromtxt(self.path + "_3D_gas_velocity_all.csv", 
                                             delimiter=",")
        
    def read_2D(self):
        self.EM_median = np.genfromtxt(self.path + "_EM_profiles_all.csv",
                                       delimiter=",")
        self.SZ_median = np.genfromtxt(self.path + "_SZ_profiles_all.csv",
                                       delimiter=",")
        self.WL_median = np.genfromtxt(self.path + "_WL_profiles_all.csv",
                                       delimiter=",")
        
    def read_2D_properties(self):
        """Reads in 2D morphology criteria obtained from projected emission
        maps of flamingo clusters."""
        morph = np.genfromtxt(self.path + "_morphology_criteria_all.csv",
                              delimiter=",")
        self.concentration = morph[:,0]
        self.symmetry = morph[:,1]
        self.alignment = morph[:,2]
        self.centroid = np.log10(morph[:,3]) #logged to give more even distribution
        
    def read_properties(self):
        """Get general halo properties."""
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
        """Reads low mass data when needed. Adds it to already
        existing variables."""
        DM_density_3D_low = np.genfromtxt(self.path + "_3D_DM_density_low_mass_all.csv", 
                                          delimiter=",") #1e10Msol / (r/R200m)^3
        if self.run != "DMO":
            gas_density_3D_low = np.genfromtxt(self.path + "_3D_gas_density_low_mass_all.csv", 
                                               delimiter=",")
            self.gas_density_3D = np.vstack((self.gas_density_3D, gas_density_3D_low))
            
            gas_properties = np.genfromtxt(self.path + "_gas_properties_low_mass_all.csv",
                                            delimiter=",")
            energy = gas_properties[:,2]
            hot_gas_fraction = gas_properties[:,1]
            baryon_fraction = gas_properties[:,0]
            self.energy = np.hstack((self.energy, energy))
            self.hot_gas_fraction = np.hstack((self.hot_gas_fraction, hot_gas_fraction))
            self.baryon_fraction = np.hstack((self.baryon_fraction, baryon_fraction))
        M200m_low = np.genfromtxt(self.path + "_M200m_low_mass.csv",
                                  delimiter=",")
        accretion_low = np.genfromtxt(self.path + "_accretion_low_mass.csv",
                                      delimiter=",")
        self.DM_density_3D = np.vstack((self.DM_density_3D, DM_density_3D_low))
        self.M200m = np.hstack((self.M200m, M200m_low))
        self.accretion = np.hstack((self.accretion, accretion_low))
       
        
    def read_magnitude_gap(self, twodim=False):
        magnitudes = np.genfromtxt(self.path + "_galaxy_magnitudes.csv", delimiter=",")
        sorted_magnitudes = np.sort(magnitudes)
        mag_bcg = sorted_magnitudes[:,0]
        mag_fourth = sorted_magnitudes[:,2]
        self.gap = mag_fourth - mag_bcg
        if twodim:
            self.gap = np.hstack((self.gap, self.gap, self.gap))
        

def stack_fixed(data, split, split_bins, dim="3D", bootstrap=False,
                print_data=False):
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
    bootstrap : boolean,
        Whether or not to calculate bootstrap errors of Rsp values.
    print_data : boolean
        Whether or not to print information about the number of halos in each
        bin.

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
        temperature = hasattr(data, 'gas_temperature_3D')
        velocity = hasattr(data, 'gas_velocity_3D')
        
        stacking_data = np.dstack((data.DM_density_3D, data.gas_density_3D))
        N_profiles = 2
        saving_strings = ["_DM", "_gas"]
        if pressure:
            stacking_data = np.dstack((stacking_data,
                                       data.gas_pressure_3D))
            N_profiles += 1
            saving_strings.append("_P")
        if entropy :
            stacking_data = np.dstack((stacking_data,
                                       data.gas_entropy_3D))
            N_profiles += 1
            saving_strings.append("_K")
        if temperature:
            stacking_data = np.dstack((stacking_data,
                                       data.gas_temperature_3D))
            N_profiles += 1
            saving_strings.append("_T")
        if velocity:
            stacking_data = np.dstack((stacking_data,
                                       data.gas_velocity_3D))
            N_profiles += 1
            saving_strings.append("_v")
            
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
    if print_data:
        print("")
        print(names[data.run])
        print(split)
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        if print_data:
            print(len(bin_mask))
        for j in range(N_profiles):
            stacked_data[i,:,j] = stack_data(stacking_data[not_nan,:,j][bin_mask,:])
            
    for i in range(N_profiles):
        log = log_gradients(data.rad_mid, stacked_data[:,:,i])
        setattr(data, split+ "_profile" + saving_strings[i], stacked_data[:,:,i]) #previously density instead of profile
        setattr(data, split+ "_log" + saving_strings[i], log)
    
    if bootstrap:
        Rsp_error, gamma_error = bootstrap_errors(data, stacking_data, split, split_data, split_bins)
        for i in range(N_profiles):
            setattr(data, "error_R" + saving_strings[i] + "_" + split , Rsp_error[i,:])
            setattr(data, "error_depth" + saving_strings[i] + "_" + split , gamma_error[i,:])
    

def stack_and_find_3D(data, split, split_bins, bootstrap=False,
                      print_data=False):
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
    bootstrap : bool
        Whether or not to calculate bootstrap errors of Rsp.
    """
    pressure = hasattr(data, 'gas_pressure_3D')
    entropy = hasattr(data, 'gas_entropy_3D')
    velocity = hasattr(data, 'gas_velocity_3D')
    temperature = hasattr(data, 'gas_temperature_3D')
    
    stack_fixed(data, split, split_bins, 
                bootstrap=bootstrap, 
                print_data=print_data)
    log_DM = getattr(data, split+"_log_DM")
    log_gas = getattr(data, split+"_log_gas")
    R_SP_DM, second_DM, depth_DM, depth_second = dr.depth_cut(data.rad_mid, 
                                                              log_DM, 
                                                              cut=-2.5,
                                                              depth_value="y",
                                                              second_caustic="y")
    R_SP_gas, depth_gas = dr.depth_cut(data.rad_mid, log_gas, cut=-2, depth_value="y")
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
    
    if velocity:
        log_v = getattr(data, split+"_log_v")
        R_SP_v = dr.find_maxima(data.rad_mid, 
                                log_v, 
                                cut=0.5)
        setattr(data, "R_v_"+split, R_SP_v)
        
    if temperature:
        log_T = getattr(data, split+"_log_T")
        R_SP_T = dr.depth_cut(data.rad_mid, 
                              log_T, 
                              cut=0.5)
        setattr(data, "R_T_"+split, R_SP_T)
    
    setattr(data, "R_DM_"+split, R_SP_DM)
    setattr(data, "2_DM_"+split, second_DM)
    setattr(data, "R_gas_"+split, R_SP_gas)
    setattr(data, "depth_DM_"+split, depth_DM)
    setattr(data, "depth_gas_"+split, depth_gas)
    

def stack_and_find_2D(data, split, split_bins, 
                      bootstrap=False,
                      print_data=False):
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
    stack_fixed(data, split, split_bins, dim="2D", 
                bootstrap=bootstrap, print_data=print_data)
    log_EM = getattr(data, split+"_log_EM")
    log_SZ = getattr(data, split+"_log_SZ")
    log_WL = getattr(data, split+"_log_WL")
    

    R_SP_EM, second_EM, depth_EM, _ = dr.depth_cut(data.rad_mid, 
                                                   log_EM, 
                                                   cut=-4,
                                                   depth_value="y",
                                                   second_caustic="y")
    setattr(data, "R_EM_"+split, R_SP_EM)
    setattr(data, "depth_EM_"+split, depth_EM)
    setattr(data, "second_EM_"+split, second_EM)
    
    R_SP_SZ, second_SZ, depth_SZ, _ = dr.depth_cut(data.rad_mid, 
                                                   log_SZ, 
                                                   cut=-2.5,
                                                   depth_value="y",
                                                   second_caustic="y")
    setattr(data, "R_SZ_"+split, R_SP_SZ)
    setattr(data, "depth_SZ_"+split, depth_SZ)
    setattr(data, "second_SZ_"+split, second_SZ)
    
    R_SP_WL, second_WL, depth_WL, _ = dr.depth_cut(data.rad_mid, 
                                                   log_WL, 
                                                   cut=-1,
                                                   depth_value="y",
                                                   second_caustic="y")
    setattr(data, "R_WL_"+split, R_SP_WL)
    setattr(data, "depth_WL_"+split, depth_WL)
    setattr(data, "second_WL_"+split, second_WL)
    
    
def second_caustic(Rsp, second):
    """
    Tests if quantity is a splashback or second caustic and replaces if necessary.

    Parameters
    ----------
    data : obj,
        Simulation data object.
    split : str
        Name of quantity used for stacking the profiles.

    Returns
    -------
    Rsp : numpy array
        Array of splashback radii values
    second : numpy array
        Array of second caustic radii values
    """
    second_mask = np.where(np.isfinite(second))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if Rsp[index] < second[index]:
            larger = second[index]
            smaller = Rsp[index]
            Rsp[index] = larger
            second[index] = smaller
    return Rsp, second
    

def bootstrap_errors(data, stacking_data, split, split_data, split_bins, 
                     dim="3D"):
    """
    Calculates errors on radius and depth of minima due to sampling bias.

    Parameters
    ----------
    data : obj
        Simulation object containing data
    stacking_data : numpy array
        Profiles to be sampled and then stacked.
    split : str
        Name of criteria used for the stacking.
    split_data : array
        Stacking criteria data.
    split_bins : array
        Edges of bins used for stacking.
    dim : str, optional
        Dimension of profiles, 2D or 3D. The default is "3D".

    Returns
    -------
    Rsp_error : array, float
        Sampling error on radius of minima.
    depth_error : array, float
        Sampling error on depth of minima.

    """
    if stacking_data.ndim != 3:
        stacking_data = stacking_data[:,:,np.newaxis]
    N_bootstrap = 250
    N_profiles = stacking_data.shape[2]
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    Rsp_error = np.zeros((N_profiles, N_bins))
    depth_error = np.zeros((N_profiles, N_bins))
    
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        stacked_data = np.zeros((N_bootstrap, data.N_rad, N_profiles))
        log_sample = np.zeros((N_bootstrap, data.N_rad, N_profiles))
        for j in range(N_bootstrap):
            # Select random sample from bin with replacement
            sample = np.random.choice(bin_mask, 
                                      size=len(bin_mask),
                                      replace=True)
            for k in range(N_profiles):
                stacked_data[j,:,k] = stack_data(stacking_data[not_nan[sample],:,k])
        for k in range(N_profiles):
            log_sample = log_gradients(data.rad_mid, stacked_data[:,:,k])
            Rsp_sample, second, gamma, _ = dr.depth_cut(data.rad_mid, 
                                                        log_sample, 
                                                        cut=-1,
                                                        second_caustic="y",
                                                        depth_value="y")
            if split == "accretion":
                Rsp_sample, _ = second_caustic(Rsp_sample, second)
            Rsp_error[k,i] = np.nanstd(Rsp_sample)
            depth_error[k,i] = np.nanstd(gamma)
    return Rsp_error, depth_error


def stack_data(array):
    """
    Stacks data. Calculates the median value in each radial bin.

    Parameters
    ----------
    array : 2D numpy array
    Each row a cluster, each column a rad bin. Data to stack.

    Returns
    -------
    profile : 1D numpy array
    Same number of rad bins as above. Stacked data.

    """
    N_bins = np.shape(array)[1]
    profile = np.zeros(N_bins)
    for i in range(N_bins):
        profile[i] = np.nanmedian(array[:,i])
    return profile


def log_gradients(radii, array, window=19, order=4, smooth=True):
    """
    Takes profile, calculates the gradient and then smoothes using a 
    Savitzky-Golay filter.

    Parameters
    ----------
    radii : 1D numpy array 
        Radii corresponsing to array. (N_rad)
    array : 1 or 2D numpy array, 
        Profile(s) to convert.(NxN_rad)
    window : int, optional
        window size for Sav-Gol filter. The default is 19.
    order : int, optional
        Polynomial order for Sav-Gol filter. The default is 4.
    smooth : boolean
        Whether or not to smooth the gradient profile.

    Returns
    -------
    smoothed_array : 1 or 2D numpy array 
        Smoothed, gradient profiles. (NxN_rad)

    """
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
        
        try:
            dlnrho_dlnr = np.gradient(np.log10(density), np.log10(temp_radii))
            if smooth:
                smoothed_array[i,finite] = savgol_filter(dlnrho_dlnr, window, order)
            else:
                smoothed_array[i,finite] = dlnrho_dlnr
        except IndexError: #all nan values in profile
            smoothed_array[i,:] = np.nan
    if N == 1:
        smoothed_array = smoothed_array.flatten()
        
    return smoothed_array