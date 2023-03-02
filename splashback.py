"""Read main data and do most common calculations. E.g. finding splashback
radius and stacking data"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from math import factorial
import determine_radius as dr

PATH = "splashback_data/"

#Define radii array 
N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins2 = 45
log_radii2 = np.linspace(-1, 0.7, N_bins2+1)
rad_mid2 = (10**log_radii2[1:] + 10**log_radii2[:-1]) / 2


        
class flamingo:
    """Read flamingo data"""
    def __init__(self, box, run):
        
        # self.accretion = np.genfromtxt(PATH + "macsis/accretion_rates.csv", 
        #                                delimiter=",")
        # self.M_200m = np.genfromtxt(PATH + "macsis/M_200m_macsis.csv", 
        #                             delimiter=",")
        # self.substructure = np.genfromtxt(PATH  + "macsis/substructure_fractions.csv",
        #                                   delimiter=",")
        
        #3D profiles
        self.box = box
        self.run = run
        self.path = PATH + "flamingo/" + box + "_" + run
        
        self.DM_density_3D = np.genfromtxt(self.path + "_3D_DM_density_all.csv", 
                                           delimiter=",") #1e10Msol / (r/R200m)^3
        self.gas_density_3D = np.genfromtxt(self.path + "_3D_gas_density_all.csv", 
                                           delimiter=",")
        self.gas_pressure_3D = np.genfromtxt(self.path + "_3D_gas_pressure_all.csv", 
                                           delimiter=",")
        self.gas_entropy_3D = np.genfromtxt(self.path + "_3D_gas_entropy_all.csv", 
                                           delimiter=",")
        
    def read_2D(self):
        self.EM_median = np.genfromtxt(self.path + "_EM_profiles_all.csv",
                                       delimiter=",")
        self.SZ_median = np.genfromtxt(self.path + "_SZ_profiles_all.csv",
                                       delimiter=",")
        self.WL_median = np.genfromtxt(self.path + "_WL_profiles_all.csv",
                                       delimiter=",")
        
    def read_properties(self):
        accretion = np.genfromtxt(self.path 
                                  + "_accretion.csv",
                                  delimiter=",")
        accretion[np.isinf(accretion)] = np.nan
        self.accretion = accretion
        gas_properties = np.genfromtxt(self.path + "_gas_properties_all.csv",
                                       delimiter=",")
        
        self.energy = gas_properties[:,2]
        self.hot_gas_fraction = gas_properties[:,1]
        self.baryon_fraction = gas_properties[:,0]
        self.M200m = np.genfromtxt(self.path
                                   + "_M200m.csv",
                                   delimiter=",")

class macsis:
    """Read macsis data"""
    def __init__(self):
        
        self.accretion = np.genfromtxt(PATH + "macsis/accretion_rates.csv", 
                                       delimiter=",")
        self.M_200m = np.genfromtxt(PATH + "macsis/M_200m_macsis.csv", 
                                    delimiter=",")
        self.substructure = np.genfromtxt(PATH  + "macsis/substructure_fractions.csv",
                                          delimiter=",")
        
        #3D profiles
        self.DM_density_3D = np.genfromtxt(PATH + 
                                           "macsis/log_density_profiles_DM.csv", 
                                           delimiter=",")
        self.gas_density_3D = np.genfromtxt(PATH + 
                                            "macsis/log_density_profiles_gas_z0.csv", 
                                            delimiter=",")
        self.star_density_3D = np.genfromtxt(PATH + 
                                             "macsis/log_density_profiles_stars.csv", 
                                             delimiter=",")
        
        
        #Read observable files
        self.DM_median, self.DM_mean = read_obs_data("DM", "macsis")
        self.EM_median, self.EM_mean = read_obs_data("EM", "macsis")
        self.SD_median, self.SD_mean = read_obs_data("stellar", "macsis")
        self.SZ_median, self.SZ_mean = read_obs_data("SZ", "macsis")
        
        N_bins = 40
        log_radii = np.linspace(-1, 0.6, N_bins+1)
        self.rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2
        
        self.exclude = np.array([306, 307, 325, 326, 327, 331, 336, 345, 357, 360, 369, 370, 374])
        #need to manually replace these values with np.nan when needed
        
    def read_z05(self):
        """Gives 3D profile data for macsis simulation at z=0.46"""
        #data is taken at z=0.46 but named 05 for consistency with CEAGLE and CELR-B
        self.accretion_z05 = np.genfromtxt(PATH + 
                                           "/accretion_rates_z046.csv", 
                                           delimiter=",")
        
        self.DM_density_z05 = np.genfromtxt(PATH + 
                                            "/log_density_profiles_DM_z046.csv", 
                                            delimiter=",")
        self.gas_density_z05 = np.genfromtxt(PATH + 
                                             "/log_density_profiles_gas_z046.csv", 
                                             delimiter=",")
        self.star_density_z05 = np.genfromtxt(PATH + 
                                              "/log_density_profiles_stars_z046.csv", 
                                              delimiter=",")
        
        self.M_200m_z05 = np.genfromtxt(PATH + "/M_200m_macsis_z05.csv", 
                                        delimiter=",")
        
        
    def read_z1(self):
        """Gives 2D observable profile data for macsis at z=1""" 
        self.DM_median_z1, self.DM_mean_z1 = read_obs_data("DM", "macsis", z=1)
        self.EM_median_z1, self.EM_mean_z1 = read_obs_data("EM", "macsis", z=1)
        self.SD_median_z1, self.SD_mean_z1 = read_obs_data("stellar", "macsis", z=1)
        self.SZ_median_z1, self.SZ_mean_z1 = read_obs_data("SZ", "macsis", z=1)
        
        
    def read_DMO(self):
        """Reads dark matter only simulation data"""
        path = "splashback_data/macsis/"
        DMO_3D = np.genfromtxt(path + "density_DMO_macsis_3D.csv", delimiter=",")
        self.DMO_3D = log_gradients(rad_mid2, DMO_3D)
        
        DMO_2D_x = np.genfromtxt(path + "projected_profiles/log_DMO_grad_profiles_x_macsis_median_all.csv", 
                                 delimiter=",")
        DMO_2D_y = np.genfromtxt(path + "projected_profiles/log_DMO_grad_profiles_y_macsis_median_all.csv", 
                                 delimiter=",")
        DMO_2D_z = np.genfromtxt(path + "projected_profiles/log_DMO_grad_profiles_z_macsis_median_all.csv", 
                                 delimiter=",")
        self.DMO_2D = np.vstack((DMO_2D_x, DMO_2D_y, DMO_2D_z))
        
    
    def read_semi_DMO(self):
        """Read data from hydro simulations that gives surface density if only including DM"""
        path = "splashback_data/macsis/"
        DM_x = np.genfromtxt(path + "projected_profiles/log_DM_only_grad_profiles_x_macsis_median_all.csv",
                             delimiter=",")
        DM_y = np.genfromtxt(path + "projected_profiles/log_DM_only_grad_profiles_y_macsis_median_all.csv",
                             delimiter=",")
        DM_z = np.genfromtxt(path + "projected_profiles/log_DM_only_grad_profiles_z_macsis_median_all.csv",
                             delimiter=",")
        self.DM_2D_xyz = np.vstack((DM_x, DM_y, DM_z))
        
    
    def read_morphologies(self):
        morph_x = np.genfromtxt("../Analysis_files/morphology_stats_x.csv",
                                delimiter=",")
        morph_y = np.genfromtxt("../Analysis_files/morphology_stats_y.csv",
                                delimiter=",")
        morph_z = np.genfromtxt("../Analysis_files/morphology_stats_z.csv",
                                delimiter=",")
        
        self.concentration = np.hstack((morph_x[:,0], morph_y[:,0], morph_z[:,0]))
        self.symmetry = np.hstack((morph_x[:,1], morph_y[:,1], morph_z[:,1]))
        self.alignment = np.hstack((morph_x[:,2], morph_y[:,2], morph_z[:,2]))
        self.centroid = np.hstack((morph_x[:,3], morph_y[:,3], morph_z[:,3]))
        
        
    def read_high_radius(self):
        path = "splashback_data/macsis/high_r/"
        
        def read_file(obs_type, proj):
            filename = path + obs_type + "_profiles_" + proj + "_macsis_high_r.csv"
            return np.genfromtxt(filename, delimiter=",")
        
        DM_x = read_file("DM", "x")
        DM_y = read_file("DM", "y")
        DM_z = read_file("DM", "z")
        
        self.DM_median_high_r = np.vstack((DM_x, DM_y, DM_z))
        
        EM_x = read_file("EM", "x")
        EM_y = read_file("EM", "y")
        EM_z = read_file("EM", "z")
        
        self.EM_median_high_r = np.vstack((EM_x, EM_y, EM_z))
        
        SD_x = read_file("SD", "x")
        SD_y = read_file("SD", "y")
        SD_z = read_file("SD", "z")
        
        self.SD_median_high_r = np.vstack((SD_x, SD_y, SD_z))
        
        SZ_x = read_file("SZ", "x")
        SZ_y = read_file("SZ", "y")
        SZ_z = read_file("SZ", "z")
        
        self.SZ_median_high_r = np.vstack((SZ_x, SZ_y, SZ_z))
        
        
    def read_gas_profiles(self):
        
        self.DM_density = np.genfromtxt(PATH + "macsis/DM_density_3D_macsis.csv",
                                        delimiter=",")
        self.gas_density = np.genfromtxt(PATH + "macsis/gas_density_3D_macsis.csv",
                                        delimiter=",")
        self.entropy = np.genfromtxt(PATH + "macsis/entropy_profiles_macsis.csv",
                                     delimiter=",")
        self.pressure = np.genfromtxt(PATH + "macsis/pressure_profiles_macsis.csv",
                                     delimiter=",")
        self.temperature = np.genfromtxt(PATH + "macsis/temperature_profiles_macsis.csv",
                                         delimiter=",")
        
    
    def calculate_Rsp_2D(self):   
        N_clusters = 1170 #390 x 3 projections
    
        self.RSP_DM_mean = dr.standard(rad_mid, self.DM_mean)
        self.RSP_EM_mean = dr.standard(rad_mid, self.EM_mean)
        self.RSP_SD_mean = dr.standard(rad_mid, self.SD_mean)
        self.RSP_SZ_mean = dr.standard(rad_mid, self.SZ_mean)
            
        self.RSP_DM_median = dr.standard(rad_mid, self.DM_median)
        self.RSP_EM_median = dr.standard(rad_mid, self.EM_median)
        self.RSP_SZ_median = dr.standard(rad_mid, self.SZ_median)
            
        self.RSP_SD_median = np.zeros(N_clusters)
        for i in range(N_clusters):
            ignore = np.where(self.SD_median[i,:] != -2000)[0]
            
            if len(ignore) > 30: #remove clusters which have lost too many data points to find reasonable results
                self.RSP_SD_median[i] = dr.standard(rad_mid[ignore], self.SD_median[i,ignore])
            else:
                self.RSP_SD_median[i] = -1


    def define_bad_halos(self):
        """Define well and poorly matched Rsp halos in MACSIS"""
            
        #determined by eye initially
        poor_DM = np.array([17,18,24,28,32,33,39,41,51,58,62,66,69,71,73,75,
                            85, 87,92,94,102,104,105,107,108,111,112,114,119,
                            120,129,130,139,140,145,154,157,159,160,164,166,
                            167,169,175,182,188,190,191,200,210,215,218,219,
                            228,240,241,243,247,249,252,260,261,266,269,272,
                            277,283,284,296,298,301,305,310,320,322,327,333,
                            335,339,345,352,355,361,381,383,389])
            
        poor_gas = np.array([1,15,16,18,20,22,24,26,28,34,42,43,47,51,54,
                             66,67,68,71,72,73,76,77,79,82,84,88,89,94,
                             102,105,119,120,130,132,133,134,140,152,159,
                             160,165,168,169,172,173,177,178,184,187,190,
                             191,197,199,201,204,206,210,213,215,225,227,
                             230,235,244,246,248,250,252,253,262,269,270,
                             272,277,283,287,291,294,298,310,314,318,321,
                             336,338,339,345,353,354,357,359,361,363,364,
                             371,377,380,382,383,384,386,387,389])
            
        poor_star = np.array([15,17,18,24,26,28,31,33,51,61,66,73,79,94,
                              102,105,111,112,119,120,121,128,129,130,143,
                              146,147,156,157,159,164,176,182,185,186,187,
                              188,190,194,195,197,198,200,202,204,205,206,
                              207,241,250,252,261,266,269,270,271,272,273,
                              277,279,283,287,296,300,310,313,315,316,320,
                              321,323,325,327,333,335,336,338,339,340,346,
                              347,351,352,359,360,361,365,368,371,372,376,
                              377,383])
            
        good = np.arange(390)
            
        good_DM = np.delete(good, poor_DM)
        good_gas = np.delete(good, poor_gas)
        good_star = np.delete(good, poor_star)
            
        ### Segment good and poor clusters into 4 categories - both poor, 1 poor, both good
        #DM and gas
        self.good_DM_gas, good_DM_i, good_gas_i = np.intersect1d(good_DM, good_gas, 
                                                                 assume_unique=True, 
                                                                 return_indices=True)
        self.poor_DM_gas = np.intersect1d(poor_DM, poor_gas, assume_unique=True)
        self.good_DM_poor_gas = np.delete(good_DM, good_DM_i)
        self.poor_DM_good_gas = np.delete(good_gas, good_gas_i)
            
        #DM stars
        self.good_DM_star, good_DM_i, good_star_i = np.intersect1d(good_DM, good_star, 
                                                                   assume_unique=True, 
                                                                   return_indices=True)
        self.poor_DM_star = np.intersect1d(poor_DM, poor_star, assume_unique=True)
        self.good_DM_poor_star = np.delete(good_DM, good_DM_i)
        self.poor_DM_good_star = np.delete(good_star, good_star_i)
            
        #Stars and gas
        self.good_gas_star, good_gas_i, good_star_i = np.intersect1d(good_gas, good_star, 
                                                                     assume_unique=True, 
                                                                     return_indices=True)
        self.poor_gas_star = np.intersect1d(poor_gas, poor_star, assume_unique=True)
        self.good_gas_poor_star = np.delete(good_gas, good_gas_i)
        self.poor_gas_good_star = np.delete(good_star, good_star_i)

class ceagle: 
    """Read C-EAGLE data"""
    def __init__(self):

        #3D profiles
        self.DM_density = np.genfromtxt(PATH + "/CEAGLE/log_density_profiles_eagle_DM.csv", 
                                           delimiter=",")
        self.gas_density = np.genfromtxt(PATH + "/CEAGLE//log_density_profiles_eagle_gas.csv", 
                                            delimiter=",")
        self.star_density = np.genfromtxt(PATH + "/CEAGLE//log_density_profiles_eagle_star.csv", 
                                             delimiter=",")
        
        #2D observable profiles
        self.DM_median, self.DM_mean = read_obs_data("DM", "ceagle")
        self.EM_median, self.EM_mean = read_obs_data("EM", "ceagle")
        self.SD_median, self.SD_mean = read_obs_data("stellar", "ceagle")
        self.SZ_median, self.SZ_mean = read_obs_data("SZ", "ceagle")
        
        
    def read_z05(self):
        self.DM_density_z05 = np.genfromtxt(PATH + 
                                            "/ceagle/log_density_profiles_eagle_DM_z05.csv", 
                                            delimiter=",")
        self.gas_density_z05 = np.genfromtxt(PATH + 
                                             "/ceagle/log_density_profiles_eagle_gas_z05.csv",
                                             delimiter=",")
        self.star_density_z05 = np.genfromtxt(PATH + 
                                              "/ceagle/log_density_profiles_eagle_star_z05.csv",
                                              delimiter=",")
        

class celrb:
    """Read CELR-B data"""
    def __init__(self):
        self.DM_density = np.genfromtxt(PATH + "/CELR-B/log_density_profiles_celrb_DM.csv", 
                                            delimiter=",")
        self.gas_density = np.genfromtxt(PATH + "/CELR-B/log_density_profiles_celrb_gas.csv", 
                                            delimiter=",")
        self.star_density = np.genfromtxt(PATH + "/CELR-B/log_density_profiles_celrb_star.csv", 
                                              delimiter=",")

    def read_z05(self):
        self.DM_density_z05 = np.genfromtxt(PATH + 
                                            "/CELR-B/log_density_profiles_celrb_DM_z05.csv",
                                            delimiter=",")
        self.gas_density_z05 = np.genfromtxt(PATH + 
                                             "/CELR-B/log_density_profiles_celrb_gas_z05.csv",
                                             delimiter=",")
        self.star_density_z05 = np.genfromtxt(PATH + 
                                              "/CELR-B/log_density_profiles_celrb_star_z05.csv", 
                                              delimiter=",")


def read_obs_data(obs_type, simulation, z=0):
    """Read observable data.
    obs_type: type of observable, DM, EM, stellar or SZ
    simulation: which dataset to use, ceagle or macsis
        
    returns
    log gradient profiles of chosen observable and simulation for xyz, 
    stacked on top of one another"""
    
    N_bins = 40 #?
    
    projection = ["x", "y", "z"]
    median = np.empty((0, N_bins))
    mean = np.empty((0, N_bins))
    
    for i in range(3):
        if z == 0:
            filename_median = "/" + simulation + "/projected_profiles/log_" + \
                            obs_type + "_grad_profiles_" + projection[i] + "_"\
                            + simulation + "_median_all.csv"
            filename_mean = "/" + simulation + "/projected_profiles/log_" + \
                            obs_type + "_grad_profiles_" + projection[i] +  \
                            "_" + simulation + "_mean_all.csv"
        elif z == 1:
            filename_median = "/" + simulation + "_z1/log_" + obs_type + \
                        "_grad_profiles_" + projection[i] + "_" + simulation +\
                        "_median_z1_all.csv"
            filename_mean = "/" + simulation + "_z1/log_" + obs_type + \
                        "_grad_profiles_" + projection[i] + "_" + simulation + \
                        "_mean_z1_all.csv"
                    
        median = np.append(median, np.genfromtxt(PATH + filename_median, delimiter=","), 
                               axis=0)
        mean = np.append(mean, np.genfromtxt(PATH + filename_mean, delimiter=","), 
                           axis=0)
        
    return median, mean


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


def grad_plots(DM, gas, star, RSP, title="none"):
    
    x_interp = 10**np.linspace(np.log10(0.2), np.log10(2.9), 200)
    
    DM_interp = interp1d(rad_mid, DM, kind="cubic")
    gas_interp = interp1d(rad_mid, gas, kind="cubic")
    star_interp = interp1d(rad_mid, star, kind="cubic")
    
    DM_y = DM_interp(x_interp)
    gas_y = gas_interp(x_interp)
    star_y = star_interp(x_interp)
    
    min_DM = np.min(DM_y)
    min_gas = np.min(gas_y)
    min_star = np.min(star_y)
    mins = np.array([min_DM, min_gas, min_star])
        
    max_DM = np.max(DM_y)
    max_gas = np.max(gas_y)
    max_star = np.max(star_y)
    maxes = np.array([max_DM, max_gas, max_star])
        
    y_SP = (np.min(mins)*1.1, np.max(maxes) + 0.1*abs(np.min(mins)))
    #Plot different profiles together
    plt.figure()
    plt.semilogx(x_interp, DM_y, label="DM", color="k")
    plt.semilogx(x_interp, gas_y, label="Gas", color="b")
    plt.semilogx(x_interp, star_y, label="Stars", color="r")
    plt.plot(np.array([RSP[2], RSP[2]]), y_SP, color="r")
    plt.plot(np.array([RSP[1], RSP[1]]), y_SP, color="b", linestyle="--")
    plt.plot(np.array([RSP[0], RSP[0]]), y_SP, color="k", linestyle=":")
    plt.legend()
    if title != "none":
        plt.title(title)
    plt.xlabel("r/$R_{200m}$")
    plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
    #plt.ylim(y_SP)
    plt.ylim((-10,1))
    # plt.savefig("halo_" + str(i) + "_splashback_profiles.png", dpi=300)
    plt.show()
    
    
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

def stack_fixed_width(N_bins, split_data, array, min_per_bin=15):
    """Splits clusters into bins of fixed width. Looks at data used for splitting
    clusters, anything more than 2 std away from the mean is ignored. Uses those
    bounds to make initial bins. If any of these bins have too few clusters, the
    min and max value are changed to the next bin with enough clusters. Bins are
    then recalculated.
    
    Inputs
    N_bins: number of bins to use in stacking
    split_data: criteria used for binning
    array: profiles to stack
    min_per_bin: minimum number of clusters in each bin
    
    Returns
    stacked_profiles: median stacked profiles from each bin
    split_data_mid: mid-points of bins used
    """
    mean = np.nanmean(split_data)
    std = np.nanstd(split_data)
    data_min = mean - 2*std #get rids of tail of distribution
    data_max = mean + 2*std #assumes roughly gaussian
    
    N_rad = array.shape[1]
    ordered_data = np.sort(split_data)
    bin_range = np.linspace(data_min, data_max, N_bins+1) #initial range for bins
    
    redo_bins = []
    redo = True
    count = 0
    while redo == True:
        ordered_data = ordered_data[np.where((ordered_data >= bin_range[0]) &
                                             (ordered_data <= bin_range[-1]))[0]]
        count += 1
        redo = False
        bin_range = np.delete(bin_range, redo_bins) #check deletion for end bins
        bin_range = np.linspace(bin_range[0], bin_range[-1], N_bins+1) 
        
        if len(np.where((split_data >=bin_range[0]) & 
                        (split_data < bin_range[-1]))[0]) < N_bins * min_per_bin:
            print("ERROR: Too many bins")
            break
        elif count > 30:
            print("ERROR: not fitting, fiddle parameters")
            break
        
        redo_bins = []
        stacked_profiles = np.zeros((N_bins, N_rad))
        
        not_nan = np.where(np.isfinite(split_data)==True)[0]
        bins_sort = np.digitize(split_data[not_nan], bin_range)
        for i in range(N_bins):
            bin_mask = np.where(bins_sort == i+1)[0]
            
            to_stack = array[bin_mask,:]
                
            stacked_profiles[i,:] = stack_data(to_stack)
            
            if len(bin_mask) <= min_per_bin:
                redo = True
                if i <= N_bins/2: #get rid of a start bin
                    redo_bins.append(i)
                elif i >= N_bins/2: #get rid of an end bin
                    redo_bins.append(i+1)
        # print(len(ordered_data))              
    #split_data_mid = (bin_range[1:] + bin_range[:-1]) / 2
    
    return stacked_profiles, bin_range

def bin_fixed_width(N_bins, split_data, min_per_bin=15):
    """Splits clusters into bins of fixed width. Looks at data used for splitting
    clusters, anything more than 2 std away from the mean is ignored. Uses those
    bounds to make initial bins. If any of these bins have too few clusters, the
    min and max value are changed to the next bin with enough clusters. Bins are
    then recalculated.
    
    Inputs
    N_bins: number of bins to use in stacking
    split_data: criteria used for binning
    min_per_bin: minimum number of clusters in each bin
    
    Returns
    stacked_profiles: median stacked profiles from each bin
    split_data_mid: mid-points of bins used
    """
    mean = np.nanmean(split_data)
    std = np.nanstd(split_data)
    data_min = mean - 2*std #get rids of tail of distribution
    data_max = mean + 2*std #assumes roughly gaussian
    
    ordered_data = np.sort(split_data)
    bin_range = np.linspace(data_min, data_max, N_bins+1) #initial range for bins
    
    redo_bins = []
    redo = True
    count = 0
    while redo == True:
        ordered_data = ordered_data[np.where((ordered_data >= bin_range[0]) &
                                             (ordered_data <= bin_range[-1]))[0]]
        count += 1
        redo = False
        bin_range = np.delete(bin_range, redo_bins) #check deletion for end bins
        bin_range = np.linspace(bin_range[0], bin_range[-1], N_bins+1) 
        
        if len(np.where((split_data >=bin_range[0]) & 
                        (split_data < bin_range[-1]))[0]) < N_bins * min_per_bin:
            print("ERROR: Too many bins")
            break
        elif count > 30:
            print("ERROR: not fitting, fiddle parameters")
            break
        
        redo_bins = []
        
        not_nan = np.where(np.isfinite(split_data)==True)[0]
        bins_sort = np.digitize(split_data[not_nan], bin_range)
        for i in range(N_bins):
            bin_mask = np.where(bins_sort == i+1)[0]
            
            if len(bin_mask) <= min_per_bin:
                redo = True
                if i <= N_bins/2: #get rid of a start bin
                    redo_bins.append(i)
                elif i >= N_bins/2: #get rid of an end bin
                    redo_bins.append(i+1)
        # print(len(ordered_data))              
    #split_data_mid = (bin_range[1:] + bin_range[:-1]) / 2
    
    return bin_range


def stack_fixed_width_new(N_bins, split_data, array, min_per_bin=15):
    """Splits clusters into bins of fixed width. Looks at data used for splitting
    clusters, anything more than 2 std away from the mean is ignored. Uses those
    bounds to make initial bins. If any of these bins have too few clusters, the
    min and max value are changed to the next bin with enough clusters. Bins are
    then recalculated.
    
    Inputs
    N_bins: number of bins to use in stacking
    split_data: criteria used for binning
    array: profiles to stack
    min_per_bin: minimum number of clusters in each bin
    
    Returns
    stacked_profiles: median stacked profiles from each bin
    split_data_mid: mid-points of bins used
    """
    median = np.nanmedian(split_data)
    # std = np.nanstd(split_data)
    data_min = np.nanmin(split_data)
    data_max = np.nanmax(split_data)
    split_data = split_data[np.where(np.isfinite(split_data)==True)[0]]
    ordered_data = np.sort(split_data)
    N_rad = array.shape[1]
    
    redo = True
    count = 0
    while redo == True:
        ordered_data = ordered_data[np.where((ordered_data >= data_min) &
                                             (ordered_data <= data_max))[0]]
        count += 1
        redo = False
        bin_range = np.linspace(data_min, data_max, N_bins+1)
        
        if len(np.where((split_data >= data_min) & 
                        (split_data < data_max))[0]) < N_bins * min_per_bin:
            print("ERROR: Too many bins")
            break
        elif count > 200:
            print("ERROR: not fitting, fiddle parameters")
            break
        
        redo_bins_low = []
        redo_bins_high = []
        stacked_profiles = np.zeros((N_bins, N_rad))
        
        sorted_bins = np.digitize(split_data, bin_range)
        for i in range(N_bins):
            bin_mask = np.where(sorted_bins==i+1)[0]
            to_stack = array[bin_mask,:]
            stacked_profiles[i,:] = stack_data(to_stack)
            
            if len(bin_mask) <= min_per_bin:
                redo = True
                if i <= N_bins/2: #get rid of a start bin
                    redo_bins_low.append(i)
                elif i >= N_bins/2: #get rid of an end bin
                    redo_bins_high.append(i+1)
        
        if redo:
            # print()
            diff = abs(ordered_data - median)
            if diff[0] > diff[-1]:
                data_min = ordered_data[1]
                if data_min == ordered_data[0]:
                    data_max = ordered_data[2]
                # print("Data min =" + str(data_min))
            else:
                data_max = ordered_data[-2]
                if data_max == ordered_data[-1]:
                    data_max = ordered_data[-3]
                # print("Data max =" + str(data_max))
            # print(len(ordered_data), data_min, data_max)
            
            # check_diff_low = np.where(np.diff(redo_bins_low) == 1)[0] #check for consecutive numbers
            # check_diff_high = np.where(np.diff(redo_bins_high) == 1)[0]
            # if len(check_diff_low) != 0 and redo_bins_low[0] == 0:
            #     remove_low = redo_bins_low[check_diff_low[-1] + 1]#gives bin index
            #     data_min = bin_range[remove_low]
            # if len(check_diff_high) != 0 and redo_bins_high[-1] == N_bins-1:
            #     remove_high = redo_bins_high[check_diff_high[0]] #gives bin index
            #     data_max = bin_range[remove_high]
                
    print(len(ordered_data), data_min, data_max)
    split_data_mid = (bin_range[1:] + bin_range[:-1]) / 2
    
    return stacked_profiles, split_data_mid