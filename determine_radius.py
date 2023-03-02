import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def standard(radii, array, plot="n"): 
    """Calculating Rsp by interpolating profiles.
    radii   - corresponding radii data for log gradient profiles
    density - array of gradient of log density values for either one cluster
              or a group of clusters but must be size (N_clusters, N_rad_bins)
    plot    - optional plotting of profile to identify minima"""
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    
    rad_range = radii[[9,-9]] #upper and lower bound set by window size
    
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
        if len(density) > 0:
            density_interp = interp1d(temp_radii, density, kind="cubic")
            x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), 200)
            y_interp = density_interp(x_interp)
            
            x_min = np.where(x_interp > rad_range[0])[0][0]
            x_max = np.where(x_interp < rad_range[1])[0][-1]
            
            minima = argrelextrema(y_interp, np.less)[0]
            
                    
            test_SP = np.argmin(y_interp)
                         
            #Look for deep minima in range closest to R200m first to ignore deep minima
            #at high and low radii that could skew it.
            #These numbers are somewhat arbitrary but small changes don't make a
            #difference to results
            #minima_range = np.arange(69,165,1) #165 for high radii, 155 for smaller
            minima_range = np.arange(x_min, x_max, 1)
                    
            #Findsplashback
        
            k = 0
            l = 0
            if len(minima) > 0:
                #where to saev minima values found to be in range
                minima_in_range = np.array([], dtype=int)
                for j in minima:
                    if j in minima_range:
                        test_SP = j
                        minima_in_range = np.append(minima_in_range, j)
                        l += 1
                    else:
                        k += 1
                            
                if l > 1:
                    #if there are more than one minima in range, take the deepest
                    test_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                            
                if k == len(minima):
                    #if there are no minima in range, take the deepest overall
                    test_SP = minima[np.argmin(y_interp[minima])]
            
                if plot != "n":   
                    # Looks at minima in distribution to check Rsp is chosen properly
                    plt.figure()
                    plt.semilogx(x_interp, y_interp, color="k")
                    plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
                    plt.scatter(x_interp[test_SP], y_interp[test_SP], marker="*", 
                                color="gold", s=100)
                    plt.xlabel("r/$R_{200m}$")
                    plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
                    plt.title(plot)
                    plt.show()
                    
                R_sp[i] = x_interp[test_SP]
                
            else:
                R_sp[i] = np.nan
        else:
            R_sp[i] = np.nan
        
    return R_sp

def no_priors(radii, array, plot="n"): 
    """Calculating Rsp by interpolating profiles.
    Same as above but looks for deepest minima without looking in a range first.
    radii   - corresponding radii data for log gradient profiles
    density - array of gradient of log density values for one cluster only
    plot    - optional plotting of profile to identify minima"""
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(radii[0]), np.log10(radii[-2]), 200)
        y_interp = density_interp(x_interp)
                
        minima = argrelextrema(y_interp, np.less)[0]
                
        test_SP = np.argmin(y_interp)
                 
        if plot != "n":   
            # Looks at minima in distribution to check Rsp is chosen properly
            plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
            plt.scatter(x_interp[test_SP], y_interp[test_SP], marker="*", 
                        color="gold", s=100)
            plt.xlabel("r/$R_{200m}$")
            plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            plt.title(plot)
            plt.show()
            
        R_sp[i] = x_interp[test_SP]
        
    return R_sp

def DM_prior(radii, array, RSP_DM, plot="n"): 
    """Calculating Rsp by interpolating profiles, uses splashback radius from
    dark matter profiles to set a prior on gas density profiles.
    radii   - corresponding radii data for log gradient profiles
    density - array of gradient of log density values for either one cluster
              or a group of clusters but must be size (N_clusters, N_rad_bins)
    RSP_DM  - DM splashback radius to use as prior
    plot    - optional plotting of profile to identify minima"""
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
            
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), 200)
        y_interp = density_interp(x_interp)
        
        minima = argrelextrema(y_interp, np.less)[0]
                
        test_SP = np.argmin(y_interp)
                     
        RSP_DM_index = np.where(x_interp > RSP_DM[i])[0][0] 
        #gives first index in x_interp that is larger than RSP_DM
        minima_range = np.arange(RSP_DM_index-6, RSP_DM_index+5, 1) 
                
        #Findsplashback
    
        k = 0
        l = 0
            
        #where to saev minima values found to be in range
        minima_in_range = np.array([], dtype=int)
        for j in minima:
            if j in minima_range:
                test_SP = j
                minima_in_range = np.append(minima_in_range, j)
                l += 1
            else:
                k += 1
                    
        if l > 1:
            #if there are more than one minima in range, take the deepest
            test_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                    
        if k == len(minima):
            #if there are no minima in range, take the deepest overall
            test_SP = minima[np.argmin(y_interp[minima])]
    
        if plot != "n":   
            # Looks at minima in distribution to check Rsp is chosen properly
            plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
            plt.scatter(x_interp[test_SP], y_interp[test_SP], marker="*", 
                        color="gold", s=100)
            plt.xlabel("r/$R_{200m}$")
            plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            plt.title(plot)
            plt.show()
                
        R_sp[i] = x_interp[test_SP]
        
    return R_sp

def DM_prior_new(radii, array, RSP_DM, plot="n"): 
    """Calculating Rsp by interpolating profiles, uses splashback radius from
    dark matter profiles to set a prior on gas density profiles.
    radii   - corresponding radii data for log gradient profiles
    density - array of gradient of log density values for either one cluster
              or a group of clusters but must be size (N_clusters, N_rad_bins)
    RSP_DM  - DM splashback radius to use as prior
    plot    - optional plotting of profile to identify minima"""
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
            
        density_interp = interp1d(temp_radii, density, kind="cubic")
        N_interp = 200
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), N_interp)
        y_interp = density_interp(x_interp)
        
        mc, residuals, rank, sing_vals, rcond = np.polyfit(x_interp, y_interp, 1, full=True)
        width = np.sqrt(residuals/N_interp)
        least_squares_fit = mc[0] * x_interp + mc[1]
        minimum_bound = least_squares_fit - 0.1*width

        minima = argrelextrema(y_interp, np.less)[0]
        minima_depths = y_interp[minima]
        
        depth_cut = np.where(minima_depths <= minimum_bound[minima])[0]
        
        candidates = minima[depth_cut]
        #print("Cluster: " + str(i))
                
        RSP_DM_index = np.abs(x_interp - RSP_DM[i]).argmin()
        close_index = np.abs(candidates - RSP_DM_index).argmin()
        
        R_sp[i] = x_interp[candidates[close_index]]
        
        # R_sp[i] = x_interp[test_SP]
    
        # if plot != "n":   
        #     # Looks at minima in distribution to check Rsp is chosen properly
        #     plt.figure()
        #     plt.semilogx(x_interp, y_interp, color="k")
        #     plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
        #     plt.scatter(x_interp[test_SP], y_interp[test_SP], marker="*", 
        #                 color="gold", s=100)
        #     plt.xlabel("r/$R_{200m}$")
        #     plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
        #     plt.title(plot)
        #     plt.show()
                
    return R_sp

def depth_cut(radii, array, plot="n", cut=-3, depth_value="n"):

    """Calculating Rsp by interpolating profiles.
    radii   - corresponding radii data for log gradient profiles
    density - array of gradient of log density values for either one cluster
              or a group of clusters but must be size (N_clusters, N_rad_bins)
    """
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    depth = np.zeros(N)
        
    rad_range = radii[[9,-9]] #upper and lower bound set by window size
    
    for i in range(N):
        if np.isnan(array[i,:]).all() == True:
            R_sp[i] = np.nan
            depth[i] = np.nan
            continue
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
            
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), 1000)
        y_interp = density_interp(x_interp)
            
        x_min = np.where(x_interp > rad_range[0])[0][0]
        x_max = np.where(x_interp < rad_range[1])[0][-1]
        if x_max > 2.4:
            x_max = np.where(x_interp > 2.4)[0][0]
              
        minima = argrelextrema(y_interp, np.less)[0]
        depth_mask = np.where(y_interp[minima] <= cut)[0]
        minima = minima[depth_mask]
                        
        test_SP = np.argmin(y_interp)
                             
        #Look for deep minima in range closest to R200m first to ignore deep minima
        #at high and low radii that could skew it.
        #These numbers are somewhat arbitrary but small changes don't make a
        #difference to results
        #minima_range = np.arange(69,155,1) #165 for high radii, 155 for smaller
        minima_range = np.arange(x_min, x_max, 1)
                        
        #Findsplashback
        k = 0
        l = 0
        if len(minima) == 0:
            R_sp[i] = np.nan
            depth[i] = np.nan
            continue
        #where to saev minima values found to be in range
        minima_in_range = np.array([], dtype=int)
        for j in minima:
            if j in minima_range:
                test_SP = j
                minima_in_range = np.append(minima_in_range, j)
                l += 1
            else:
                k += 1
                                
        if l > 1:
            #if there are more than one minima in range, take the deepest
            test_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                                
        if k == len(minima):
            #if there are no minima in range, take the deepest overall
            test_SP = minima[np.argmin(y_interp[minima])]
                
        R_sp[i] = x_interp[test_SP]
        depth[i] = y_interp[test_SP]
                    
        if plot != "n":   
            # Looks at minima in distribution to check Rsp is chosen properly
            plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
            plt.scatter(x_interp[test_SP], y_interp[test_SP], marker="*", 
                        color="gold", s=100)
            plt.xlabel("r/$R_{200m}$")
            plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            plt.title(plot)
            plt.show()
    
    if depth_value != "n":
        return R_sp, depth
    return R_sp

def shock_finder(radii, array):
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_shock = np.zeros(N)
    rad_range = radii[[9,-9]] #upper and lower bound set by window size
    
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
        if len(density) == 0:
            R_shock[i] = np.nan
            continue
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), 200)
        y_interp = density_interp(x_interp)
            
        x_min = np.where(x_interp > rad_range[0])[0][0]
        if x_interp[x_min] < 1.6:
            x_min = np.where(x_interp > 1.6)[0][0] #cannot look for shock radii below 1.6
        x_max = np.where(x_interp < rad_range[1])[0][-1]
        minima_range = np.arange(x_min, x_max, 1)
            
        minima = argrelextrema(y_interp, np.less)[0]
            
        test_SP = np.argmin(y_interp)
        #Findsplashback
        k = 0
        l = 0
        if len(minima) == 0:
            R_shock[i] = np.nan
            continue
                
        minima_in_range = np.array([], dtype=int)
        for j in minima:
            if j in minima_range:
                test_SP = j
                minima_in_range = np.append(minima_in_range, j)
                l += 1
            else:
                k += 1
                            
        if l > 1:
            #if there are more than one minima in range, take the deepest
            test_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                            
        if k == len(minima):
            #if there are no minima in range, take the deepest overall
            test_SP = minima[np.argmin(y_interp[minima])]
                              
        R_shock[i] = x_interp[test_SP]
        
    return R_shock

def entropy_prior(radii, gas_density, entropy):
    """
    Uses entropy profile to identify shock radius in the gas and
    excludes this from splashback radius identification to reuce
    the effect of outliers.
    
    Inputs:
    radii - radial range of gas density and entropy profiles (M)
    gas_density - gas density profiles of N clusters (N x M)
    entropy - gas entropy profiles of N clusters (N x M)
    
    Returns:
    R_sp - splashback radii identified in gas density profiles (N)
    """
    
    if gas_density.ndim != 1:
        N = gas_density.shape[0]
    else:
        N = 1
        gas_density = gas_density[np.newaxis,:]
        
    R_shock = shock_finder(radii, entropy) #expand on this to find multiple minima in case of many shock radii
    R_sp = np.zeros(N)
    rad_range = radii[[9,-9]] #upper and lower bound set by window size
    
    for i in range(N):
        density = gas_density[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
        
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), 
                                   np.log10(temp_radii[-2]), 200)
        y_interp = density_interp(x_interp)
        minima = argrelextrema(y_interp, np.less)[0]
            
        x_min = np.where(x_interp > rad_range[0])[0][0]
        #stop range before shock feature begins
        if R_shock[i] > 1.6 and R_shock[i] < rad_range[1]:
            x_max = np.where(x_interp < R_shock[i])[0][-10]
        else:
            x_max = np.where(x_interp < rad_range[1])[0][-1]
        #stay within region without smoothing edge effects
        minima_range = np.arange(x_min, x_max, 1)
        
        #Findsplashback
        k = 0
        l = 0
        test_SP = np.argmin(y_interp)
        if len(minima) == 0:
            R_sp[i] = np.nan
            continue
        minima_in_range = np.array([], dtype=int)
        for j in minima:
            if j in minima_range:
                test_SP = j
                minima_in_range = np.append(minima_in_range, j)
                l += 1
            else:
                k += 1
                            
        if l > 1:
            #if there are more than one minima in range, take the deepest
            test_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                            
        if k == len(minima):
            #if there are no minima in range, take the deepest overall
            test_SP = minima[np.argmin(y_interp[minima])]
            
                    
        R_sp[i] = x_interp[test_SP]
        
    return R_sp
    