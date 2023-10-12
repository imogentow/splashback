import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def depth_cut(radii, array, 
              plot=False, 
              cut=-3, 
              depth_value=False,
              second_caustic=False):
    """
    Calculating Rsp by interpolating profiles.

    Parameters
    ----------
    radii : array
        Radius corresponding to given profiles.
    array : array
        Profile used to find the minima radii.
    plot : bool, optional
        Whether to plot profiles to visualise minima location. 
        The default is False.
    cut : float, optional
        Depth cut of to use to identify minima. The default is -3.
    depth_value : bool, optional
        Whether to also give the depth of the minima as well as radius.
        The default is False.
    second_caustic : bool, optional
        Whether to give the radii of other minima. The default is False.

    Returns
    -------
    Rsp : float
        Locations of minima.
    depth : float
        Depths of minima.
    secondary : float
        Locations of second deepest minima.
    second_depth : float
        Depth of second deepest minima.
    """
    
    if array.ndim != 1:
        N = array.shape[0]
    else:
        N = 1
        array = array[np.newaxis,:]
        
    R_sp = np.zeros(N)
    depth = np.zeros(N)
    secondary = np.full(N, np.nan)
    second_depth = np.full(N, np.nan)
        
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
        if x_interp[x_max] > 2.4:
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
        #where to save minima values found to be in range
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
            deepest = np.argmin(y_interp[minima])
            test_SP = minima[deepest]
            minima = np.delete(minima, deepest) #remove deepst minima to find secondary
            second = np.argmin(y_interp[minima])
            second_min = minima[second]
            secondary[i] = x_interp[second_min]
            second_depth[i] = y_interp[second_min]
                                
        elif k == len(minima):
            #if there are no minima in range, take the deepest overall
            deepest = np.argmin(y_interp[minima])
            test_SP = minima[deepest]
                
        R_sp[i] = x_interp[test_SP]
        depth[i] = y_interp[test_SP]
                    
        if plot:   
            # Looks at minima in distribution to check Rsp is chosen properly
            plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[minima], y_interp[minima], 
                        marker="o", 
                        color="r")
            plt.scatter(x_interp[test_SP], y_interp[test_SP], 
                        marker="*", 
                        color="gold", 
                        s=100)
            plt.xlabel("r/$R_{200m}$")
            plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            plt.show()
    
    if depth_value and not second_caustic:
        return R_sp, depth #if looking for depth values, only gives deepest minima
    elif not depth_value and second_caustic:
        return R_sp, secondary
    elif depth_value and second_caustic:
        return R_sp, secondary, depth, second_depth
    return R_sp


def find_maxima(radii, array, 
                cut=-3,
                plot=False):
    """
    Calculates locations of maxima in profiles by interpolation.

    radii : array
        Radius corresponding to given profiles.
    array : array
        Profile used to find the minima radii.
    plot : bool, optional
        Whether to plot profiles to visualise minima location. 
        The default is False.
    cut : float, optional
        Depth cut of to use to identify minima. The default is -3.

    Returns
    -------
    Rsp : float
        Locations of minima.

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
        if x_interp[x_max] > 2.4:
            x_max = np.where(x_interp > 2.4)[0][0]
              
        maxima = argrelextrema(y_interp, np.greater)[0]
        depth_mask = np.where(y_interp[maxima] >= cut)[0]
        maxima = maxima[depth_mask]
                        
        test_SP = np.argmax(y_interp)
        maxima_range = np.arange(x_min, x_max, 1)
                        
        #Findsplashback
        k = 0
        l = 0
        if len(maxima) == 0:
            R_sp[i] = np.nan
            depth[i] = np.nan
            continue
        #where to save minima values found to be in range
        maxima_in_range = np.array([], dtype=int)
        for j in maxima:
            if j in maxima_range:
                test_SP = j
                maxima_in_range = np.append(maxima_in_range, j)
                l += 1
            else:
                k += 1
                                
        if l > 1:
            #if there are more than one minima in range, take the deepest
            deepest = np.argmax(y_interp[maxima])
            test_SP = maxima[deepest]
                                
        elif k == len(maxima):
            #if there are no minima in range, take the deepest overall
            deepest = np.argmax(y_interp[maxima])
            test_SP = maxima[deepest]
                
        R_sp[i] = x_interp[test_SP]
        depth[i] = y_interp[test_SP]
                    
        if plot:   
            # Looks at minima in distribution to check Rsp is chosen properly
            plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[maxima], y_interp[maxima], 
                        marker="o", 
                        color="r")
            plt.scatter(x_interp[test_SP], y_interp[test_SP], 
                        marker="*", 
                        color="gold", 
                        s=100)
            plt.xlabel("r/$R_{200m}$")
            plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            plt.title(plot)
            plt.show()
    return R_sp