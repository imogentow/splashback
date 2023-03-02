import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import splashback as sp

"""CODE DOESN'T CURRENTLY READ IN STANDARD DATA.
NEED TO RUN CODE TO GET FRESH DATA THAT IS CONSISTENT WITH CURRENT DATA.
ALSO DON'T HAVE DATA AT HIGH RADII FRO ALL PROJECTIONS YET"""

path = "splashback_data/"
window = 15
order = 4

N_macsis = 390
N_ceagle = 30

N_bins = 40
log_radii = np.linspace(-1, 0.6, N_bins+1)
radii_bins = 10 ** log_radii
radii_mid = (radii_bins[:-1] + radii_bins[1:]) / 2 #r/r_200m

macsis = sp.macsis()
ceagle = sp.ceagle()

"""Read in splashback radii"""
R_SP_macsis = np.genfromtxt(path + "R_SP_macsis_3D_DM.csv", delimiter=",")
R_SP_CE = np.genfromtxt(path + "R_SP_ceagle_3D_DM.csv", delimiter=",")


def minima_finding(radii, density, plot="n"):
    finite = np.isfinite(density)
    density = density[finite]
    radii = radii[finite]
        
    density_interp = interp1d(radii, density, kind="cubic")
    x_interp = 10**np.linspace(np.log10(radii[0]), np.log10(radii[-2]), 200)
    y_interp = density_interp(x_interp)
            
    minima = argrelextrema(y_interp, np.less)[0]
    
    minima_range = np.arange(69,155,1) #this one is for the interpolated profiles
            
    #Findsplashback
        
    k = 0
    l = 0
    minima_in_range = np.array([], dtype=int)
    for j in minima:
        if j in minima_range: #searches for minima within specific range
            arg_SP = j
            minima_in_range = np.append(minima_in_range, j)
            l += 1
        else:
            k += 1
                
        if l > 1: #in case multiple minima are found within the range given
            arg_SP = minima_in_range[np.argmin(y_interp[minima_in_range])]
                
        if k == len(minima): #if no minima are found within range
            arg_SP = minima[np.argmin(y_interp[minima])]
    
    SP_radius = x_interp[arg_SP]
    
    try:
        minima = np.delete(minima, np.where(minima <= arg_SP+20)[0]) #only look at minima beyond splashback radius
        shock_arg = minima[np.argmin(y_interp[minima])]
        
        if plot != "n":   
            # Looks at minima in distribution to check Rsp is chosen properly
            #plt.figure()
            plt.semilogx(x_interp, y_interp, color="k")
            plt.scatter(x_interp[minima], y_interp[minima], marker="o", color="r")
            plt.scatter(x_interp[arg_SP], y_interp[arg_SP], marker="*", 
                        color="gold", s=100)
            plt.scatter(x_interp[shock_arg], y_interp[shock_arg], marker="*", color="b", s=100)
            # plt.xlabel("r/$R_{200m}$")
            # plt.ylabel("$\mathrm{d} \log \\rho / \mathrm{d} \log r$")
            # plt.title(plot)
            # plt.show()
            
    except ValueError:
        return SP_radius, np.nan
    
    shock_radius = x_interp[shock_arg]
    return SP_radius, shock_radius

log_EM_macsis = np.zeros((N_macsis, N_bins))
log_SZ_macsis = np.zeros((N_macsis, N_bins))
log_WL_macsis = np.zeros((N_macsis, N_bins))


SP_radius = np.zeros(N_macsis)
shock_radius = np.zeros(N_macsis)

SP_radius_EM = np.zeros(N_macsis)
shock_radius_EM = np.zeros(N_macsis)

SP_radius_SZ = np.zeros(N_macsis)
shock_radius_SZ = np.zeros(N_macsis)

SP_radius_WL = np.zeros(N_macsis)
shock_radius_WL = np.zeros(N_macsis)
for i in range(50):
    SP_radius[i], shock_radius[i] = minima_finding(radii_mid, macsis.gas_density_3D[i,:])
    SP_radius_EM[i], shock_radius_EM[i] = minima_finding(radii_mid, macsis.EM_median[i,:])
    SP_radius_SZ[i], shock_radius_SZ[i] = minima_finding(radii_mid, macsis.SZ_median[i,:])
    SP_radius_WL[i], shock_radius_WL[i] = minima_finding(radii_mid, macsis.DM_median[i,:])
    

plt.figure(figsize=(7,6))
plt.semilogx(radii_mid, macsis.EM_median[26,:], color="gold", label="EM")
plt.semilogx(radii_mid, macsis.SZ_median[26,:], color="cyan", label="SZ")
ylim = plt.gca().get_ylim()
plt.semilogx([SP_radius_EM[26], SP_radius_EM[26]], ylim, color="darkkhaki", linestyle="--")
plt.semilogx([SP_radius_SZ[26], SP_radius_SZ[26]], ylim, color="darkcyan", linestyle="--")
plt.semilogx([shock_radius_EM[26], shock_radius_EM[26]], ylim, color="darkkhaki", linestyle="--")
plt.semilogx([shock_radius_SZ[26], shock_radius_SZ[26]], ylim, color="darkcyan", linestyle="--")
ylim2 = plt.gca().get_ylim()
plt.ylim((ylim[0], ylim[1]))
plt.xlabel("$r/R_{200m}$")
plt.ylabel("$d \ln \Sigma / d \ln r$")
plt.legend()
plt.tight_layout()
#plt.savefig("shock_radii_macsis_26.png", dpi=300)
plt.show()









