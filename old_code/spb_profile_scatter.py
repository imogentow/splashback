import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from math import factorial
import splashback as sp

path = "Analysis_files/Splashback/"

# SMALL_SIZE = 22
# MEDIUM_SIZE = 24
# BIGGER_SIZE = 28

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

H = 67.77 * 1000 / 3.09e22 #SI
h = 0.6777
G = 6.67e-11 #SI

critical_density = 3 * H**2 / (8 * np.pi * G)

N_MACSIS = 390
N_EAGLE = 30

window = 15
order = 4


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


def log_gradients(radii, density):
    dlnrho_dlnr = np.gradient(np.log10(density), np.log10(radii))
    smoothed = savitzky_golay(dlnrho_dlnr, window_size=window, order=order)
    return smoothed


def density_r_squared_profiles(radius, density):
    """Gives relevant radii and desnity*r^2 profiles
    radius - chosen scale radius, length is how many clusters need to be investigated
    density - unscaled density profiles"""
    shape = density.shape
    N = shape[0]
    radii_new = np.zeros(shape)
    density_r_squared = np.zeros(shape)
    radii_Mpc = np.zeros(shape)
    
    for i in range(N):
        radii_Mpc[i,:] = radii_200m * R_200m_macsis[i] #Mpc
        radii_new[i,:] = radii_Mpc[i,:] / radius[i]
        density_r_squared[i,:] = density[i,:] / critical_density * radii_new[i,:]**2
        
    return radii_new, density_r_squared


R_200m_macsis = np.genfromtxt(path + "R_200m_macsis.csv", delimiter=",") / h #Mpc
R_200c_macsis = np.genfromtxt(path + "R_200c_macsis.csv", delimiter=",") / h #Mpc

R_200m_CE = np.genfromtxt(path + "R_200m_ceagle.csv", delimiter=",") / h #Mpc 
R_200c_CE = np.genfromtxt(path + "R_200c_ceagle.csv", delimiter=",") / h #Mpc

"""Read in 3D density data"""
density_gas_macsis = np.genfromtxt(path + "gas_density_3D_macsis.csv", delimiter=",") #kg/m^3
density_DM_macsis = np.genfromtxt(path + "DM_density_3D_macsis.csv", delimiter=",")
density_stars_macsis = np.genfromtxt(path + "stars_density_3D_macsis.csv", delimiter=",")

density_DMO_macsis = np.genfromtxt(path + "density_DMO_macsis_3D.csv", delimiter=",")

R_SP_macsis = np.genfromtxt(path + "R_SP_macsis_3D_DM.csv", delimiter=",") * R_200m_macsis #Mpc

R_SP_DMO = np.genfromtxt(path + "R_SP_macsis_3D_DMO.csv", delimiter=",") * R_200m_macsis

R_SP_macsis_xyz = np.hstack((R_SP_macsis, R_SP_macsis, R_SP_macsis))
upper_m = np.where(R_SP_macsis > R_200m_macsis)[0]
lower_m = np.where(R_SP_macsis < R_200m_macsis)[0]

M_200m = np.genfromtxt(path + "M_200m_macsis.csv", delimiter=",")
mass_bound = 10**5.18
low_mass = np.where(M_200m < mass_bound)[0]
high_mass = np.where(M_200m >= mass_bound)[0]

accretion = np.genfromtxt(path + "accretion_rates.csv", delimiter=",")
accretion_bound = 2.4
low_a = np.where(accretion < accretion_bound)[0]
high_a = np.where(accretion >= accretion_bound)[0]

energy_ratio = np.genfromtxt(path + "energy_ratio_200m_macsis.csv", delimiter=",")
energy_bound = 0.29
low_E = np.where(energy_ratio < energy_bound)[0]
high_E = np.where(energy_ratio >= energy_bound)[0]

density_gas_CE = np.genfromtxt(path + "gas_density_3D_ceagle.csv", delimiter=",") #kg/m^3
density_DM_CE = np.genfromtxt(path + "DM_density_3D_ceagle.csv", delimiter=",")
density_stars_CE = np.genfromtxt(path + "stars_density_3D_ceagle.csv", delimiter=",")

R_SP_CE = np.genfromtxt(path + "R_SP_ceagle_3D_DM.csv", delimiter=",") * R_200m_CE #Mpc

# upper = np.where(R_SP_CE > R_200m_CE)[0]
# lower = np.where(R_SP_CE < R_200m_CE)[0]
# vlow = np.array([0,5,28])
# plt.figure()
# plt.scatter(R_200m_CE[upper], R_SP_CE[upper]/R_200m_CE[upper], color="gold")
# plt.scatter(R_200m_CE[lower], R_SP_CE[lower]/R_200m_CE[lower], color="b")
# plt.scatter(R_200m_CE[[0,5,28]], R_SP_CE[[0,5,28]]/R_200m_CE[[0,5,28]], color="r")
# plt.figure()

"""Read in 2D observable data"""
emission_measure_macsis = np.genfromtxt(path + "emission_measure_macsis_x_all.csv", delimiter=",")
surface_density_macsis = np.genfromtxt(path + "surface_density_macsis_x_all.csv", delimiter=",")
stellar_density_macsis = np.genfromtxt(path + "stellar_density_macsis_x_all.csv", delimiter=",")
SZ_macsis = np.genfromtxt(path + "SZ_macsis_x_all.csv", delimiter=",")

R_200m_macsis_xyz = np.hstack((R_200m_macsis, R_200m_macsis, R_200m_macsis))

R_SP_macsis_DM = np.genfromtxt(path + "R_SP_macsis_2D_DM.csv", delimiter=",") * R_200m_macsis_xyz
R_SP_macsis_EM = np.genfromtxt(path + "R_SP_macsis_2D_EM.csv", delimiter=",") * R_200m_macsis_xyz
R_SP_macsis_SD = np.genfromtxt(path + "R_SP_macsis_2D_SD.csv", delimiter=",") * R_200m_macsis_xyz
R_SP_macsis_SZ = np.genfromtxt(path + "R_SP_macsis_2D_SZ.csv", delimiter=",") * R_200m_macsis_xyz


#Exclude bad clusters from MACSIS sample
low_gas = np.array([306, 307, 325, 326, 327, 331, 336, 345, 357, 360, 369, 370, 374])
good_gas = np.delete(np.arange(390), low_gas)

hydrangea = np.array([ 6, 7, 12, 13, 14, 15, 16, 18, 22, 24, 25, 28, 29])

#Initialise arrays
N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins+1)
radii_bins = 10 ** log_radii
radii_200m = (radii_bins[:-1] + radii_bins[1:]) / 2 #same for all clusters by way of measuring the density distribution

"""Create arrays of scaled radii and density"""
radii_Mpc = np.zeros((N_MACSIS, N_bins))
for i in range(N_MACSIS):
    radii_Mpc[i,:] = radii_200m * R_200m_macsis[i] #Mpc

fill, density_DM_200m = density_r_squared_profiles(R_200m_macsis, density_DM_macsis)
radii_200c, density_DM_200c = density_r_squared_profiles(R_200c_macsis, density_DM_macsis)
radii_SP, density_DM_SP = density_r_squared_profiles(R_SP_macsis, density_DM_macsis)

fill, density_gas_200m = density_r_squared_profiles(R_200m_macsis, density_gas_macsis)
radii_200c, density_gas_200c = density_r_squared_profiles(R_200c_macsis, density_gas_macsis)
radii_SP, density_gas_SP = density_r_squared_profiles(R_SP_macsis, density_gas_macsis)

fill, density_stars_200m = density_r_squared_profiles(R_200m_macsis, density_stars_macsis)
radii_200c, density_stars_200c = density_r_squared_profiles(R_200c_macsis, density_stars_macsis)
radii_SP, density_stars_SP = density_r_squared_profiles(R_SP_macsis, density_stars_macsis)

"""Compare scatter in profiles when split by mass/accretion rate/energy ratio"""
# plt.figure()
# for i in range(N_MACSIS):
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="k", alpha=0.2)
# plt.xlim((0,5))
# ylim = plt.gca().get_ylim()
# plt.show()

# plt.figure()
# for i in low_mass:
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="k", alpha=0.2)
# plt.xlim((0,5))
# plt.ylim(ylim)
# plt.show()

# plt.figure()
# for i in high_mass:
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="r", alpha=0.2)
# plt.xlim((0,5))
# plt.ylim(ylim)
# plt.show()

"""Stacking in 2D"""
radii_Mpc_xyz = radii_Mpc #np.vstack((radii_Mpc, radii_Mpc, radii_Mpc))
radii_200c_xyz = radii_200c #np.vstack((radii_200c, radii_200c, radii_200c))
radii_SP_xyz = radii_SP #np.vstack((radii_SP, radii_SP, radii_SP))
    
fill, density_WL_200m = density_r_squared_profiles(R_200m_macsis, surface_density_macsis)
fill, density_WL_200c = density_r_squared_profiles(R_200c_macsis, surface_density_macsis)
radii_SP_DM, density_WL_SP_DM = density_r_squared_profiles(R_SP_macsis_DM, surface_density_macsis)  
fill, density_WL_SP = density_r_squared_profiles(R_SP_macsis, surface_density_macsis)

fill, density_SD_200m = density_r_squared_profiles(R_200m_macsis, surface_density_macsis)
fill, density_SD_200c = density_r_squared_profiles(R_200c_macsis, surface_density_macsis)
fill, density_SD_SP = density_r_squared_profiles(R_SP_macsis_DM, surface_density_macsis)  

"""Interpolate profiles so a median can be calculated"""
N_interp = 50
x_interp_200c = np.linspace(0.2, 7.45, N_interp)
x_interp_SP = np.linspace(0.21, 4, N_interp)

interp_density_gas_200c = np.zeros((N_MACSIS, N_interp))
interp_density_gas_SP = np.zeros((N_MACSIS, N_interp))

interp_density_DM_200c = np.zeros((N_MACSIS, N_interp))
interp_density_DM_SP = np.zeros((N_MACSIS, N_interp))

interp_density_stars_200c = np.zeros((N_MACSIS, N_interp))
interp_density_stars_SP = np.zeros((N_MACSIS, N_interp))

for i in range(N_MACSIS):
    interp_gas_200c = interpolate.interp1d(radii_200c[i,:], density_gas_200c[i,:])
    interp_gas_SP = interpolate.interp1d(radii_SP[i,:], density_gas_SP[i,:])
    
    try:
        interp_density_gas_200c[i,:] = interp_gas_200c(x_interp_200c)
        interp_density_gas_SP[i,:] = interp_gas_SP(x_interp_SP)
    except ValueError:
        interp_density_gas_200c[i,:] = np.nan
        interp_density_gas_SP[i,:] = np.nan
        
    
    interp_DM_200c = interpolate.interp1d(radii_200c[i,:], density_DM_200c[i,:])
    interp_DM_SP = interpolate.interp1d(radii_SP[i,:], density_DM_SP[i,:])
    
    try:
        interp_density_DM_200c[i,:] = interp_DM_200c(x_interp_200c)
        interp_density_DM_SP[i,:] = interp_DM_SP(x_interp_SP)
    except ValueError:
        interp_density_DM_200c[i,:] = np.nan
        interp_density_DM_SP[i,:] = np.nan
        
        
    interp_stars_200c = interpolate.interp1d(radii_200c[i,:], density_stars_200c[i,:])
    interp_stars_SP = interpolate.interp1d(radii_SP[i,:], density_stars_SP[i,:])
    
    try:
        interp_density_stars_200c[i,:] = interp_stars_200c(x_interp_200c)
        interp_density_stars_SP[i,:] = interp_stars_SP(x_interp_SP)
    except ValueError:
        interp_density_stars_200c[i,:] = np.nan
        interp_density_stars_SP[i,:] = np.nan
        
    
median_gas_200m = np.zeros(N_bins)
median_gas_200c = np.zeros(N_interp)
median_gas_SP = np.zeros(N_interp)
upp_gas_SP = np.zeros(N_interp)
low_gas_SP = np.zeros(N_interp)

median_DM_200m = np.zeros(N_bins)
median_DM_200c = np.zeros(N_interp)
median_DM_SP = np.zeros(N_interp)

upp_DM_SP = np.zeros(N_interp)
low_DM_SP = np.zeros(N_interp)

upp_DM_200c = np.zeros(N_interp)
low_DM_200c = np.zeros(N_interp)

upp_DM_200m = np.zeros(N_bins)
low_DM_200m = np.zeros(N_bins)

median_stars_200m = np.zeros(N_bins)
median_stars_200c = np.zeros(N_interp)
median_stars_SP = np.zeros(N_interp)
    
for i in range(N_bins):
    median_gas_200m[i] = np.nanmedian(density_gas_200m[:,i])
    median_DM_200m[i] = np.nanmedian(density_DM_200m[:,i])
    median_stars_200m[i] = np.nanmedian(density_stars_200m[:,i])
    
    upp_DM_200m[i] = np.nanpercentile(density_DM_200m[:,i], 25)
    low_DM_200m[i] = np.nanpercentile(density_DM_200m[:,i], 75)
    

median_low_mass = np.zeros(N_interp)
median_high_mass = np.zeros(N_interp)
upp_low_mass = np.zeros(N_interp)
low_low_mass = np.zeros(N_interp)
upp_high_mass = np.zeros(N_interp)
low_high_mass = np.zeros(N_interp)

median_low_a = np.zeros(N_interp)
median_high_a = np.zeros(N_interp)
upp_low_a = np.zeros(N_interp)
low_low_a = np.zeros(N_interp)
upp_high_a = np.zeros(N_interp)
low_high_a = np.zeros(N_interp)

median_low_E = np.zeros(N_interp)
median_high_E = np.zeros(N_interp)
upp_low_E = np.zeros(N_interp)
low_low_E = np.zeros(N_interp)
upp_high_E = np.zeros(N_interp)
low_high_E = np.zeros(N_interp)

for i in range(N_interp):
    median_gas_200c[i] = np.nanmedian(interp_density_gas_200c[:,i])
    median_gas_SP[i] = np.nanmedian(interp_density_gas_SP[:,i])
    
    upp_gas_SP[i] = np.nanpercentile(interp_density_gas_SP[:,i], 25)
    low_gas_SP[i] = np.nanpercentile(interp_density_gas_SP[:,i], 75)
    
    median_DM_200c[i] = np.nanmedian(interp_density_DM_200c[:,i])
    median_DM_SP[i] = np.nanmedian(interp_density_DM_SP[:,i])
    
    upp_DM_200c[i] = np.nanpercentile(interp_density_DM_200c[:,i], 25)
    low_DM_200c[i] = np.nanpercentile(interp_density_DM_200c[:,i], 75)
    
    upp_DM_SP[i] = np.nanpercentile(interp_density_DM_SP[:,i], 25)
    low_DM_SP[i] = np.nanpercentile(interp_density_DM_SP[:,i], 75)
    
    median_stars_200c[i] = np.nanmedian(interp_density_stars_200c[:,i])
    median_stars_SP[i] = np.nanmedian(interp_density_stars_SP[:,i])
    
    #Medians for sub-groups
    median_low_mass[i] = np.nanmedian(interp_density_DM_SP[low_mass,i])
    median_high_mass[i] = np.nanmedian(interp_density_DM_SP[high_mass,i])
    upp_low_mass[i] = np.nanpercentile(interp_density_DM_SP[low_mass,i], 25)
    low_low_mass[i] = np.nanpercentile(interp_density_DM_SP[low_mass,i], 75)
    upp_high_mass[i] = np.nanpercentile(interp_density_DM_SP[high_mass,i], 25)
    low_high_mass[i] = np.nanpercentile(interp_density_DM_SP[high_mass,i], 75)
    
    median_low_a[i] = np.nanmedian(interp_density_DM_SP[low_a,i])
    median_high_a[i] = np.nanmedian(interp_density_DM_SP[high_a,i])
    upp_low_a[i] = np.nanpercentile(interp_density_DM_SP[low_a,i], 25)
    low_low_a[i] = np.nanpercentile(interp_density_DM_SP[low_a,i], 75)
    upp_high_a[i] = np.nanpercentile(interp_density_DM_SP[high_a,i], 25)
    low_high_a[i] = np.nanpercentile(interp_density_DM_SP[high_a,i], 75)
    
    median_low_E[i] = np.nanmedian(interp_density_DM_SP[low_E,i])
    median_high_E[i] = np.nanmedian(interp_density_DM_SP[high_E,i])
    upp_low_E[i] = np.nanpercentile(interp_density_DM_SP[low_E,i], 25)
    low_low_E[i] = np.nanpercentile(interp_density_DM_SP[low_E,i], 75)
    upp_high_E[i] = np.nanpercentile(interp_density_DM_SP[high_E,i], 25)
    low_high_E[i] = np.nanpercentile(interp_density_DM_SP[high_E,i], 75)
    
# plt.figure()
# plt.semilogy(x_interp_SP, median_low_mass, color="r", label="low mass")
# plt.semilogy(x_interp_SP, median_high_mass, color="k", label="high mass")
# plt.fill_between(x_interp_SP, low_low_mass, upp_low_mass, facecolor="r", edgecolor="none", alpha=0.4)
# plt.fill_between(x_interp_SP, low_high_mass, upp_high_mass, facecolor="grey", edgecolor="none", alpha=0.4)
# plt.legend()
# plt.xlabel("$r/R_{SP}$")
# plt.ylabel("$(r/R_{SP})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")
# plt.show()

# plt.figure()
# plt.semilogy(x_interp_SP, median_low_a, color="r", label="low $\Gamma$")
# plt.semilogy(x_interp_SP, median_high_a, color="k", label="high $\Gamma$")
# plt.fill_between(x_interp_SP, low_low_a, upp_low_a, facecolor="r", edgecolor="none", alpha=0.4)
# plt.fill_between(x_interp_SP, low_high_a, upp_high_a, facecolor="grey", edgecolor="none", alpha=0.4)
# plt.legend()
# plt.xlabel("$r/R_{SP}$")
# plt.ylabel("$(r/R_{SP})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")
# plt.show()

# plt.figure()
# plt.semilogy(x_interp_SP, median_low_E, color="r", label="low $E_{\\rm{kin}} / E_{\\rm{therm}}$")
# plt.semilogy(x_interp_SP, median_high_E, color="k", label="high $E_{\\rm{kin}} / E_{\\rm{therm}}$")
# plt.fill_between(x_interp_SP, low_low_E, upp_low_E, facecolor="r", edgecolor="none", alpha=0.4)
# plt.fill_between(x_interp_SP, low_high_E, upp_high_E, facecolor="grey", edgecolor="none", alpha=0.4)
# plt.legend()
# plt.xlabel("$r/R_{SP}$")
# plt.ylabel("$(r/R_{SP})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")
# plt.show()
    
plt.figure(figsize=(6,4))
plt.semilogy(x_interp_200c, median_DM_200c, color="teal", label="$R_{200c}$")
plt.fill_between(x_interp_200c, low_DM_200c, upp_DM_200c, facecolor="teal", edgecolor="none", alpha=0.6)
plt.semilogy(x_interp_SP, median_DM_SP, color="slateblue", label="$R_{SP}$")
plt.fill_between(x_interp_SP, low_DM_SP, upp_DM_SP, facecolor="slateblue", edgecolor="none", alpha=0.6)
plt.semilogy(radii_200m, median_DM_200m, color="orangered", label="$R_{200m}$")
plt.fill_between(radii_200m, low_DM_200m, upp_DM_200m, facecolor="orangered", edgecolor="none", alpha=0.6)
plt.xlim((0.2,3))
plt.xlabel("$r/R_{X}$")
plt.ylabel("$(r/R_{x})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_radius_macsis.png", dpi=300, transparent=True)
plt.show()

plt.figure(figsize=(5,5*2/3))
plt.semilogy(x_interp_200c, median_DM_200c, color="teal", label="$R_{200c}$")
plt.fill_between(x_interp_200c, low_DM_200c, upp_DM_200c, facecolor="teal", edgecolor="none", alpha=0.6)
plt.semilogy(x_interp_SP, median_DM_SP, color="slateblue", label="$R_{SP}$")
plt.fill_between(x_interp_SP, low_DM_SP, upp_DM_SP, facecolor="slateblue", edgecolor="none", alpha=0.6)
plt.semilogy(radii_200m, median_DM_200m, color="orangered", label="$R_{200m}$")
plt.fill_between(radii_200m, low_DM_200m, upp_DM_200m, facecolor="orangered", edgecolor="none", alpha=0.6)
plt.xlim((0.2,1.1))
plt.xlabel("$r/R_{X}$")
plt.ylabel("$(r/R_{x})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")
plt.tight_layout()
plt.savefig("scatter_radius_macsis_zoom.png", dpi=300, transparent=True)
plt.show()
    
# #Gas
# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_200m, density_gas_200m[i,:], alpha=0.2, color="r")
# plt.semilogy(radii_200m, median_gas_200m, linewidth=3, color="maroon") 
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{200m})^{2}$")
# plt.show()

# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_200c[i,:], density_gas_200c[i,:], alpha=0.2, color="r")
# plt.semilogy(x_interp_200c, median_gas_200c, linewidth=3, color="maroon")
# plt.xlabel("r/$R_{200c}$")
# plt.xlim((0, 7.6))
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{200c})^{2}$")
# plt.show()

# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_SP[i,:], density_gas_SP[i,:], alpha=0.2, color="r")
# plt.semilogy(x_interp_SP, median_gas_SP, linewidth=3, color="maroon")
# plt.semilogy(x_interp_SP, upp_gas_SP, linewidth=3, color="maroon")
# plt.semilogy(x_interp_SP, low_gas_SP, linewidth=3, color="maroon")
# plt.xlabel("r/$R_{SP}$")
# plt.xlim((0, 7.6))
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# plt.show()


#DM
# fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3,7))
# for i in good_gas:
#     ax[0].semilogy(radii_200m, density_DM_200m[i,:], alpha=0.1, color="k")
# ax[0].semilogy(radii_200m, median_DM_200m, color="r", linewidth=2)
# ax[0].set_xlabel("r/$R_{200m}$")
# #ax[0].set_ylabel("$\\rho(r) / \\rho_{crit} (r/R_{200m})^{2}$")

# for i in good_gas:
#     ax[1].semilogy(radii_200c[i,:], density_DM_200c[i,:], alpha=0.2, color="k")
# ax[1].semilogy(x_interp_200c, median_DM_200c, color="r", linewidth=2)
# ax[1].set_xlabel("r/$R_{200c}$")
# ax[1].set_xlim((0, 7.6))
# ax[1].set_ylabel("$(r/R_{200c})^{2} \\rho(r) / \\rho_{crit}$")

# for i in good_gas:
#     ax[2].semilogy(radii_SP[i,:], density_DM_SP[i,:], alpha=0.2, color="k")
# ax[2].semilogy(x_interp_SP, median_DM_SP, color="r", linewidth=2)
# ax[2].set_xlabel("r/$R_{SP}$")
# ax[2].set_xlim((0, 7.6))
# plt.subplots_adjust(top=1.3)
# #ax[2].set_ylabel("$ (r/R_{SP})^{2} \\rho_{DM}(r) / \\rho_{crit} $")
# plt.show()


# #stars
# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_200m, density_stars_200m[i,:], alpha=0.1, color="cornflowerblue")
# plt.semilogy(radii_200m, median_stars_200m, color="midnightblue", linewidth=2)
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{200m})^{2}$")
# plt.show()

# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_200c[i,:], density_stars_200c[i,:], alpha=0.2, color="cornflowerblue")
# plt.semilogy(x_interp_200c, median_stars_200c, color="midnightblue", linewidth=2)
# plt.xlabel("r/$R_{200c}$")
# plt.xlim((0, 7.6))
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{200c})^{2}$")
# plt.show()

# plt.figure()
# for i in good_gas:
#     plt.semilogy(radii_SP[i,:], density_stars_SP[i,:], alpha=0.2, color="cornflowerblue")
# plt.semilogy(x_interp_SP, median_stars_SP, color="midnightblue", linewidth=2)
# plt.xlabel("r/$R_{SP}$")
# plt.xlim((0, 7.6))
# plt.ylabel("$\\rho(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# plt.show()


############################# C-EAGLE Analysis #############################
# radii_Mpc = np.zeros((N_EAGLE, N_bins))
# radii_200c = np.zeros((N_EAGLE, N_bins))
# radii_SP = np.zeros((N_EAGLE, N_bins))

# density_gas_200m = np.zeros((N_EAGLE, N_bins))
# density_gas_200c = np.zeros((N_EAGLE, N_bins))
# density_gas_SP = np.zeros((N_EAGLE, N_bins))

# density_DM_200m = np.zeros((N_EAGLE, N_bins))
# density_DM_200c = np.zeros((N_EAGLE, N_bins))
# density_DM_SP = np.zeros((N_EAGLE, N_bins))

# density_stars_200m = np.zeros((N_EAGLE, N_bins))
# density_stars_200c = np.zeros((N_EAGLE, N_bins))
# density_stars_SP = np.zeros((N_EAGLE, N_bins))

# for i in range(N_EAGLE):
#     radii_Mpc[i,:] = radii_200m * R_200m_CE[i] #Mpc
#     radii_200c[i,:] = radii_Mpc[i,:] / R_200c_CE[i]
#     radii_SP[i,:] = radii_Mpc[i,:] / R_SP_CE[i]
    
#     density_gas_200m[i,:] = density_gas_CE[i,:] / critical_density * radii_200m**2
#     density_gas_200c[i,:] = density_gas_CE[i,:] / critical_density * radii_200c[i,:]**2
#     density_gas_SP[i,:] = density_gas_CE[i,:] / critical_density * radii_SP[i,:]**2
    
#     density_DM_200m[i,:] = density_DM_CE[i,:] / critical_density * radii_200m**2
#     density_DM_200c[i,:] = density_DM_CE[i,:] / critical_density * radii_200c[i,:]**2
#     density_DM_SP[i,:] = density_DM_CE[i,:] / critical_density * radii_SP[i,:]**2
    
#     density_stars_200m[i,:] = density_stars_CE[i,:] / critical_density * radii_200m**2
#     density_stars_200c[i,:] = density_stars_CE[i,:] / critical_density * radii_200c[i,:]**2
#     density_stars_SP[i,:] = density_stars_CE[i,:] / critical_density * radii_SP[i,:]**2
   
    
# x_interp_200c = np.linspace(0.2, 7.45, N_interp)
# x_interp_SP = np.linspace(0.21, 4, N_interp)

# interp_density_gas_200c = np.zeros((N_EAGLE, N_interp))
# interp_density_gas_SP = np.zeros((N_EAGLE, N_interp))

# interp_density_DM_200c = np.zeros((N_EAGLE, N_interp))
# interp_density_DM_SP = np.zeros((N_EAGLE, N_interp))

# interp_density_stars_200c = np.zeros((N_EAGLE, N_interp))
# interp_density_stars_SP = np.zeros((N_EAGLE, N_interp))

# log_DM_200m = np.zeros((N_EAGLE, N_bins))
# log_DM_SP = np.zeros((N_EAGLE, N_bins))

# for i in range(N_EAGLE):
#     interp_gas_200c = interpolate.interp1d(radii_200c[i,:], density_gas_200c[i,:])
#     interp_gas_SP = interpolate.interp1d(radii_SP[i,:], density_gas_SP[i,:])
    
#     try:
#         interp_density_gas_200c[i,:] = interp_gas_200c(x_interp_200c)
#         interp_density_gas_SP[i,:] = interp_gas_SP(x_interp_SP)
#     except ValueError:
#         interp_density_gas_200c[i,:] = np.nan
#         interp_density_gas_SP[i,:] = np.nan
        
    
#     interp_DM_200c = interpolate.interp1d(radii_200c[i,:], density_DM_200c[i,:])
#     interp_DM_SP = interpolate.interp1d(radii_SP[i,:], density_DM_SP[i,:])
    
#     try:
#         interp_density_DM_200c[i,:] = interp_DM_200c(x_interp_200c)
#         interp_density_DM_SP[i,:] = interp_DM_SP(x_interp_SP)
#     except ValueError:
#         interp_density_DM_200c[i,:] = np.nan
#         interp_density_DM_SP[i,:] = np.nan
        
        
#     interp_stars_200c = interpolate.interp1d(radii_200c[i,:], density_stars_200c[i,:])
#     interp_stars_SP = interpolate.interp1d(radii_SP[i,:], density_stars_SP[i,:])
    
#     try:
#         interp_density_stars_200c[i,:] = interp_stars_200c(x_interp_200c)
#         interp_density_stars_SP[i,:] = interp_stars_SP(x_interp_SP)
#     except ValueError:
#         interp_density_stars_200c[i,:] = np.nan
#         interp_density_stars_SP[i,:] = np.nan
        
#     log_DM_200m[i,:] = log_gradients(radii_200m, density_DM_CE[i,:])
#     log_DM_SP[i,:] = log_gradients(radii_SP[i,:], density_DM_SP[i,:])
    
    
# median_gas_200m = np.zeros(N_bins)
# median_gas_200c = np.zeros(N_interp)
# median_gas_SP = np.zeros(N_interp)
# upp_gas_SP = np.zeros(N_interp)
# low_gas_SP = np.zeros(N_interp)

# median_DM_200m = np.zeros(N_bins)
# median_DM_200c = np.zeros(N_interp)
# median_DM_SP = np.zeros(N_interp)

# median_stars_200m = np.zeros(N_bins)
# median_stars_200c = np.zeros(N_interp)
# median_stars_SP = np.zeros(N_interp)
    
# for i in range(N_bins):
#     median_gas_200m[i] = np.nanmedian(density_gas_200m[:,i])
#     median_DM_200m[i] = np.nanmedian(density_DM_200m[:,i])
#     median_stars_200m[i] = np.nanmedian(density_stars_200m[:,i])  
    
# for i in range(N_interp):
#     median_gas_200c[i] = np.nanmedian(interp_density_gas_200c[:,i])
#     median_gas_SP[i] = np.nanmedian(interp_density_gas_SP[:,i])
    
#     upp_gas_SP[i] = np.nanpercentile(interp_density_gas_SP[:,i], 25)
#     low_gas_SP[i] = np.nanpercentile(interp_density_gas_SP[:,i], 75)
    
#     median_DM_200c[i] = np.nanmedian(interp_density_DM_200c[:,i])
#     median_DM_SP[i] = np.nanmedian(interp_density_DM_SP[:,i])
    
#     median_stars_200c[i] = np.nanmedian(interp_density_stars_200c[:,i])
#     median_stars_SP[i] = np.nanmedian(interp_density_stars_SP[:,i])
    
#gas 
# plt.figure()
# for i in range(N_EAGLE):
#     plt.semilogy(radii_200m, density_gas_200m[i,:], color="r", alpha=0.6)
# plt.semilogy(radii_200m, median_gas_200m, color="maroon")
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel("$\\rho_{\mathrm{gas}}(r) / \\rho_{crit} (r/R_{200m})^{2}$")
# plt.show()
    
# plt.figure()
# for i in range(N_EAGLE):
#     plt.semilogy(radii_200c[i,:], density_gas_200c[i,:], color="r", alpha=0.6)
# plt.semilogy(x_interp_200c, median_gas_200c, color="maroon")
# plt.xlabel("r/$R_{200c}$")
# plt.ylabel("$\\rho_{\mathrm{gas}}(r) / \\rho_{crit} (r/R_{200c})^{2}$")
# plt.show() 
    
# plt.figure()
# for i in range(N_EAGLE):
#     plt.semilogy(radii_SP[i,:], density_gas_SP[i,:], color="r", alpha=0.6)
# plt.semilogy(x_interp_SP, median_gas_SP, color="maroon")
# plt.xlabel("r/$R_{SP}$")
# plt.ylabel("$\\rho_{\mathrm{gas}}(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# plt.xlim(-0.2, 7.5)
# plt.show()  

#DM
# plt.figure()
# for i in range(N_EAGLE):
#     plt.semilogy(radii_200m, density_DM_200m[i,:], color="k", alpha=0.6)
# plt.semilogy(radii_200m, median_DM_200m, color="maroon", linewidth=3)
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel("$\\rho_{\mathrm{DM}}(r) / \\rho_{crit} (r/R_{200m})^{2}$")
# plt.show()
    
# plt.figure()
# for i in range(N_EAGLE):
#     plt.semilogy(radii_200c[i,:], density_DM_200c[i,:], color="k", alpha=0.6)
# plt.semilogy(x_interp_200c, median_DM_200c, color="maroon", linewidth=3)
# plt.xlabel("r/$R_{200c}$")
# plt.ylabel("$\\rho_{\mathrm{DM}}(r) / \\rho_{crit} (r/R_{200c})^{2}$")
# plt.show() 
    
# plt.figure()
# for i in upper:
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="k", alpha=0.6)
# for i in lower:
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="b", alpha=0.6)
# for i in vlow:
#     plt.semilogy(radii_SP[i,:], density_DM_SP[i,:], color="r", alpha=0.6)
# #plt.semilogy(x_interp_SP, median_DM_SP, color="maroon", linewidth=3)
# plt.xlabel("r/$R_{SP}$")
# plt.ylabel("$\\rho_{\mathrm{DM}}(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# plt.xlim((0,1))
# plt.show()

# plt.figure()
# for i in upper:
#     plt.semilogx(radii_SP[i,:], log_DM_SP[i,:], color="r", alpha=0.6)
# for i in lower:
#     plt.semilogx(radii_SP[i,:], log_DM_SP[i,:], color="b", alpha=0.6)
# for i in vlow:
#     plt.semilogx(radii_SP[i,:], log_DM_SP[i,:], color="k", alpha=0.6)
# plt.xlabel("r/$R_{SP}$")
# plt.ylabel("$\\rho_{\mathrm{DM}}(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# plt.xlim((0.1, 4))
# plt.ylim((-5,5))
# plt.show()

# plt.figure()
# for i in upper:
#     plt.semilogx(radii_200m, log_DM_200m[i,:], color="r", alpha=0.6)
# for i in lower:
#     plt.semilogx(radii_200m, log_DM_200m[i,:], color="b", alpha=0.6)
# for i in vlow:
#     plt.semilogx(radii_200m, log_DM_200m[i,:], color="k", alpha=0.6)
# plt.xlabel("r/$R_{SP}$")
# plt.ylabel("$\\rho_{\mathrm{DM}}(r) / \\rho_{crit} (r/R_{SP})^{2}$")
# #plt.xlim((0.1, 4))
# #plt.ylim((-6,2))
# plt.show()







