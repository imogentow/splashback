import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

plt.style.use("mnras.mplstyle")

def calculate_offset(func, log_DM, log_gas, radii):
    RSP_DM = func(radii, log_DM)
    RSP_gas = func(radii, log_gas, cut=-2.5)

    offset = abs(RSP_gas - RSP_DM) / RSP_DM
    
    return offset

def number_of_minima(radii, array):
    N = array.shape[0]
    minima_length = np.zeros(N)
    for i in range(N):
        density = array[i,:]
        finite = np.isfinite(density)
        density = density[finite] #delete bad values in profiles
        temp_radii = radii[finite]
            
        density_interp = interp1d(temp_radii, density, kind="cubic")
        x_interp = 10**np.linspace(np.log10(temp_radii[1]), np.log10(temp_radii[-2]), 200)
        y_interp = density_interp(x_interp)
                
        minima = argrelextrema(y_interp, np.less)[0]
        minima_length[i] = len(minima)
        
    return minima_length


box = "L1000N1800" #"L2800N5040"
flm = sp.flamingo(box)
flm.read_properties()

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

log_DM_density = sp.log_gradients(rad_mid, flm.DM_density_3D)
log_gas_density = sp.log_gradients(rad_mid, flm.gas_density_3D)

offset = calculate_offset(dr.depth_cut, log_DM_density, log_gas_density, rad_mid)
min_length_DM = number_of_minima(rad_mid, log_DM_density)
min_length_gas = number_of_minima(rad_mid, log_gas_density)

plt.scatter(flm.M200m, offset, edgecolor="k")
plt.xscale('log')
plt.ylim((0,2))
plt.xlabel(r"$M_{\rm{200m}}$")
plt.ylabel(r"$|R_{\rm{SP, gas}} - R_{\rm{SP,DM}}| / R_{\rm{SP,DM}}$")
# plt.savefig("offset_mass.png", dpi=300)
plt.show()

plt.scatter(flm.accretion, offset, edgecolor="k")
plt.xlim((0,4))
plt.ylim((0,2))
plt.xlabel("$\Gamma$")
plt.ylabel(r"$|R_{\rm{SP, gas}} - R_{\rm{SP,DM}}| / R_{\rm{SP,DM}}$")
# plt.savefig("offset_accretion.png", dpi=300)
plt.show()

plt.scatter(flm.energy_ratio, offset, edgecolor="k")
#plt.xlim((0,4))
plt.ylim((0,2))
plt.xlabel(r"$E_{\rm{kin}}/E_{\rm{therm}}$")
plt.ylabel(r"$|R_{\rm{SP, gas}} - R_{\rm{SP,DM}}| / R_{\rm{SP,DM}}$")
# plt.savefig("offset_Erat.png", dpi=300)
plt.show()

plt.scatter(flm.hot_gas_fraction, offset, edgecolor="k")
plt.ylim((0,2))
plt.xlabel(r"$f_{\rm{hot}}$")
plt.ylabel(r"$|R_{\rm{SP, gas}} - R_{\rm{SP,DM}}| / R_{\rm{SP,DM}}$")
# plt.savefig("offset_Erat.png", dpi=300)
plt.show()

# plt.scatter(min_length_DM, offset, edgecolor="k")
# plt.scatter(min_length_gas, offset, edgecolor="k")
# plt.show()

mask = np.where(offset > 2)[0]
for i in mask:
    plt.semilogx(rad_mid, log_gas_density[i,:], label="Gas", color="b")
    plt.semilogx(rad_mid, log_DM_density[i,:], label="DM", color="k")
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, (-3,-3), c="grey", linestyle="--")
    plt.xlim(xlim)
    plt.legend()
    plt.xlabel("$r/R_{200m}$")
    plt.ylabel("$d \log y / d \log r$")
    plt.title(i)
    plt.show()