import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr

plt.style.use("mnras.mplstyle")

def compare(func, log_DM, log_gas, radii):
    
    RSP_DM = func(radii, log_DM)
    RSP_gas = func(radii, log_gas)
        
    func_name = func.__name__
    filename = "Rsp_compare_3D_" + func_name + ".png"
        
    plt.figure()
    plt.scatter(RSP_DM, RSP_gas, edgecolor="k")
    plt.xlabel("$R_{SP, DM}$")
    plt.ylabel("$R_{SP, gas}$")
    plt.xlim((0.3,2.3))
    # plt.ylim((0.3,2.3))
    # plt.savefig(filename, dpi=300)
    plt.show()
        
    return RSP_DM, RSP_gas
        

box = "L2800N5040" #"L1000N1800"
flm = sp.flamingo(box)

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

log_DM_density = sp.log_gradients(rad_mid, flm.DM_density_3D)
log_gas_density = sp.log_gradients(rad_mid, flm.gas_density_3D)
# log_gas_pressure = sp.log_gradients(rad_mid, flm.gas_pressure_3D)
log_gas_entropy = sp.log_gradients(rad_mid, flm.gas_entropy_3D)

RSP_DM, RSP_gas = compare(dr.standard, log_DM_density, log_gas_density, rad_mid) #use depth cut to determine Rsp
RSP_DM, RSP_gas = compare(dr.depth_cut, log_DM_density, log_gas_density, rad_mid) #use depth cut to determine Rsp

    
