import numpy as np
import matplotlib.pyplot as plt
import splashback

mcs = splashback.macsis()
mcs.read_gas_profiles()

N_rad_bins = 45
log_radii = np.linspace(-1, 0.7, N_rad_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

def smooth_stack_grad(radii, array): #need to play around with this order
    # N = array.shape[0]
    # smoothed = np.zeros(array.shape)
    # for i in range(N):
    #     smoothed[i,:] = splashback.savitzky_golay(array[i,:], 
    #                                               window_size=15, order=4)
    # stacked = splashback.stack_data(smoothed)
    # gradient = np.gradient(np.log10(stacked), np.log10(radii))
    
    ####
    
    # log_grad = splashback.log_gradients(radii, array)
    # stacked = splashback.stack_data(log_grad)
    
    ####
    
    stacked = splashback.stack_data(array)
    log_grad = splashback.log_gradients(radii, stacked)
    
    return log_grad

accretion = np.genfromtxt("splashback_data/macsis/accretion_rates.csv", delimiter=",")
#bin_test = np.where((accretion > 2.2) & (accretion < 2.5))[0]

density_DM = smooth_stack_grad(rad_mid, mcs.DM_density)
density_gas = smooth_stack_grad(rad_mid, mcs.gas_density)
entropy = smooth_stack_grad(rad_mid, mcs.entropy)
pressure = smooth_stack_grad(rad_mid, mcs.pressure)
temperature = smooth_stack_grad(rad_mid, mcs.temperature)

plt.figure()
plt.semilogx(rad_mid, density_DM, color="green", label=r"$\rho_{\rm{DM}}$")
plt.semilogx(rad_mid, density_gas, color="red", label=r"$\rho_{\rm{gas}}$")
plt.semilogx(rad_mid, entropy, color="blue", label=r"$K$")
plt.semilogx(rad_mid, pressure, color="magenta", label="$P$")
plt.semilogx(rad_mid, temperature, color="gold", label="$T$")
plt.xlabel(r"r/$R_{\rm{200m}}$")
plt.ylabel("$ d \log y / d \log r$")
plt.legend(loc="lower left")
#plt.savefig("profiles_all_clusters.png", dpi=300)
plt.show()



#Individual cluster
#cluster_id = 5
for cluster_id in [95,106]:
    density_DM_MCS = splashback.log_gradients(rad_mid, mcs.DM_density[cluster_id,:])
    density_gas_MCS = splashback.log_gradients(rad_mid, mcs.gas_density[cluster_id,:])
    entropy_MCS = splashback.log_gradients(rad_mid, mcs.entropy[cluster_id,:])
    pressure_MCS = splashback.log_gradients(rad_mid, mcs.pressure[cluster_id,:])
    temperature_MCS = splashback.log_gradients(rad_mid, mcs.temperature[cluster_id,:])
    
    plt.figure()
    plt.semilogx(rad_mid, density_DM_MCS, color="green", label=r"$\rho_{\rm{DM}}$")
    plt.semilogx(rad_mid, density_gas_MCS, color="red", label=r"$\rho_{\rm{gas}}$")
    plt.semilogx(rad_mid, entropy_MCS, color="blue", label=r"$K$")
    plt.semilogx(rad_mid, pressure_MCS, color="magenta", label="$P$")
    plt.semilogx(rad_mid, temperature_MCS, color="gold", label="$T$")
    plt.xlabel(r"r/$R_{\rm{200m}}$")
    plt.ylabel("$ d \log y / d \log r$")
    plt.legend(loc="lower left")
    plt.ylim((-7,2.5))
    plt.title(cluster_id)
    #plt.savefig("profiles_cluster_" + str(cluster_id) + ".png", dpi=300)
    plt.show()








