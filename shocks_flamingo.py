import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")

N_bins = 71
log_radii = np.linspace(-1, 0.9, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_temp = 45
log_radii = np.linspace(-1, 0.7, N_temp)
rad_temp = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

# N_temp = 71
# log_radii = np.linspace(-1, 0.9, N_bins)
# rad_temp = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

box = "L1000N1800" #"L2800N5040"
run = "HF"
# flm = sp.flamingo(box)

path = "splashback_data/flamingo/"
filename = path + box + "_" + run + "_3D_gas_" 
entropy = np.genfromtxt(filename + "entropy_long_all.csv", delimiter=",")
gas_density = np.genfromtxt(filename + "density_long_all.csv", delimiter=",")
DM_density = np.genfromtxt(path + box + "_" + run + "_3D_DM_density_all.csv",
                     delimiter=",")

log_grad_DM = sp.log_gradients(rad_temp, DM_density)
log_grad_entropy = sp.log_gradients(rad_mid, entropy) 
log_grad_density = sp.log_gradients(rad_mid, gas_density)

R_shock = dr.shock_finder(rad_mid, log_grad_entropy)

R_DM = dr.depth_cut(rad_temp, log_grad_DM, cut=-2.8)
R_gas_old = dr.depth_cut(rad_mid, log_grad_density, cut=-2.5)
R_gas_prior = dr.entropy_prior(rad_mid,log_grad_density, log_grad_entropy)

plt.scatter(R_DM, R_gas_prior, edgecolor="k")
ylim = plt.gca().get_ylim()
xlim = plt.gca().get_xlim()
plt.show()

plt.scatter(R_DM, R_gas_old, edgecolor="k")
plt.xlim(xlim)
plt.ylim(ylim)
plt.show()

mask = np.where(R_DM > 3)[0]
for i in mask:
    log_plot = log_grad_entropy[i,:]
    mask = np.where(log_plot != 0)[0]
    log_plot = log_plot[mask]
    rad_plot = rad_mid[mask]
    
    plt.semilogx(rad_plot, log_plot, label="K", color="gold")
    plt.semilogx(rad_mid, log_grad_density[i,:], label="$\\rho_{\\rm{gas}}$", color="b")
    plt.semilogx(rad_temp, log_grad_DM[i,:], label="$\\rho_{\\rm{DM}}$", color="k")
    ylim = plt.gca().get_ylim()
    if ylim[0] < -5:
        y_new0 = -5
    else:
        y_new0 = ylim[0]
    if ylim[1] > 3:
        y_new1 = 3
    else:
        y_new1 = ylim[1]
    ylim_new = (y_new0, y_new1)
    plt.plot((R_shock[i], R_shock[i]), ylim_new, color="gold")
    plt.plot((R_gas_old[i], R_gas_old[i]), ylim_new, linestyle="--", color="b")
    plt.plot((R_gas_prior[i], R_gas_prior[i]), ylim_new, linestyle=":", color="r")
    plt.plot((R_DM[i], R_DM[i]), ylim_new, color="k")
    plt.ylim(ylim_new)
    plt.legend()
    plt.title(i)
    plt.show()
    
