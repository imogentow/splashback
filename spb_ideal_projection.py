import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import splashback as sp
from scipy.integrate import quad

plt.style.use("mnras.mplstyle")

H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G)

def density_model(radii, rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e):
    """
    Density model from O'Neil et al 2021
    
    radii needs to be given in r/R200m
    rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e are free parameters to be fit
    
    returns density model
    """
    rho_m = 0.307 * rho_crit
    rho_inner = rho_s * np.exp(-2/alpha * ((radii/r_s)**alpha - 1))
    f_trans = (1 + (radii/r_t)**beta)**(-gamma/beta)
    rho_outer = rho_m * (b_e * (radii/5)**(-1*S_e) + 1)
    
    return np.log(rho_inner * f_trans + rho_outer)

path = "splashback_data/macsis/"
DM_density = np.genfromtxt(path + "DM_density_3D_macsis.csv", delimiter=",")
DM_density_2D = np.genfromtxt(path + "projected_profiles/log_DM_only_grad_profiles_x_macsis_median_all.csv",
                              delimiter=",")

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins2 = 40
log_radii2 = np.linspace(-1, 0.6, N_bins2+1)
rad_mid2 = (10**log_radii2[1:] + 10**log_radii2[:-1]) / 2

cluster_index = 86

guess = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
guess[0] = DM_density[cluster_index,0]*100
popt, pcov = curve_fit(density_model, rad_mid, np.log(DM_density[cluster_index,:]),
                       guess, maxfev=10000)

fit_radii = np.linspace(0.11, 4.9)
fit = np.exp(density_model(fit_radii, popt[0], popt[1], popt[2], popt[3],
                           popt[4], popt[5], popt[6], popt[7]))
# fit = np.exp(density_model(rad_mid, guess[0], guess[1], guess[2], guess[3],
#                            guess[4], guess[5], guess[6], guess[7]))
# plt.figure()
# plt.loglog(rad_mid, DM_density[cluster_index,:],
#            linestyle="--",
#            color="k")
# plt.loglog(rad_mid, fit, 
#            color="r")
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel("$\\rho_{DM}$")
# plt.show()
    
log_gradient_data = sp.log_gradients(rad_mid, DM_density[cluster_index,:])

log_gradient_model = np.gradient(np.log10(fit), np.log10(fit_radii))
# plt.semilogx(rad_mid, log_gradient_data,
#              linestyle="--",
#              color="k")
# plt.semilogx(rad_mid, log_gradient_model,
#              color="r")
# plt.xlabel("r/$R_{200m}$")
# plt.ylabel(r"$\rm{d} \ln \rho_{DM} / \rm{d} \ln r$")
# plt.show()
   
splashback_3D = rad_mid[np.argmin(log_gradient_model)]
R_max = 4
projected_density = np.zeros(N_bins)

for i in range(N_bins):
    R = rad_mid[i]
    integrand = lambda r: np.exp(density_model(r, popt[0], popt[1], popt[2], popt[3],
                           popt[4], popt[5], popt[6], popt[7])) * r / np.sqrt(r**2 - R**2)
    projected_density[i] = quad(integrand, R, R_max)[0]
    
log_gradient_projected_model = np.gradient(np.log10(projected_density), np.log10(rad_mid))
splashback_2D = rad_mid[np.argmin(log_gradient_projected_model[:30])]


fig, ax = plt.subplots(nrows=2,
             ncols=1,
             sharex=True,
             figsize=(3.321,4))

ax[0].loglog(rad_mid, DM_density[cluster_index,:]/rho_crit,
           linestyle="--",
           color="k",
           label="Data")
ax[0].loglog(fit_radii, fit/rho_crit, 
           color="r",
           label="Model")
ax[0].legend()
ax[0].set_ylabel("$\\rho_{\\rm{DM}} / \\rho_{\\rm{crit}}$")

ax[1].semilogx(rad_mid, log_gradient_data,
             label="Data",
             color="maroon",
             linestyle="--")
ax[1].semilogx(fit_radii, log_gradient_model,
             label="Model",
             color="r")
ax[1].semilogx(rad_mid2, DM_density_2D[cluster_index,:],
              label="Projected data",
              color="b",
              linestyle="--")
ax[1].semilogx(rad_mid, log_gradient_projected_model,
             label="Projected model",
             color="navy")
plt.xlabel("r/$R_{200m}$")
ax[1].set_ylabel(r"$\rm{d} \ln \rho_{DM} / \rm{d} \ln r$")
ax[1].legend()
plt.savefig("projection_effect_splashback.png")
plt.show()

















