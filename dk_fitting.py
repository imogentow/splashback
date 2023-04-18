import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import splashback as sp
from scipy import integrate

plt.style.use("mnras.mplstyle")

H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G) #kg/m^3
unit_converter = 1.99e30 / (3.09e22**3)
rho_crit = rho_crit / unit_converter

def density_model(log_radii, rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e):
    """
    Density model from O'Neil et al 2021
    
    radii needs to be given in r/R200m
    rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e are free parameters to be fit
    
    returns density model
    """
    rho_s = 10**rho_s
    radii = 10**log_radii
    rho_m = 0.307 * rho_crit
    rho_inner = rho_s * np.exp(-2/alpha * ((radii/r_s)**alpha - 1))
    f_trans = (1 + (radii/r_t)**beta)**(-gamma/beta)
    rho_outer = rho_m * (b_e * (radii/5)**(-1*S_e) + 1)
    rho_total = rho_inner * f_trans + rho_outer
    
    return np.log10(rho_total)


def stack_profiles(N_bins):
    mass_bins = np.linspace(14, 15, N_bins)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins)
    energy_bins = np.append(energy_bins, 1)
    c_bins = np.linspace(0.0, 0.4, N_bins)
    c_bins = np.append(c_bins, 1)
    s_bins = np.linspace(0.0, 1.5, N_bins-1) #set extra limits on both sides
    s_bins = np.append(s_bins, 2.2)
    s_bins = np.append(-1.5, s_bins)
    a_bins = np.linspace(0.4, 1.6, N_bins-1) #set extra limits on both sides
    a_bins = np.append(a_bins, 5)
    a_bins = np.append(-1., a_bins)
    w_bins = np.linspace(-3, -1, N_bins-1) #set extra limits on both sides
    w_bins = np.append(w_bins, 0)
    w_bins = np.append(-5, w_bins)
    
    bins = np.vstack((accretion_bins, mass_bins, energy_bins, c_bins,
                      s_bins, a_bins, w_bins))
    bin_type = np.array(["accretion", "mass", "energy", 
                         "concentration", "symmetry", "alignment", "centroid"])
    for i in range(1):
        sp.stack_fixed(flm, bin_type[i], bins[i,:], dim="2D")
        sp.stack_fixed(flm, bin_type[i], bins[i,:])
        
        split_data = getattr(flm, bin_type[i])
        not_nan = np.where(np.isfinite(split_data)==True)[0]
        #will return 0 or len for values outside the range
        bins_sort = np.digitize(split_data[not_nan], bins[i,:])
        N_bins = len(bins[i,:]) - 1
        stacked_data = np.zeros((N_bins, N_rad))
        for j in range(N_bins):
            bin_mask = np.where(bins_sort == j+1)[0]
            stacked_data[j,:] = sp.stack_data(flm.DM_density_2D[not_nan,:][bin_mask,:])
                
        log = sp.log_gradients(rad_mid, stacked_data)
        setattr(flm, bin_type[i]+ "_density_DM_2D", stacked_data)
        setattr(flm, bin_type[i]+ "_log_DM_2D", log)
    
    
def fit_models():
    params = np.zeros((N_bins, 8))
    
    for i in range(N_bins):
        guess = np.array([1, 0.15, 1.5, 0.06, 6, 84, 1.5, 0.9], dtype=np.float32)
        guess[0] = np.log10(flm.accretion_density_DM[i,0]*100)
        popt, pcov = curve_fit(density_model, np.log10(rad_mid), 
                               np.log10(flm.accretion_density_DM[i,:]),
                               guess, maxfev=10000, method='trf')
        params[i,:] = popt
        # test_data = density_model(np.log10(rad_mid), guess[0], guess[1], 
        #                           guess[2], guess[3], guess[4],
        #                           guess[5], guess[6], guess[7]) 
        # test_data2 = density_model(np.log10(rad_mid), popt[0], popt[1], 
        #                           popt[2], popt[3], popt[4],
        #                           popt[5], popt[6], popt[7])
        # plt.loglog(rad_mid, flm.accretion_density_DM[i,:]/rho_crit, label="True")
        # #plt.loglog(rad_mid, 10**test_data, label="Test")
        # plt.loglog(rad_mid, 10**test_data2/rho_crit, label="Fit", linestyle="--")
        # plt.legend()
        # plt.show()
    return params
 
    
def project_models(params):
    R_max = 4
    projected_density = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        popt = params[i,:]
        for j in range(N_rad):
            R = rad_mid[j]
            integrand = lambda r: 10**(density_model(r, popt[0], popt[1], popt[2], popt[3],
                                   popt[4], popt[5], popt[6], popt[7])) * r / np.sqrt(r**2 - R**2)
            projected_density[i,j] = integrate.quad(integrand, R, R_max)[0] #CHECK HERE - VALUES TOO LOW
    return projected_density


def plot_projection(params, projected_density):
    for i in range(N_bins):
        popt = params[i,:]
        fit_3D = density_model(np.log10(rad_mid), popt[0], popt[1], 
                               popt[2], popt[3], popt[4],
                               popt[5], popt[6], popt[7])
        log_fit_3D = np.gradient(fit_3D, np.log10(rad_mid))
        
        log_fit_2D = np.gradient(np.log10(projected_density[i,:]), np.log10(rad_mid))
        
        
        fig, ax = plt.subplots(nrows=2,
             ncols=1,
             sharex=True,
             figsize=(3.321,4))

        ax[0].loglog(rad_mid, flm.accretion_density_DM[i,:]/rho_crit,
                   linestyle="--",
                   color="k",
                   label="Data")
        ax[0].loglog(rad_mid, 10**fit_3D/rho_crit, 
                   color="r",
                   label="Model")
        ax[0].legend()
        ax[0].set_ylabel("$\\rho_{\\rm{DM}} / \\rho_{\\rm{crit}}$")
        
        ax[1].semilogx(rad_mid, flm.accretion_log_DM_2D[i,:],
                       label="Projected data",
                       color="b",
                       linestyle="--")
        ax[1].semilogx(rad_mid, log_fit_2D,
                       label="Projected model",
                       color="navy")
        ax[1].semilogx(rad_mid, flm.accretion_log_DM[i,:],
                       label="Data",
                       color="maroon",
                       linestyle="--")
        ax[1].semilogx(rad_mid, log_fit_3D,
                       label="Model",
                       color="r")
                     
        plt.xlabel("r/$R_{200m}$")
        ax[1].set_ylabel(r"$\rm{d} \ln \rho_{DM} / \rm{d} \ln r$")
        ax[1].legend()
        # plt.savefig("projection_effect_splashback.png")
        plt.show()
    
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

if __name__ == "__main__":
    box = "L1000N1800"
    
    flm = sp.flamingo(box, "HF")
    flm.read_2D()
    flm.read_2D_properties()
    flm.read_properties()
    flm.DM_density_2D = np.genfromtxt(flm.path + "_DMO_profiles_10r200m_all.csv",
                                      delimiter=",")
    
    N_bins = 5
    stack_profiles(N_bins)
    params = fit_models()
    projected_density = project_models(params)
    plot_projection(params, projected_density)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    