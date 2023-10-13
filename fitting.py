import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
import determine_radius as dr
import splashback as sp

plt.style.use("mnras.mplstyle")
H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G) #kg/m^3
unit_converter = 1.99e30 / (3.09e22**3)
rho_crit = rho_crit / unit_converter


def log_grad_model(log_radii, rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e):
    """
    Density model from DK14.

    Parameters
    ----------
    log_radii : array
        Log radii of profiles, given in r/R200m.
    rho_s : str
        Fitting parameter.
    r_s : str
        Fitting parameter.
    r_t : str
        Fitting parameter.
    alpha : str
        Fitting parameter.
    beta : str
        Fitting parameter.
    gamma : str
        Fitting parameter.
    b_e : str
        Fitting parameter.
    S_e : str
        Fitting parameter.

    Returns
    -------
    log_grad : array
        Log-gradient of profiles.

    """

    rho_s = 10**rho_s
    radii = 10**log_radii
    rho_m = 0.307 * rho_crit
    rho_inner = rho_s * np.exp(-2/alpha * ((radii/r_s)**alpha - 1))
    f_trans = (1 + (radii/r_t)**beta)**(-gamma/beta)
    rho_outer = rho_m * (b_e * (radii/5)**(-1*S_e) + 1)
    rho_total = rho_inner * f_trans + rho_outer
    log_grad = np.gradient(np.log10(rho_total), log_radii)
    return log_grad
    

def density_model(log_radii, rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e):
    """
    DK14 density model.

    log_radii : array
        Log radii of profiles, given in r/R200m.
    rho_s : str
        Fitting parameter.
    r_s : str
        Fitting parameter.
    r_t : str
        Fitting parameter.
    alpha : str
        Fitting parameter.
    beta : str
        Fitting parameter.
    gamma : str
        Fitting parameter.
    b_e : str
        Fitting parameter.
    S_e : str
        Fitting parameter.

    Returns
    -------
    log_density : array
        Density from parameters.

    """
    rho_s = 10**rho_s
    radii = 10**log_radii
    rho_m = 0.307 * rho_crit
    rho_inner = rho_s * np.exp(-2/alpha * ((radii/r_s)**alpha - 1))
    f_trans = (1 + (radii/r_t)**beta)**(-gamma/beta)
    rho_outer = rho_m * (b_e * (radii/5)**(-1*S_e) + 1)
    rho_total = rho_inner * f_trans + rho_outer
    log_density = np.log10(rho_total)
    return log_density


def fit_log_models(flm, split, bootstrap=False):
    """
    Fits models to 

    Parameters
    ----------
    flm : TYPE
        DESCRIPTION.
    split : TYPE
        DESCRIPTION.
    bootstrap : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    
    profile = getattr(flm, split+"_profile_DM")
    log_profile = getattr(flm, split+"_log_DM")
    if bootstrap:
        log_profile = getattr(flm, split+"_log_DM_temp")
        profile = getattr(flm, split+"_profile_DM_temp")
    N_bins = np.shape(log_profile)[0]
    guess = np.array([1, 0.15, 1.5, 0.06, 6, 84, 1.5, 0.9], dtype=np.float32)
    params = np.zeros((N_bins, 8))
    for i in range(N_bins):
        guess[0] = np.log10(profile[i,0]*100)
        try:
            popt, pcov = curve_fit(log_grad_model, np.log10(flm.rad_mid), 
                                    log_profile[i,:],
                                    guess, maxfev=10000, method='trf')
            params[i,:] = popt
        except RuntimeError:
            params[i,:] = np.nan
    return params


def project_model(radii, params):
    """
    Uses density profile parameters to get a projected density profile.

    Parameters
    ----------
    radii : array, float
        Radii to find profile for.
    params : array, float
        Fitted parameters for profile

    Returns
    -------
    projected_density : array, float
        Projected density profile

    """
    R_max = 5
    N_rad = len(radii)
    N_bins = np.shape(params)[0]
    projected_density = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        popt = params[i,:]
        for j in range(N_rad):
            R = radii[j]
            integrand = lambda r: 10**(density_model(np.log10(r), popt[0], 
                                                     popt[1], popt[2], popt[3],
                                                     popt[4], popt[5], popt[6], 
                                                     popt[7])) * r / np.sqrt(r**2 - R**2)
            projected_density[i,j], err = integrate.quad(integrand, R, R_max, 
                                                         epsrel=1e-40) 
    return projected_density


def find_sort_R(flm, radii, array, names, plot=False):
    """
    Finds radii of minima locations in profile. Automatically sorts between
    second caustics and splashback features

    Parameters
    ----------
    radii : numpy array, float
        Array of radii corresponding to profile.
    array : numpy array, float
        Profile values corresponding to radii locations.
    names : list
        List of length two. First entry giving the name of the profile, e.g. 
        DM, gas or SZ. Second entry gives the name of the stacking criteria.

    Returns
    -------
    None.

    """
    R_sp, second = dr.depth_cut(radii, array,
                                cut=-1, 
                                second_caustic="y",
                                plot=plot)
    second_mask = np.where(np.isfinite(second))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if R_sp[index] < second[index]:
            larger = second[index]
            #smaller = R_sp[index]
            R_sp[index] = larger
            #second[index] = smaller
    return R_sp
    
    
def bootstrap_errors(data, split):
    """
    Calculates sampling error of splashback radius from fitted and projected
    dark matter density profiles.

    Parameters
    ----------
    data : obj
        simulation data
    split : str
        Name of criteria used to stack the profiles.

    Returns
    -------
    Rsp_error : array, float
        Error values for splashback radius from projected density profiles.

    """
    stacking_data = data.DM_density_3D
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)
    split_bins = getattr(data, split+"_bins")
    N_bootstrap = 100
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins) - 1
    Rsp_error = np.zeros(N_bins)
    # depth_error = np.zeros(N_bins)
    
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        stacked_data = np.zeros((N_bootstrap, data.N_rad))
        log_sample = np.zeros((N_bootstrap, data.N_rad))
        print(i, len(bin_mask))
        for j in range(N_bootstrap):
            # Select random sample from bin with replacement
            sample = np.random.choice(bin_mask, 
                                      size=len(bin_mask),
                                      replace=True)
    
            stacked_data[j,:] = sp.stack_data(stacking_data[not_nan[sample],:])
        log_sample = sp.log_gradients(data.rad_mid, stacked_data)
        setattr(data, split+"_profile_DM_temp", stacked_data)
        setattr(data, split+"_log_DM_temp", log_sample)
        params = fit_log_models(data, split, bootstrap=True)
        projected_density = project_model(data.rad_mid, params)
        projected_model_log_DM = sp.log_gradients(data.rad_mid, projected_density,
                                                  smooth=False)
        # #Projected splashback model from 3D
        R_model = find_sort_R(data, data.rad_mid, projected_model_log_DM, 
                              ["model", split])
        Rsp_error[i] = np.nanstd(R_model)
    return Rsp_error