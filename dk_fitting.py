import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import splashback as sp
from scipy import integrate
import determine_radius as dr

plt.style.use("mnras.mplstyle")

H0 = 67.77 * 1000 / 3.09e22
G = 6.67e-11
rho_crit = 3 * H0**2 / (8 * np.pi * G) #kg/m^3
unit_converter = 1.99e30 / (3.09e22**3)
rho_crit = rho_crit / unit_converter

def simple_density(radii, popt):
    N_rad = len(radii)
    density = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        params = popt[i,:]
        rho_s = 10**params[0]
        rho_m = 0.307 * rho_crit
        rho_inner = rho_s * np.exp(-2/params[3] * ((radii/params[1])**params[3] - 1))
        f_trans = (1 + (radii/params[2])**params[4])**(-params[5]/params[4])
        rho_outer = rho_m * (params[6] * (radii/5)**(-1*params[7]) + 1)
        density[i,:] = rho_inner * f_trans + rho_outer
    return density

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


def log_grad_model(log_radii, rho_s, r_s, r_t, alpha, beta, gamma, b_e, S_e):
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
    log_grad = np.gradient(np.log10(rho_total), log_radii)
    return log_grad


def stack_profiles(bins, list_of_names):
    N_types = len(list_of_names)
    for i in range(N_types):
        sp.stack_fixed(flm, list_of_names[i], bins[i,:], dim="2D")
        sp.stack_fixed(flm, list_of_names[i], bins[i,:])
        if list_of_names[i] == "mass":
           split_data = np.log10(getattr(flm,"M200m"))
        else:
            split_data = getattr(flm, list_of_names[i])
        not_nan = np.where(np.isfinite(split_data)==True)[0]
        #will return 0 or len for values outside the range
        bins_sort = np.digitize(split_data[not_nan], bins[i,:])
        N_bins = len(bins[i,:]) - 1
        stacked_data = np.zeros((N_bins, N_rad))
        # print("")
        for j in range(N_bins):
            bin_mask = np.where(bins_sort == j+1)[0]
            # print(len(bin_mask))
            stacked_data[j,:] = sp.stack_data(flm.DM_density_2D[not_nan,:][bin_mask,:])
                  
        log = sp.log_gradients(rad_mid, stacked_data)
        setattr(flm, list_of_names[i]+ "_density_DM_2D", stacked_data) #currently only stacks one projection, not all
        setattr(flm, list_of_names[i]+ "_log_DM_2D", log)
    
    
def fit_models(split):
    params = np.zeros((N_bins, 8))
    profile = getattr(flm, split+"_profile_DM")
    for i in range(N_bins):
        guess = np.array([1, 0.15, 1.5, 0.06, 6, 84, 1.5, 0.9], dtype=np.float32)
        guess[0] = np.log10(profile[i,0]*100)
        popt, pcov = curve_fit(density_model, np.log10(rad_mid), 
                               np.log10(profile[i,:]),
                               guess, maxfev=10000, method='trf')
        params[i,:] = popt
        # test_data = density_model(np.log10(rad_mid), guess[0], guess[1], 
        #                           guess[2], guess[3], guess[4],
        #                           guess[5], guess[6], guess[7]) 
        # test_data2 = density_model(np.log10(rad_mid), popt[0], popt[1], 
        #                           popt[2], popt[3], popt[4],
        #                           popt[5], popt[6], popt[7])
        # plt.loglog(rad_mid, flm.accretion_profile_DM[i,:]/rho_crit, label="True")
        # #plt.loglog(rad_mid, 10**test_data, label="Test")
        # plt.loglog(rad_mid, 10**test_data2/rho_crit, label="Fit", linestyle="--")
        # plt.legend()
        # plt.show()
    return params


def fit_log_models(split):
    params = np.zeros((N_bins, 8))
    profile = getattr(flm, split+"_profile_DM")#[:,:40]
    log_profile = getattr(flm, split+"_log_DM")#[:,:40]
    guess = np.array([1, 0.15, 1.5, 0.06, 6, 84, 1.5, 0.9], dtype=np.float32)
    for i in range(N_bins):
        guess[0] = np.log10(profile[i,0]*100)
        popt, pcov = curve_fit(log_grad_model, np.log10(rad_mid), 
                                log_profile[i,:],
                                guess, maxfev=10000, method='trf')
        params[i,:] = popt
        # test_data = density_model(np.log10(rad_mid), guess[0], guess[1], 
        #                           guess[2], guess[3], guess[4],
        #                           guess[5], guess[6], guess[7]) 
        # test_data2 = density_model(np.log10(rad_mid), popt[0], popt[1], 
        #                           popt[2], popt[3], popt[4],
        #                           popt[5], popt[6], popt[7])
        # plt.loglog(rad_mid, flm.concentration_profile_DM[i,:]/rho_crit, label="True")
        #plt.loglog(rad_mid, 10**test_data, label="Test")
        # plt.loglog(rad_mid, 10**test_data2/rho_crit, label="Fit", linestyle="--")
        # plt.legend()
        # plt.show()
    return params
 
    
def project_model(radii, params):
    R_max = 5
    N_rad = len(radii)
    projected_density = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        popt = params[i,:]
        for j in range(N_rad):
            R = radii[j]
            integrand = lambda r: 10**(density_model(np.log10(r), popt[0], popt[1], popt[2], popt[3],
                                   popt[4], popt[5], popt[6], popt[7])) * r / np.sqrt(r**2 - R**2)
            projected_density[i,j], err = integrate.quad(integrand, R, R_max, 
                                                         epsrel=1e-40) 
    return projected_density


def find_sort_R(radii, array, names, plot="n"):
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
    setattr(flm, "R_"+names[0]+"_"+names[1], R_sp)


def plot_projection(params, projected_density, split, R_sp_model, R_sp_2D):
    profile_3D = getattr(flm, split+"_profile_DM")
    profile_2D = getattr(flm, split+"_density_DM_2D")
    log_3D = getattr(flm, split+"_log_DM")
    log_2D = getattr(flm, split+"_log_DM_2D")
    for i in range(N_bins):
        popt = params[i,:]
        fit_3D = 10**density_model(np.log10(rad_mid), popt[0], popt[1], 
                                   popt[2], popt[3], popt[4],
                                   popt[5], popt[6], popt[7])
        log_fit_3D = np.gradient(np.log10(fit_3D), np.log10(rad_mid))
        # log_fit_3D = log_grad_model(np.log10(rad_mid), popt[0], popt[1], 
        #                        popt[2], popt[3], popt[4],
        #                        popt[5], popt[6], popt[7])
        log_fit_2D = sp.log_gradients(rad_mid, projected_density[i,:],
                                      smooth=False)
        
        factor_3D = np.mean(profile_3D[i,:] / fit_3D)
        factor_2D = np.mean(profile_2D[i,:] / projected_density[i,:])
        fig, ax = plt.subplots(nrows=2,
             ncols=1,
             sharex=True,
             figsize=(3.321,4))

        ax[0].loglog(rad_mid, profile_3D[i,:]/rho_crit,
                   linestyle="--",
                   color="maroon",
                   label="Data")
        ax[0].loglog(rad_mid, fit_3D*factor_3D/rho_crit, 
                   color="r",
                   label="Model")
        ax[0].loglog(rad_mid, projected_density[i,:]*factor_2D/rho_crit,
                     color="navy", label="Projected model")
        ax[0].loglog(rad_mid, profile_2D[i,:]/rho_crit,
                     color="b", linestyle="--",
                     label="Projected data")
        ax[0].legend()
        ax[0].set_ylabel("$\\rho_{\\rm{DM}} / \\rho_{\\rm{crit}}$")
        
        ax[1].semilogx(rad_mid, log_2D[i,:],
                       label="Projected data",
                       color="b",
                       linestyle="--")
        ax[1].semilogx(rad_mid, log_fit_2D,
                       label="Projected model",
                       color="navy")
        ax[1].semilogx(rad_mid, log_3D[i,:],
                       label="Data",
                       color="maroon",
                       linestyle="--")
        ax[1].semilogx(rad_mid, log_fit_3D,
                       label="Model",
                       color="r")
        ylim = ax[1].get_ylim()
        ax[1].plot((R_sp_model[i], R_sp_model[i]), ylim, color="navy")
        ax[1].plot((R_sp_2D[i], R_sp_2D[i]), ylim, color="b", linestyle="--")
        # Line above is currently not plotting in the right place 
        # - not important for results but should try and find out where code has gone wrong
        ax[1].set_ylim(ylim)
        plt.xlabel("r/$R_{200m}$")
        ax[1].set_ylabel(r"$\rm{d} \ln \rho_{DM} / \rm{d} \ln r$")
        ax[1].legend()
        # plt.savefig("projection_effect_splashback.png")
        plt.show()
        

def compare_quantities(stack):
    params = fit_log_models(stack)
    N_rad_model = 44
    r_model = np.logspace(-1, 0.7, N_rad_model)
    model_density = simple_density(r_model, params)
    projected_model_density = project_model(r_model, params)
    
    projected_model_log_DM = np.zeros((N_bins, N_rad_model))
    model_log_DM = np.zeros((N_bins, N_rad_model))
    for i in range(N_bins):
        model_log_DM[i,:] = np.gradient(np.log10(model_density[i,:]), np.log10(r_model)) 
        projected_model_log_DM[i,:] = np.gradient(np.log10(projected_model_density[i,:]), 
                                             np.log10(r_model))
    
    #Projected splashback model from 3D
    find_sort_R(r_model, projected_model_log_DM, ["model", "2D"])
    find_sort_R(r_model, model_log_DM, ["model", "3D"])
    #Splashback feature in 2D from DM
    find_sort_R(rad_mid, getattr(flm, stack+"_log_DM_2D"), ["true", "2D"])
    #"True" splashback radius
    find_sort_R(rad_mid, getattr(flm, stack+"_log_DM"), ["true", "3D"]) 
    
    # plot_projection(params, projected_model_density, stack, R_sp_model_3D, R_sp_3D)
    
    #Compare the splashback from 2D profiles with projected model
    plt.figure()
    cm = plt.cm.get_cmap('autumn')
    plt.scatter(flm.R_true_2D, flm.R_model_2D, edgecolor="k", 
                c=flm.mids[0,:], cmap=cm)
    xlim = (0.7,1.62)
    ylim = plt.gca().get_ylim()
    plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r"$R_{\rm{SP, data}}/R_{\rm{200m}}$")
    plt.ylabel(r"$R_{\rm{SP, model}} /R_{\rm{200m}}$")
    plt.show()
    
    plt.figure()
    cm = plt.cm.get_cmap('autumn')
    plt.scatter(flm.R_true_3D, flm.R_model_3D, edgecolor="k", 
                c=flm.mids[0,:], cmap=cm)
    xlim = (0.7,1.62)
    ylim = plt.gca().get_ylim()
    plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r"$R_{\rm{SP, data}}/R_{\rm{200m}}$")
    plt.ylabel(r"$R_{\rm{SP, model}} /R_{\rm{200m}}$")
    plt.show()
    
    
def scatter_compare(mids, list_of_names):
    N_types = len(list_of_names)
    for i in range(N_types):
        stack = list_of_names[i]
        params = fit_log_models(stack)
        projected_density = project_model(rad_mid, params)
        projected_model_log_DM = sp.log_gradients(rad_mid, projected_density,
                                                  smooth=False)
        #Projected splashback model from 3D
        find_sort_R(rad_mid, projected_model_log_DM, 
                    ["model", stack])
        find_sort_R(rad_mid, getattr(flm, stack+"_log_EM"),
                    ["EM", stack])
        find_sort_R(rad_mid, getattr(flm, stack+"_log_SZ"),
                    ["SZ", stack])
        find_sort_R(rad_mid, getattr(flm, stack+"_log_WL"),
                    ["WL", stack])

        # find_sort_R(rad_mid, flm.gap_log_DM_2D, ["DM", stack])#, plot="y")
        # plot_projection(params, projected_density, stack,
        #                 getattr(flm, "R_model_"+stack),
        #                 getattr(flm, "R_DM_"+stack))

    
    fig, ax = plt.subplots(nrows=3, ncols=N_types,
                           sharex='col',
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(4.5,4)) #might need to change size here
    size=50
    colours = ["winter", "spring", "autumn"]
    marks = ["*", "P", "o"]
    labels = ["$s$", "$\log \langle w \\rangle$", "$M14$"]
    x = [0.29, 0.525, 0.82]
    for i in range(N_types):
        cm = plt.cm.get_cmap(colours[i])
        cb = ax[0,i].scatter(getattr(flm, "R_model_"+list_of_names[i]),
                             getattr(flm, "R_EM_"+list_of_names[i]),
                             c=mids[i,:], 
                             cmap=cm,
                             marker=marks[i],
                             edgecolor="k",
                             s=size)
        ax[1,i].scatter(getattr(flm, "R_model_"+list_of_names[i]),
                        getattr(flm, "R_SZ_"+list_of_names[i]),
                        c=mids[i,:], 
                        cmap=cm,
                        marker=marks[i],
                        edgecolor="k",
                        s=size)
        ax[2,i].scatter(getattr(flm, "R_model_"+list_of_names[i]),
                        getattr(flm, "R_WL_"+list_of_names[i]),
                        c=mids[i,:], 
                        cmap=cm,
                        marker=marks[i],
                        edgecolor="k",
                        s=size)

        cbaxes = fig.add_axes([x[i], 0.53, 0.015, 0.08]) #fix coords
        fig.colorbar(cb, cax=cbaxes, label=labels[i])

    ax[0,0].set_ylabel("$R_{\\rm{SP,EM}} / R_{\\rm{200m}}$")
    ax[1,0].set_ylabel("$R_{\\rm{SP,SZ}} / R_{\\rm{200m}}$")
    ax[2,0].set_ylabel("$R_{\\rm{SP,WL}} / R_{\\rm{200m}}$")
    
    plt.subplots_adjust(bottom=0.1)
    plt.text(0.45, 0.01, "$R_{\\rm{SP,model}} / R_{\\rm{200m}}$", 
             transform=fig.transFigure)
    filename = "splashback_data/flamingo/plots/compare_Rsp_morph.png"
    plt.savefig(filename, dpi=300)
    plt.show()


N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

if __name__ == "__main__":
    # Read data
    box = "L1000N1800"
    flm = sp.flamingo(box, "HF")
    flm.read_2D()
    flm.read_2D_properties()
    flm.read_properties()
    flm.read_magnitude_gap()
    flm.DM_density_2D = np.genfromtxt(flm.path + "_DMO_profiles_10r200m_all.csv",
                                      delimiter=",")
    
    # Define bins and bin mid points
    N_bins = 10
    c_bins = np.linspace(0.0, 0.4, N_bins+1)
    s_bins = np.linspace(0.05, 1.4, N_bins+1)
    a_bins = np.linspace(0.5, 1.5, N_bins+1)
    w_bins = np.linspace(-2.7, -1, N_bins+1)
    gap_bins = np.linspace(0,2.5, N_bins+1)
    bins = np.vstack((c_bins, s_bins, a_bins, w_bins, gap_bins))
    c_mid = (c_bins[:-1] + c_bins[1:])/2
    s_mid = (s_bins[:-1] + s_bins[1:])/2
    a_mid = (a_bins[:-1] + a_bins[1:])/2
    w_mid = (w_bins[:-1] + w_bins[1:])/2
    gap_mid = (gap_bins[:-1] + gap_bins[1:])/2
    mids = np.vstack((c_mid, s_mid, a_mid, w_mid, gap_mid))
    flm.mids = mids
    bin_type = ["concentration", "symmetry", "alignment", "centroid", "gap"]
    
    stack_profiles(bins[[1,3,4],:], ["symmetry", "centroid", "gap"])
    scatter_compare(mids[[1,3,4],:], ["symmetry", "centroid", "gap"])
    # compare_quantities("gap")
    

    
    
    
    
    
    
    
    
    
    
    
    
    