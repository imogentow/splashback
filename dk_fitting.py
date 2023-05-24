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


def stack_profiles(N_bins):
    mass_bins = np.linspace(14, 15, N_bins)
    mass_bins = np.append(mass_bins, 16)
    accretion_bins = np.linspace(0, 4, N_bins)
    accretion_bins = np.append(accretion_bins, 20)
    energy_bins = np.linspace(0.05, 0.3, N_bins)
    energy_bins = np.append(energy_bins, 1)
    c_bins = np.linspace(0.0, 0.4, N_bins)
    c_bins = np.append(c_bins, 1)
    s_bins = np.linspace(0, 1.5, N_bins-1) #set extra limits on both sides
    
    s_bins = np.append(s_bins, 2.2)
    s_bins = np.append(-1.5, s_bins)
    a_bins = np.linspace(0.4, 1.6, N_bins-1) #set extra limits on both sides
    a_bins = np.append(a_bins, 5)
    a_bins = np.append(-1., a_bins)
    w_bins = np.linspace(-3, -1, N_bins-1) #set extra limits on both sides
    w_bins = np.append(w_bins, 0)
    w_bins = np.append(-5, w_bins)
    
    mass_mid = np.zeros(N_bins)
    accretion_mid = np.zeros(N_bins)
    energy_mid = np.zeros(N_bins)
    mass_mid[:-1] = (mass_bins[:-2] + mass_bins[1:-1])/2
    accretion_mid[:-1] = (accretion_bins[:-2] + accretion_bins[1:-1])/2
    energy_mid[:-1] = (energy_bins[:-2] + energy_bins[1:-1])/2
    mass_mid[-1] = mass_bins[-2]
    accretion_mid[-1] = accretion_bins[-2]
    energy_mid[-1] = energy_bins[-2]
    
    c_mid = np.zeros(N_bins)
    s_mid = np.zeros(N_bins)
    a_mid = np.zeros(N_bins)
    w_mid = np.zeros(N_bins)

    c_mid[:-1] = (c_bins[:-2] + c_bins[1:-1])/2
    s_mid[1:-1] = (s_bins[1:-2] + s_bins[2:-1])/2
    a_mid[1:-1] = (a_bins[1:-2] + a_bins[2:-1])/2
    w_mid[1:-1] = (w_bins[1:-2] + w_bins[2:-1])/2
    c_mid[-1] = c_bins[-2]
    s_mid[[0,-1]] = [s_bins[1], s_bins[-2]]
    a_mid[[0,-1]] = [a_bins[1], a_bins[-2]]
    w_mid[[0,-1]] = [w_bins[1], w_bins[-2]]
    
    bins = np.vstack((accretion_bins, mass_bins, energy_bins, c_bins,
                      s_bins, a_bins, w_bins))
    bin_type = np.array(["accretion", "mass", "energy", 
                         "concentration", "symmetry", "alignment", "centroid"])
    mids = np.vstack((accretion_mid, mass_mid, energy_mid, 
                      c_mid, s_mid, a_mid, w_mid))
    flm.mids = mids
    for i in range(7):
        sp.stack_fixed(flm, bin_type[i], bins[i,:], dim="2D")
        sp.stack_fixed(flm, bin_type[i], bins[i,:])
        if bin_type[i] == "mass":
           split_data = np.log10(getattr(flm,"M200m"))
        else:
            split_data = getattr(flm, bin_type[i])
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
        setattr(flm, bin_type[i]+ "_density_DM_2D", stacked_data) #currently only stacks one projection, not all
        setattr(flm, bin_type[i]+ "_log_DM_2D", log)
    
    
def fit_models(split):
    params = np.zeros((N_bins, 8))
    profile = getattr(flm, split+"_density_DM")
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
        # plt.loglog(rad_mid, flm.accretion_density_DM[i,:]/rho_crit, label="True")
        # #plt.loglog(rad_mid, 10**test_data, label="Test")
        # plt.loglog(rad_mid, 10**test_data2/rho_crit, label="Fit", linestyle="--")
        # plt.legend()
        # plt.show()
    return params


def fit_log_models(split):
    params = np.zeros((N_bins, 8))
    profile = getattr(flm, split+"_density_DM")
    log_profile = getattr(flm, split+"_log_DM")
    for i in range(N_bins):
        guess = np.array([1, 0.15, 1.5, 0.06, 6, 84, 1.5, 0.9], dtype=np.float32)
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
        # plt.loglog(rad_mid, flm.concentration_density_DM[i,:]/rho_crit, label="True")
        #plt.loglog(rad_mid, 10**test_data, label="Test")
        # plt.loglog(rad_mid, 10**test_data2/rho_crit, label="Fit", linestyle="--")
        # plt.legend()
        # plt.show()
    return params
 
    
def project_model(params):
    R_max = 5
    projected_density = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        popt = params[i,:]
        for j in range(N_rad):
            R = rad_mid[j]
            integrand = lambda r: 10**(density_model(np.log10(r), popt[0], popt[1], popt[2], popt[3],
                                   popt[4], popt[5], popt[6], popt[7])) * r / np.sqrt(r**2 - R**2)
            projected_density[i,j], err = integrate.quad(integrand, R, R_max, 
                                                         epsrel=1e-40) 
    return projected_density


def find_sort_R(array):
    R_sp, second = dr.depth_cut(rad_mid, array,
                                cut=-1, second_caustic="y")
    second_mask = np.where(np.isfinite(second))[0]
    for i in range(len(second_mask)):
        index = second_mask[i]
        if R_sp[index] < second[index]:
            larger = second[index]
            #smaller = R_sp[index]
            R_sp[index] = larger
            #second[index] = smaller
    return R_sp#, second


def plot_projection(params, projected_density, split, R_sp_model, R_sp_2D):
    profile_3D = getattr(flm, split+"_density_DM")
    profile_2D = getattr(flm, split+"_density_DM_2D")
    log_3D = getattr(flm, split+"_log_DM")
    log_2D = getattr(flm, split+"_log_DM_2D")
    for i in range(N_bins):
        popt = params[i,:]
        fit_3D = 10**density_model(np.log10(rad_mid), popt[0], popt[1], 
                                   popt[2], popt[3], popt[4],
                                   popt[5], popt[6], popt[7])
        #log_fit_3D = np.gradient(fit_3D, np.log10(rad_mid))
        log_fit_3D = log_grad_model(np.log10(rad_mid), popt[0], popt[1], 
                               popt[2], popt[3], popt[4],
                               popt[5], popt[6], popt[7])
        # fit_2D = projected_density_model(np.log10(rad_mid), popt[0], popt[1], 
        #                        popt[2], popt[3], popt[4],
        #                        popt[5], popt[6], popt[7])
        log_fit_2D = np.gradient(np.log10(projected_density[i,:]), np.log10(rad_mid))
        
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
        ax[1].set_ylim(ylim)
        plt.xlabel("r/$R_{200m}$")
        ax[1].set_ylabel(r"$\rm{d} \ln \rho_{DM} / \rm{d} \ln r$")
        ax[1].legend()
        # plt.savefig("projection_effect_splashback.png")
        plt.show()
        

def compare_quantities(stack):
    params = fit_log_models(stack)
    projected_density = project_model(params)
    projected_model_log_DM = sp.log_gradients(rad_mid, projected_density)
    #Projected splashback model from 3D
    R_sp_model = find_sort_R(projected_model_log_DM)
    #Splashback feature in 2D from DM
    R_sp_2D = find_sort_R(getattr(flm, stack+"_log_DM_2D"))
    #"True" splashback radius
    R_sp_3D = find_sort_R(getattr(flm, stack+"_log_DM")) 
    
    plot_projection(params, projected_density, stack, R_sp_model, R_sp_2D)
    print(params)
    
    #Compare the splashback from 2D profiles with projected model
    plt.figure()
    cm = plt.cm.get_cmap('autumn')
    plt.scatter(R_sp_2D, R_sp_model, edgecolor="k", 
                c=flm.mids[0,:], cmap=cm)
    xlim = (0.7,1.62)
    ylim = plt.gca().get_ylim()
    plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r"$R_{\rm{SP, data}}/R_{\rm{200m}}$")
    plt.ylabel(r"$R_{\rm{SP, model}} /R_{\rm{200m}}$")
    plt.show()
    
    #Compare splashback from 3D with projected model
    # plt.figure()
    # plt.scatter(R_sp_3D, R_sp_model,edgecolor="k", 
    #             c=flm.mids[0,:], cmap=cm)
    # plt.plot(xlim, xlim, color="k", alpha=0.6, linestyle="--")
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # plt.xlabel(r"$R_{\rm{SP, 3D}}/R_{\rm{200m}}$")
    # plt.ylabel(r"$R_{\rm{SP, model}} /R_{\rm{200m}}$")
    # plt.show()
    
    # R_sp_EM = find_sort_R(getattr(flm, stack+"_log_EM"))
    # R_sp_SZ = find_sort_R(getattr(flm, stack+"_log_SZ"))
    # R_sp_WL = find_sort_R(getattr(flm, stack+"_log_WL"))
    # fig, ax = plt.subplots(nrows=3, ncols=1,
    #                        sharex=True, sharey=True,
    #                        figsize=(3.3,5),
    #                        gridspec_kw={'hspace' : 0, 'wspace' : 0})
    # ax[0].scatter(R_sp_model, R_sp_EM,
    #               edgecolor="k", color="gold", s=75)
    # ax[1].scatter(R_sp_model, R_sp_SZ,
    #               edgecolor="k", color="cyan", s=75)
    # ax[2].scatter(R_sp_model, R_sp_WL,
    #               edgecolor="k", color="orchid", s=75)
    # xlim = ax[2].get_xlim()
    # ax[0].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[1].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[2].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[0].set_xlim(xlim)
    # plt.xlabel(r"$R_{\rm{SP,model}}/R_{\rm{200m}}$")
    # ax[0].set_ylabel(r"$R_{\rm{SP,EM}}/R_{\rm{200m}}$")
    # ax[1].set_ylabel(r"$R_{\rm{SP,SZ}}/R_{\rm{200m}}$")
    # ax[2].set_ylabel(r"$R_{\rm{SP,WL}}/R_{\rm{200m}}$")
    # ax[0].set_title(stack)
    # plt.show()
    
    # fig, ax = plt.subplots(nrows=3, ncols=1,
    #                        sharex=True, sharey=True,
    #                        figsize=(3.3,5),
    #                        gridspec_kw={'hspace' : 0, 'wspace' : 0})
    # ax[0].scatter(R_sp_3D, R_sp_EM,
    #               edgecolor="k", color="gold", s=75)
    # ax[1].scatter(R_sp_3D, R_sp_SZ,
    #               edgecolor="k", color="cyan", s=75)
    # ax[2].scatter(R_sp_3D, R_sp_WL,
    #               edgecolor="k", color="orchid", s=75)
    # xlim = ax[2].get_xlim()
    # ax[0].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[1].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[2].plot(xlim, xlim, color="k", linestyle="--", alpha=0.6)
    # ax[0].set_xlim(xlim)
    # plt.xlabel(r"$R_{\rm{SP,3D}}/R_{\rm{200m}}$")
    # ax[0].set_ylabel(r"$R_{\rm{SP,EM}}/R_{\rm{200m}}$")
    # ax[1].set_ylabel(r"$R_{\rm{SP,SZ}}/R_{\rm{200m}}$")
    # ax[2].set_ylabel(r"$R_{\rm{SP,WL}}/R_{\rm{200m}}$")
    # ax[0].set_title(stack)
    # plt.show()
    
    
def scatter_compare_sw():
    stack = "symmetry"
    params = fit_log_models(stack)
    projected_density = project_model(params)
    projected_model_log_DM = sp.log_gradients(rad_mid, projected_density)
    #Projected splashback model from 3D
    R_sp_model_s = find_sort_R(projected_model_log_DM)
    #Splashback feature in 2D from DM
    # R_sp_2D_s = find_sort_R(getattr(flm, stack+"_log_DM_2D"))
    #"True" splashback radius
    # R_sp_3D_s = find_sort_R(getattr(flm, stack+"_log_DM"))
    R_sp_EM_s = find_sort_R(getattr(flm, stack+"_log_EM"))
    R_sp_SZ_s = find_sort_R(getattr(flm, stack+"_log_SZ"))
    R_sp_WL_s = find_sort_R(getattr(flm, stack+"_log_WL"))
    
    stack = "centroid"
    params = fit_log_models(stack)
    projected_density = project_model(params)
    projected_model_log_DM = sp.log_gradients(rad_mid, projected_density)
    #Projected splashback model from 3D
    R_sp_model_w = find_sort_R(projected_model_log_DM)
    #Splashback feature in 2D from DM
    # R_sp_2D_w = find_sort_R(getattr(flm, stack+"_log_DM_2D"))
    #"True" splashback radius
    # R_sp_3D_w = find_sort_R(getattr(flm, stack+"_log_DM")) 
    R_sp_EM_w = find_sort_R(getattr(flm, stack+"_log_EM"))
    R_sp_SZ_w = find_sort_R(getattr(flm, stack+"_log_SZ"))
    R_sp_WL_w = find_sort_R(getattr(flm, stack+"_log_WL"))
    
    fig, ax = plt.subplots(nrows=3, ncols=2,
                           sharex=True,
                           sharey='row',
                           gridspec_kw={'hspace' : 0, 'wspace' : 0},
                           figsize=(3.3,4))
    size=50
    cs = ax[0,0].scatter(R_sp_model_s, 
                         R_sp_EM_s,
                         c=flm.mids[4,:], cmap=plt.cm.get_cmap("winter"),
                         edgecolor="k",
                         s=size,
                         marker="*",
                         label="$s$")
    ax[1,0].scatter(R_sp_model_s, 
                    R_sp_SZ_s,
                    c=flm.mids[4,:], cmap=plt.cm.get_cmap("winter"),
                    edgecolor="k",
                    s=size,
                    marker="*")
    ax[2,0].scatter(R_sp_model_s, 
                    R_sp_WL_s,
                    c=flm.mids[4,:], cmap=plt.cm.get_cmap("winter"),
                    edgecolor="k",
                    s=size,
                    marker="*")

    cw = ax[0,1].scatter(R_sp_model_w, 
                         R_sp_EM_w,
                         c=flm.mids[6,:], cmap=plt.cm.get_cmap("spring"),
                         edgecolor="k",
                         s=size,
                         marker="P",
                         label=r"$\log \langle w \rangle$")
    ax[1,1].scatter(R_sp_model_w, 
                    R_sp_SZ_w,
                    c=flm.mids[6,:], cmap=plt.cm.get_cmap("spring"),
                    edgecolor="k",
                    s=size,
                    marker="P")
    ax[2,1].scatter(R_sp_model_w, 
                    R_sp_WL_w,
                    c=flm.mids[6,:], cmap=plt.cm.get_cmap("spring"),
                    edgecolor="k",
                    s=size,
                    marker="P")

    cbaxes2 = fig.add_axes([0.45, 0.7, 0.02, 0.08]) 
    fig.colorbar(cs, cax=cbaxes2, label="$s$")
    cbaxes4 = fig.add_axes([0.82, 0.7, 0.02, 0.08]) 
    fig.colorbar(cw, cax=cbaxes4, label=r"$\log \langle w \rangle$")

    # ax[0,0].legend()
    # ax[0,1].legend()
    ylim0 = ax[0,0].get_ylim()
    ylim1 = ax[1,0].get_ylim()
    ylim2 = ax[2,0].get_ylim()
    ylim1 = (ylim1[0], 1.38)
    xlim = (0.67,1.1)
    
    for axes in ax.flatten():
        axes.plot(xlim, xlim,
                  linestyle="--",
                  color="k",
                  alpha=0.6,
                  label="$y=x$")
    
    ax[2,0].legend()
    ax[0,0].set_ylim(ylim0)
    ax[1,0].set_ylim(ylim1)
    ax[2,0].set_ylim(ylim2)
    ax[1,0].set_xlim(xlim)
    ax[0,0].set_ylabel("$R_{\\rm{SP,EM}} / R_{\\rm{200m}}$")
    ax[1,0].set_ylabel("$R_{\\rm{SP,SZ}} / R_{\\rm{200m}}$")
    ax[2,0].set_ylabel("$R_{\\rm{SP,WL}} / R_{\\rm{200m}}$")
    
    plt.subplots_adjust(bottom=0.1)
    plt.text(0.45, 0.01, "$R_{\\rm{SP,model}} / R_{\\rm{200m}}$", 
             transform=fig.transFigure)
    
    # filename = "splashback_data/flamingo/plots/compare_Rsp_morph.png"
    # plt.savefig(filename, dpi=300)
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
    N_bins = 10
    stack_profiles(N_bins)
    # compare_quantities("accretion")
    compare_quantities("mass")
    # compare_quantities("energy")
    # compare_quantities("concentration")
    # compare_quantities("symmetry")
    # compare_quantities("alignment")
    # compare_quantities("centroid")
    
    # scatter_compare_sw()

    
    
    
    
    
    
    
    
    
    
    
    
    