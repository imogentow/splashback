import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
import splashback as sp
import determine_radius as dr

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
    

def project_model(rad_mid, popt):
    R_max = 5
    N_rad = len(rad_mid)
    projected_density = np.zeros(N_rad)
    for j in range(N_rad):
        R = rad_mid[j]
        integrand = lambda r: 10**(density_model(np.log10(r), popt[0], popt[1],
                                                 popt[2], popt[3], popt[4], 
                                                 popt[5], popt[6], popt[7])) * r / np.sqrt(r**2 - R**2)
        projected_density[j], err = integrate.quad(integrand, R, R_max) 
    return 2 * projected_density


def mass_function(popt, radii):
    """For a given set of parameters to fit a DK DM density model, returns 
    an array giving the mass within a series of radii by integrating the
    density profile."""
    N_rad = len(radii)
    integrand = lambda r: 10**density_model(np.log10(r),popt[0], popt[1],
                                            popt[2], popt[3], popt[4], 
                                            popt[5], popt[6], popt[7]) *4*np.pi*r**2
    mass_function = np.zeros(N_rad)
    for i in range(N_rad):
        R = radii[i]
        mass_function[i], _ = integrate.quad(integrand, 0, R)
    return mass_function


def project_mass(radii, popt):
    N_rad = len(radii)
    proj_mass = np.zeros(N_rad)
    integrand = lambda R, z: R * 10**density_model(np.log10(np.sqrt(R**2+z**2)),
                                                   popt[0], popt[1], popt[2], 
                                                   popt[3], popt[4], popt[5], 
                                                   popt[6], popt[7])
    for i in range(N_rad):
        R = radii[i]
        proj_mass[i], _ = integrate.dblquad(integrand, 0, r_max, 0, R)
    volume = np.pi * (radii[1:]**2 - radii[:-1]**2) * 2 * r_max
    shell_mass = 4* np.pi * (proj_mass[1:] - proj_mass[:-1])
    shell_density = shell_mass / volume
    return shell_density


def organise_particles(N_particles, params):
    N_rad = 100
    #to account for the cylindrical region being larger than the sphere
    r_particle_max = np.sqrt(2) * r_max 
    radii = np.logspace(-4.5, np.log10(r_particle_max),  N_rad)
    mass = mass_function(params, radii)
    halo_mass = mass[-1] #maximum mass of halo
    m_particle = halo_mass / N_particles #mass of one particle
    #Gives number of particles within each radius
    no_particles = mass / m_particle
    # print(no_particles)
    particles_interp = interpolate.interp1d(no_particles, radii)
    #radius of each particle
    rad_particles = particles_interp(np.arange(1, N_particles+1))
    #random angular values
    #randomly sample costheta values and scales to between 0 and 2pi - might need to change to sin?
    phi_particles = np.arccos(np.random.rand(N_particles)*2-1)
    theta_particles = np.random.rand(N_particles) * 2* np.pi 
    
    x = rad_particles * np.cos(theta_particles) * np.sin(phi_particles)
    y = rad_particles * np.sin(theta_particles) * np.sin(phi_particles)
    z = rad_particles * np.cos(phi_particles)
    # print(rad_particles)
    return x, y, z, m_particle


def measure_density(x, y, z, m_particle):
    R = np.sqrt(x**2 + y**2) #projected radii
    rad_particles = np.sqrt(x**2 + y**2 + z**2)
    
    N_bins = 75
    shell_radii = np.logspace(-1, 0.7, num=N_bins+1)
    # shell_rad_mid = (shell_radii[1:] + shell_radii[:-1]) / 2 
    shell_volume_3D = 4/3*np.pi * (shell_radii[1:]**3 - shell_radii[:-1]**3)
    shell_rho_3D = np.zeros(N_bins)
    shell_volume_2D = np.pi * (shell_radii[1:]**2 - shell_radii[:-1]**2) * 2 * r_max
    shell_rho_2D = np.zeros(N_bins)
    
    bin_info_3D = np.digitize(rad_particles, shell_radii)
    bin_info_2D = np.digitize(R, shell_radii)
    for i in range(N_bins):
        mask_3D = np.where(bin_info_3D == i+1)[0]
        shell_rho_3D[i] = len(mask_3D) * m_particle / shell_volume_3D[i]
        
        mask_2D = np.where((bin_info_2D == i+1) & (z**2 < r_max**2))[0]
        shell_rho_2D[i] = len(mask_2D) * m_particle / shell_volume_2D[i]
        # print(len(mask_3D), len(mask_2D))
    return shell_rho_3D, shell_rho_2D, shell_radii


def calculate_Rsp(params):
    x, y, z, m_particle = organise_particles(N_particles, params)
    shell_rho_3D, shell_rho_2D, shell_radii = measure_density(x, y, z, m_particle)
    # shell_radii = np.logspace(-1, 0.7, num=51)
    shell_rad_mid = (shell_radii[1:] + shell_radii[:-1]) / 2 
    r_model = shell_rad_mid #np.logspace(-1, 0.7, num=N_bins+1)
    rho_model = 10**density_model(np.log10(r_model), params[0], params[1], params[2],
                                  params[3], params[4], params[5], params[6],
                                  params[7])
    log_rho_model = np.gradient(np.log10(rho_model), np.log10(r_model))
    log_rho_data = sp.log_gradients(shell_rad_mid, shell_rho_3D, window=19, order=4)
    R_model_3D = dr.depth_cut(r_model, log_rho_model, cut=-1)
    R_data_3D = dr.depth_cut(shell_rad_mid, log_rho_data, cut=-1)
    
    # proj_rho = project_model(r_model, params) / (2*r_max)
    proj_rho = project_mass(shell_radii, params)
    log_proj_model = np.gradient(np.log10(proj_rho), np.log10(r_model))
    log_proj_data = sp.log_gradients(shell_rad_mid, shell_rho_2D, window=19)
    R_model_2D = dr.depth_cut(r_model, log_proj_model, cut=-1)
    R_data_2D = dr.depth_cut(shell_rad_mid, log_proj_data, cut=-1)
    
    # plt.loglog(r_model, rho_model/rho_crit, label="Model")
    # plt.loglog(shell_rad_mid, shell_rho_3D/rho_crit, label="Data")
    # plt.loglog(r_model, proj_rho/rho_crit, label="Model, 2D")
    # plt.loglog(shell_rad_mid, shell_rho_2D/rho_crit, label="Data, 2D")
    # plt.legend()
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    # plt.ylabel("$\\rho/\\rho_{\\rm{crit}}$")
    # plt.title(i)
    # plt.show()
    
    # plt.semilogx(r_model, log_rho_model, label="Model")
    # plt.semilogx(shell_rad_mid, log_rho_data, label="Data")
    # plt.semilogx(r_model, log_proj_model, label="Model, 2D")
    # plt.semilogx(shell_rad_mid, log_proj_data, label="Data, 2D")
    # plt.legend()
    # plt.xlabel("$r/R_{\\rm{200m}}$")
    # plt.ylabel("$d \log \\rho / d \log r$")
    # plt.title(i)
    # plt.show()
    print(R_model_3D, R_model_2D, R_data_3D, R_data_2D)
    return R_model_3D, R_model_2D, R_data_3D, R_data_2D

r_max = 5
N_particles = int(1e7)
params = np.array([[1.67998827e+01, 9.21390688e-02, 1.60806959e+00, 1.86426902e-01,
                    1.34069534e+01, 4.06623627e+00, 2.43768238e+02, 7.51250017e-01],
                    [1.49640810e+01, 9.72273080e-02, 1.40230214e+00, 1.87824494e-01,
                    9.92951188e+00, 3.21010879e+00, 3.03825187e+00, 8.41443114e-01],
                    [1.85237441e+01, 9.63444438e-02, 1.19037427e+00, 1.63468406e-01,
                    7.11749558e+00, 3.41806641e+00, 1.44687145e+04, 7.44231482e-01],
                    [1.55848782e+01, 1.06288095e-01, 1.07913721e+00, 1.43961558e-01,
                    5.82956023e+00, 3.91176225e+00, 2.14354633e+01, 8.33683109e-01],
                    [1.85434733e+01, 1.33736123e-01, 9.82119946e-01, 1.41347628e-01,
                    5.42723095e+00, 3.81912596e+00, 3.38631775e+04, 8.76823032e-01],
                    [1.81569241e+01, 1.59585807e-01, 9.27353754e-01, 9.86920805e-02,
                    4.47077140e+00, 4.25493994e+00, 2.10855782e+04, 8.52793824e-01],
                    [1.39449423e+01, 2.55846754e-01, 9.56125152e-01, 5.89482058e-02,
                    3.81839701e+00, 5.67663368e+00, 2.41601926e+00, 1.15451420e+00],
                    [5.03561193e+00, 2.54645108e+07, 8.16123020e-01, 1.14542781e-02,
                    3.45974238e+00, 4.86878238e+00, 1.24296133e+06, 8.89713675e-01],
                    [1.87784759e+01, 1.14587715e-01, 1.44369486e+00, 1.67721507e-01,
                    4.13676097e+00, 5.65116654e+00, 3.91597088e+04, 8.43989931e-01],
                    [1.45873507e+01, 1.19309521e-01, 1.32926316e+00, 1.74622242e-01,
                    4.35118779e+00, 4.71282250e+00, 1.85251886e+00, 1.08014347e+00],
                    [1.44387976e+01, 1.18615115e-01, 1.63141520e+00, 1.70225772e-01,
                    3.77455257e+00, 8.11079485e+00, 1.07894946e+00, 1.26549350e+00],
                    [1.51086311e+01, 1.27039308e-01, 1.37932003e+00, 1.76676851e-01,
                    4.12197009e+00, 5.53444031e+00, 9.72959856e+00, 9.04403335e-01],
                    [1.46435355e+01, 1.28337439e-01, 1.58160663e+00, 1.64284434e-01,
                    3.61130868e+00, 8.14305260e+00, 2.79856484e+00, 1.09297024e+00],
                    [1.80918646e+01, 1.37191441e-01, 1.22076728e+00, 2.01674314e-01,
                    5.13139706e+00, 4.21586171e+00, 1.21180814e+04, 8.43041506e-01],
                    [1.46634528e+01, 1.30380824e-01, 1.65725152e+00, 1.59122751e-01,
                    3.42028552e+00, 9.92172835e+00, 3.14150687e+00, 1.07014603e+00],
                    [1.42936385e+01, 1.48352565e-01, 1.18064483e+00, 2.25372125e-01,
                    5.50118275e+00, 4.16446390e+00, 1.21080137e+00, 1.27364110e+00],
                    [1.86676246e+01, 1.48356490e-01, 1.43409630e+00, 1.86989654e-01,
                    3.84096019e+00, 8.36830621e+00, 5.35094622e+04, 9.64178140e-01],
                    [1.43724605e+01, 1.56069974e-01, 1.24969857e+00, 1.97510882e-01,
                    4.07900708e+00, 6.37079521e+00, 2.08538467e+00, 1.22304988e+00]])
N_runs = params.shape[0]
# params = [1e8, -2, 0, 0, 0, 0, 0, 0]
# N_runs = 1

R_model_3D = np.zeros(N_runs)
R_model_2D = np.zeros(N_runs)
R_data_3D = np.zeros(N_runs)
R_data_2D = np.zeros(N_runs)
for i in range(N_runs):
    R_model_3D[i], R_model_2D[i], R_data_3D[i], R_data_2D[i] = calculate_Rsp(params[i,:])
    
plt.scatter(R_model_2D, R_data_2D, edgecolor="k")
xlim = plt.gca().get_xlim()
plt.plot(xlim, xlim, linestyle="--", color="grey")
plt.xlim(xlim)
plt.xlabel("$R_{\\rm{SP,model}}$")
plt.ylabel("$R_{\\rm{SP,data}}$")
plt.show()