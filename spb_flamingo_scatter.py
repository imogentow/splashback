import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
from scipy import interpolate
import determine_radius as dr

plt.style.use("mnras.mplstyle")

#CHECK if radii are given in /h units, everything will be scaled the same but worth checking

G = 6.67e-11
H0 = 67.77 * 1000 / 3.09e22
rho_crit = 3 * H0**2 / (8 * np.pi * G )
N_clusters = 400

N_interp = 102
log_interp = np.linspace(-2, 0.7, N_interp)
x_interp = 10**log_interp[3:-3]

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins)
rad_mid_short = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

N_bins = 71
log_radii = np.linspace(-2, 0.7, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

unit_conversion_mass = 1.99e30 #Msol to kg
unit_conversion_distance = 3.09e22 #Mpc to m
unit_conversion_density = unit_conversion_mass / unit_conversion_distance**3

def make_scaled_profiles(density, r200, radius):
    scaled_radii = rad_mid * r200 / radius #r/r_x
    rad_lims = scaled_radii[[0,-1]]
    
    density = density*unit_conversion_density #/ r200**3 #Msol / Mpc^3 to kg/m^3
    scaled_density = density / rho_crit * scaled_radii**2 #what is units of density, needs correcting here
    
    interp = interpolate.interp1d(scaled_radii, scaled_density,
                                  kind='cubic')
    interp_mask = np.where((x_interp > rad_lims[0]) & (x_interp < rad_lims[1]))[0]
    interp_density = interp(x_interp[interp_mask]) #only interpolates when x_interp is in the right range
    
    return_density = np.zeros(N_interp-6)
    return_density[:] = np.nan #will set values to nan that can't be interpolated
    return_density[interp_mask] = interp_density
    
    return return_density


def find_range(density, radii):
    N_rad = len(radii)
    
    median = np.zeros(N_rad)
    upper = np.zeros(N_rad)
    lower = np.zeros(N_rad)
    for i in range(N_rad):
        median[i] = np.nanmedian(density[:,i])
        upper[i] = np.nanpercentile(density[:,i], 75)
        lower[i] = np.nanpercentile(density[:,i], 25)
        
    return median, upper, lower


box = "L1000N1800"
flm = sp.flamingo(box)

log_DM_density = sp.log_gradients(rad_mid_short, flm.DM_density_3D)
DM_density = np.genfromtxt("splashback_data/flamingo/flm_L1000N1800_3D_DM_density_long_all.csv",
                           delimiter=",")

#scale radii
R_200m = np.genfromtxt("splashback_data/flamingo/R200m.csv", delimiter=",")
R_500c = np.genfromtxt("splashback_data/flamingo/R500c.csv", delimiter=",")
R_sp = dr.depth_cut(rad_mid_short, log_DM_density) * R_200m #Mpc

density_R200m = np.zeros((N_clusters, N_interp-6))
density_R500c = np.zeros((N_clusters, N_interp-6))
density_Rsp = np.zeros((N_clusters, N_interp-6))

for i in range(N_clusters):
    density_R200m[i,:] = make_scaled_profiles(DM_density[i,:], R_200m[i], R_200m[i])
    density_R500c[i,:] = make_scaled_profiles(DM_density[i,:], R_200m[i], R_500c[i])
    density_Rsp[i,:] = make_scaled_profiles(DM_density[i,:], R_200m[i], R_sp[i])

r200m_median, r200m_upper, r200m_lower = find_range(density_R200m, x_interp)
r500c_median, r500c_upper, r500c_lower = find_range(density_R500c, x_interp)
rsp_median, rsp_upper, rsp_lower = find_range(density_Rsp, x_interp)

plt.figure()

plt.semilogy(x_interp, rsp_median, color="slateblue", label="$R_{SP}$")
plt.fill_between(x_interp, rsp_lower, rsp_upper, facecolor="slateblue", edgecolor="none", alpha=0.6)

plt.semilogy(x_interp, r200m_median, color="orangered", label="$R_{200m}$")
plt.fill_between(x_interp, r200m_lower, r200m_upper, facecolor="orangered", edgecolor="none", alpha=0.6)

plt.semilogy(x_interp, r500c_median, color="teal", label="$R_{500c}$")
plt.fill_between(x_interp, r500c_lower, r500c_upper, facecolor="teal", edgecolor="none", alpha=0.6)

plt.xlabel("$r/R_{X}$")
plt.ylabel("$(r/R_{x})^2 \\rho_{\mathrm{DM}}(r) / \\rho_{\mathrm{crit}}$")

plt.legend()
plt.xlim(0,1.5)
# plt.savefig("splashback_data/flamingo/plots/flamingo_scatter_profiles_small.png", dpi=300)
plt.show()


