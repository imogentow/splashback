import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr

plt.style.use("mnras.mplstyle")

def stack_fixed_bins(data, split, split_bins):
    """
    For a given set of bins. Stack the DM and gas density profiles for a 
    given run according to a given stacking criteria. Assigns values with
    appropriate name to obj.
    
    Inputs.
    data: obj, run of choice
    split: str, name of stacking criteria
    split_bins: bins to use to split stacking criteria.
    """
    if split == "mass":
        split_data = np.log10(data.M200m)
    else:
        split_data = getattr(data, split)
    not_nan = np.where(np.isfinite(split_data)==True)[0]
    #will return 0 or len for values outside the range
    bins_sort = np.digitize(split_data[not_nan], split_bins)
    N_bins = len(split_bins)+1
    
    stacked_DMO = np.zeros((N_bins, N_rad))
    stacked_WL = np.zeros((N_bins, N_rad))
    
    print("")
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i)[0]
        stacked_WL[i,:] = sp.stack_data(data.WL_median[not_nan][bin_mask,:])
        stacked_DMO[i,:] = sp.stack_data(data.DMO_median[not_nan][bin_mask,:])
        print(len(bin_mask))
        
    
    log_DMO = sp.log_gradients(rad_mid, stacked_DMO)
    R_SP_DMO, depth_DMO = dr.depth_cut(rad_mid, log_DMO, 
                                     depth_value="y",
                                     cut=-1)
    log_WL = sp.log_gradients(rad_mid, stacked_WL)
    R_SP_WL, depth_WL = dr.depth_cut(rad_mid, log_WL, 
                                     depth_value="y",
                                     cut=-1)
    
    setattr(data, "R_WL_"+split, R_SP_WL)
    setattr(data, split+"_log_WL", log_WL)
    setattr(data, "depth_WL_"+split, depth_WL)
    setattr(data, "R_DMO_"+split, R_SP_DMO)
    setattr(data, split+"_log_DMO", log_DMO)
    setattr(data, "depth_DMO_"+split, depth_DMO)
    
N_rad = 44
log_radii = np.linspace(-1, 0.7, N_rad+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

box = "L1000N1800"
run = "HF"
flm = sp.flamingo(box, run)
flm.read_properties()
flm.DMO_median = np.genfromtxt(flm.path + "_DMO_profiles_10r200m_all.csv", 
                               delimiter=",")
flm.WL_median = np.genfromtxt(flm.path + "_WL_profiles_10r200m_all.csv", 
                              delimiter=",")
N_bins = 10
mass_bins = np.linspace(14.1, 15, N_bins+1)
accretion_bins = np.linspace(1, 4, N_bins+1)
energy_bins = np.linspace(0.1, 0.3, N_bins+1)
stack_fixed_bins(flm, "accretion", accretion_bins)
stack_fixed_bins(flm, "mass", mass_bins)
stack_fixed_bins(flm, "energy", energy_bins)

# fig, ax = plt.subplots(nrows=3, ncols=2,
#                         sharey=True,
#                         figsize=(5,6),
#                         gridspec_kw={'hspace' : 0, 'wspace' : 0})
# cm_DMO = plt.cm.winter(np.linspace(0,1,N_bins+2))
# cm_WL = plt.cm.copper(np.linspace(0,1,N_bins+2))
# for i in range(N_bins+2):
#     ax[0,0].semilogx(rad_mid, flm.accretion_log_WL[i,:],
#                       color=cm_WL[i])
#     ax[0,1].semilogx(rad_mid, flm.accretion_log_DMO[i,:],
#                       color=cm_DMO[i])
#     ax[1,0].semilogx(rad_mid, flm.mass_log_WL[i,:],
#                       color=cm_WL[i])
#     ax[1,1].semilogx(rad_mid, flm.mass_log_DMO[i,:],
#                       color=cm_DMO[i])
#     ax[2,0].semilogx(rad_mid, flm.energy_log_WL[i,:],
#                       color=cm_WL[i])
#     ax[2,1].semilogx(rad_mid, flm.energy_log_DMO[i,:],
#                       color=cm_DMO[i])
# ax[0,0].text(0.05,0.05, "WL", transform=ax[0,0].transAxes)
# ax[0,1].text(0.05,0.05, "DMO", transform=ax[0,1].transAxes)
# ax[1,0].set_ylabel(r"$d \log \rho / d \log r$")
# plt.text(0.5, 0.05, r"$r/R_{\rm{200m}}$", transform=fig.transFigure)
# plt.show()

plt.figure()
plt.scatter(flm.R_WL_accretion, flm.R_DMO_accretion, 
            marker="o", edgecolor="k", label="$\Gamma$")
plt.scatter(flm.R_WL_mass, flm.R_DMO_mass, 
            marker="v", edgecolor="k", label="Mass")
plt.scatter(flm.R_WL_energy, flm.R_DMO_energy,
            marker="*", edgecolor="k", label="$E_{\\rm{kin}} / E_{\\rm{therm}}$")
plt.legend()
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.plot(xlim, ylim, color="k")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r"$R_{\rm{SP, WL}}$")
plt.ylabel(r"$R_{\rm{SP, DM}}$")
plt.show()
