import numpy as np
import matplotlib.pyplot as plt
import splashback as sp
import determine_radius as dr

plt.style.use("mnras.mplstyle")
box = "L1000N1800"

def stack_find_splash(array, stack_criteria, N_bins=10):
    N_clusters = array.shape[0]
    N_rad = array.shape[1]
    bins = sp.bin_fixed_width(N_bins, 
                              stack_criteria,
                              min_per_bin=15)
    not_nan = np.where(np.isfinite(stack_criteria)==True)[0]
    bins_sort = np.digitize(stack_criteria[not_nan], bins)
    if N_clusters == 3*len(stack_criteria):
        #for when stacking projected profiles with 3D quantities
        bins_sort = np.hstack((bins_sort, bins_sort, bins_sort))
    stacked_profiles = np.zeros((N_bins, N_rad))
    for i in range(N_bins):
        bin_mask = np.where(bins_sort == i+1)[0]
        to_stack = array[bin_mask,:]
        stacked_profiles[i,:] = sp.stack_data(to_stack)
    log = sp.log_gradients(rad_mid, stacked_profiles)
    return bins, log


def bin_and_fit(d, stacking_criteria):
    bins, log = stack_find_splash(d.EM_median, getattr(d,stacking_criteria))
    R = dr.depth_cut(rad_mid, log)
    setattr(d, stacking_criteria +"_bins_EM", bins)
    setattr(d, "R_EM_" + stacking_criteria, R)
    setattr(d, "log_EM" + stacking_criteria, log)
    
    bins, log = stack_find_splash(d.SZ_median, getattr(d,stacking_criteria))
    R = dr.depth_cut(rad_mid, log, cut=-1)
    setattr(d, stacking_criteria +"_bins_SZ", bins)
    setattr(d, "R_SZ_" + stacking_criteria, R)
    setattr(d, "log_SZ" + stacking_criteria, log)
    
    bins, log = stack_find_splash(d.WL_median, getattr(d,stacking_criteria))
    R = dr.depth_cut(rad_mid, log, cut=-1)
    setattr(d, stacking_criteria +"_bins_WL", bins)
    setattr(d, "R_WL_" + stacking_criteria, R)
    setattr(d, "log_WL" + stacking_criteria, log)


# def compare_groups(N_stack, split_data,
#                    sim_type="hydro",
#                    hydro_input=flm.WL_median,
#                    label=r"$M_{200m}$"):
#     # if sim_type == "DMO":
#     #     threeD = np.vstack((mcs.DMO_3D, mcs.DMO_3D, mcs.DMO_3D))
#     #     twoD = mcs.DMO_2D
#     #     rad_mid_2D = rad_mid_short
#     #     rad_mid_3D = rad_mid_long
        
#     # elif sim_type == "hydro":
#     threeD = np.vstack((flm.DM_log, flm.DM_log, flm.DM_log))
#     twoD = hydro_input
    
#     stack_3D, split_data_mid = sp.stack_fixed_width(N_stack, split_data, threeD)
#     stack_2D, split_data_mid = sp.stack_fixed_width(N_stack, split_data, twoD)
    
#     RSP_stack_2D = dr.standard(rad_mid, stack_2D)
#     RSP_stack_3D = dr.standard(rad_mid, stack_3D)
    
    
#     # for i in range(N_stack):
#     #     plt.figure()
#     #     plt.semilogx(rad_mid, stack_2D[i,:], color="blue", 
#     #                   linestyle="--", label="2D")
#     #     plt.semilogx(rad_mid, stack_3D[i,:], color="blue",
#     #                   label="3D")
#     #     ylim = plt.gca().get_ylim()
#     #     plt.plot([RSP_stack_2D[i], RSP_stack_2D[i]], ylim,
#     #               color="blue", linestyle="--")
#     #     plt.plot([RSP_stack_3D[i], RSP_stack_3D[i]], ylim,
#     #               color="blue")
#     #     plt.ylim(ylim)
#     #     plt.xlabel(r"$r/R_{200\rm{m}}$")
#     #     plt.ylabel(r"$d \log \rho / d \log r$")
#     #     plt.legend()
#     #     #plt.savefig("example_profile_bin10.png", dpi=300)
#     #     plt.show()
        
#     #m, c = np.polyfit(RSP_stack_3D, RSP_stack_2D, 1)
#     #a = 1
#     b = 0
#     a = np.sqrt(np.mean((RSP_stack_2D/RSP_stack_3D)**2))
#     xlim = np.array([0.5,1.5])
#     y = a*xlim + b
    
#     distance = abs((a*RSP_stack_3D - RSP_stack_2D) / np.sqrt(a**2 +1))
#     avg_scatter = np.mean(distance)
#     print(avg_scatter)
    
#     fig = plt.figure()
#     cm = plt.cm.get_cmap('rainbow')
#     plt.scatter(RSP_stack_3D, RSP_stack_2D, 
#                 c=split_data_mid, edgecolor="k", cmap=cm, s=75)
#     plt.plot(xlim, y, linestyle="--", color="k")
#     plt.xlabel(r"$R_{\rm{SP,DM}} / R_{\rm{200m}}$")
#     plt.ylabel(r"$R_{\rm{SP,2D}} / R_{\rm{200m}}$")
#     plt.xlim(xlim)
#     #plt.title("Hydro - EM")
#     plt.ylim((0.5,1.5))
#     cbaxes = fig.add_axes([0.2, 0.72, 0.02, 0.2]) 
#     cbar = plt.colorbar(cax=cbaxes, label=label)
#     #cbar.set_ticks([1e5, 2e5])
#     #cbar.set_ticklabels([r"10$^{15}$", r"2$\times$10$^{15}$"])
#     # filename = "stacking_EM_accretion.png"
#     # plt.savefig(filename, dpi=300)
#     plt.show()


N_rad = 71
log_radii = np.linspace(-1, 0.9, N_rad)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

flm_HF = sp.flamingo(box, "HF")
flm_HF.read_properties()
flm_HF.read_2D()
bin_and_fit(flm_HF, "energy_ratio")
bins, log = stack_find_splash(flm_HF.DM_density_3D, flm_HF.energy_ratio)
R_sp = dr.depth_cut(rad_mid, log)

plt.scatter(R_sp, flm_HF.R_EM_energy_ratio,
            marker="*", color="gold", label="EM")
plt.scatter(R_sp, flm_HF.R_SZ_energy_ratio,
            marker="o", color="cyan", label="SZ")
plt.scatter(R_sp, flm_HF.R_WL_energy_ratio,
            marker="v", color="purple", label="WL")
plt.show()
    
# flm_HWA = sp.flamingo(box, "HWA")
# flm_HWA.read_properties()
# flm_HWA.read_2D()
    
# flm_HSA = sp.flamingo(box, "HSA")
# flm_HSA.read_properties()
# flm_HSA.read_2D()