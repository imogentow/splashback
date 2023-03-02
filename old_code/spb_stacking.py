import numpy as np
import matplotlib.pyplot as plt
import splashback
from scipy.stats import spearmanr
import determine_radius as dr

#plt.style.use("mnras.mplstyle")

#Read in data for binning
path = "splashback_data/macsis/"

mass = np.genfromtxt(path + "M_200m_macsis.csv", delimiter=",")
mass_DMO = np.genfromtxt(path + "M_200m_macsis_DMO.csv", delimiter=",")

accretion = np.genfromtxt(path + "accretion_rates.csv", delimiter=",")
accretion_DMO = np.genfromtxt(path + "accretion_macsis_DMO.csv", delimiter=",")

energy_ratio = np.genfromtxt(path + "energy_ratio_200m_macsis.csv", delimiter=",")

#Read MACSIS profile data
mcs = splashback.macsis()
mcs.read_DMO()
mcs.read_semi_DMO()

mass[mcs.exclude] = np.nan
accretion[mcs.exclude] = np.nan
energy_ratio[mcs.exclude] = np.nan
mcs.substructure[mcs.exclude] = np.nan

N_rad_bins_short = 40
log_radii_short = np.linspace(-1, 0.6, N_rad_bins_short+1)
rad_mid_short = (10**log_radii_short[1:] + 10**log_radii_short[:-1]) / 2

N_rad_bins_long = 45
log_radii_long = np.linspace(-1, 0.7, N_rad_bins_long+1)
rad_mid_long = (10**log_radii_long[1:] + 10**log_radii_long[:-1]) / 2


def stack_fixed_N(N_bins, split_data, array):
    """Splits array of profiles into bins and then stacks profiles in each bin.
    N_bins - number of bins to split data into
    split_order - array, matching [0] length of data giving positions of 
                  profiles when arranged according to particular criteria,
                  done before using this function.
    array - data to be split and stacked [0] length should match split_order"""
    N_rad = array.shape[1]
    split_order = np.argsort(split_data)
    
    if array.shape[0] == 1170:
        array1 = array[:390,:]
        array2 = array[390:780,:]
        array3 = array[780:,:]
    
    stacked_profiles = np.zeros((N_bins, N_rad))

    splits = np.array_split(split_order, N_bins)
    for i in range(N_bins):
        if array.shape[0] == 1170:
            to_stack = np.vstack((array1[splits[i],:], array2[splits[i],:], 
                               array3[splits[i],:]))
        elif array.shape[0] == 390:
            to_stack = array[splits[i],:]
        stacked_profiles[i,:] = splashback.stack_data(to_stack)

    return stacked_profiles


def stack_fixed_width(N_bins, split_data, array):
    """Splits clusters into bins of fixed width. Looks at data used for splitting
    clusters, anything more than 2 std away from the mean is ignored. Uses those
    bounds to make initial bins. If any of these bins have too few clusters, the
    min and max value are changed to the next bin with enough clusters. Bins are
    then recalculated."""
    mean = np.nanmean(split_data)
    std = np.nanstd(split_data)
    data_min = mean - 2*std #get rids of tail of distribution
    data_max = mean + 2*std #assumes roughly gaussian
    
    min_per_bin = 15 #if using substructure to bin, it will break - change this
    N_rad = array.shape[1]
    
    if array.shape[0] == 1170:
        array1 = array[:390,:]
        array2 = array[390:780,:]
        array3 = array[780:,:]
    
    bin_range = np.linspace(data_min, data_max, N_bins+1) #initial range for bins
    
    redo_bins = []
    redo = True
    while redo == True:

        redo = False
        bin_range = np.delete(bin_range, redo_bins) #check deletion for end bins
        bin_range = np.linspace(bin_range[0], bin_range[-1], N_bins+1) 
        
        if len(np.where((split_data >=bin_range[0]) & 
                        (split_data < bin_range[-1]))[0]) < N_bins * min_per_bin:
            print("ERROR: Too many bins")
            break
        
        redo_bins = []
        stacked_profiles = np.zeros((N_bins, N_rad))
        
        for i in range(N_bins):
            bin_mask = np.where((split_data >= bin_range[i]) & 
                                (split_data < bin_range[i+1]))[0]
            
            to_stack = array[bin_mask,:]
                
            stacked_profiles[i,:] = splashback.stack_data(to_stack)
            
            if len(bin_mask) <= min_per_bin:
                redo = True
                if i <= N_bins/2: #get rid of a start bin
                    redo_bins.append(i)
                elif i >= N_bins/2: #get rid of an end bin
                    redo_bins.append(i+1)
                    
    split_data_mid = (bin_range[1:] + bin_range[:-1]) / 2
    
    return stacked_profiles, split_data_mid
                
        
def distance_from_line(RSP_stack_2D, RSP_stack_3D, compare_data):
    distance = np.sqrt(2) / 2 * (np.abs(RSP_stack_2D - RSP_stack_3D))
    
    plt.figure()
    plt.scatter(compare_data, distance, color="blue", edgecolor="k")
    plt.xlabel(r"$\Gamma$")
    plt.ylabel(r"$\rm{d} R_{\rm{SP}}$")
    plt.show()


def compare_groups(N_stack, split_data, 
                   sim_type="DMO", 
                   hydro_input=mcs.DM_median,
                   label=r"$M_{200m}$"):
    if sim_type == "DMO":
        threeD = np.vstack((mcs.DMO_3D, mcs.DMO_3D, mcs.DMO_3D))
        twoD = mcs.DMO_2D
        rad_mid_2D = rad_mid_short
        rad_mid_3D = rad_mid_long
        
    elif sim_type == "hydro":
        threeD = np.vstack((mcs.DM_density_3D, mcs.DM_density_3D, mcs.DM_density_3D))
        twoD = hydro_input
        rad_mid_2D = rad_mid_short
        rad_mid_3D = rad_mid_short
        
    # stack_3D = stack_fixed_N(N_stack, split_data, threeD)
    # stack_2D = stack_fixed_N(N_stack, split_data, twoD)
    
    stack_3D, split_data_mid = stack_fixed_width(N_stack, split_data, threeD)
    stack_2D, split_data_mid = stack_fixed_width(N_stack, split_data, twoD)
    
    RSP_stack_2D = dr.standard(rad_mid_2D, stack_2D)
    RSP_stack_3D = dr.standard(rad_mid_3D, stack_3D)
    
    # for i in range(N_stack):
    #     plt.figure()
    #     plt.semilogx(rad_mid_2D, stack_2D[i,:], color="blue", 
    #                   linestyle="--", label="2D")
    #     plt.semilogx(rad_mid_3D, stack_3D[i,:], color="blue",
    #                   label="3D")
    #     ylim = plt.gca().get_ylim()
    #     plt.plot([RSP_stack_2D[i], RSP_stack_2D[i]], ylim,
    #               color="blue", linestyle="--")
    #     plt.plot([RSP_stack_3D[i], RSP_stack_3D[i]], ylim,
    #               color="blue")
    #     plt.ylim(ylim)
    #     plt.xlabel(r"$r/R_{200\rm{m}}$")
    #     plt.ylabel(r"$d \log \rho / d \log r$")
    #     plt.legend()
    #     #plt.savefig("example_profile_bin10.png", dpi=300)
    #     plt.show()
        
    #m, c = np.polyfit(RSP_stack_3D, RSP_stack_2D, 1)
    #a = 1
    b = 0
    a = np.sqrt(np.mean((RSP_stack_2D/RSP_stack_3D)**2))
    xlim = np.array([0.5,1.5])
    y = a*xlim + b
    
    distance = abs((a*RSP_stack_3D - RSP_stack_2D) / np.sqrt(a**2 +1))
    avg_scatter = np.mean(distance)
    print(avg_scatter)
    
    fig = plt.figure()
    cm = plt.cm.get_cmap('rainbow')
    plt.scatter(RSP_stack_3D, RSP_stack_2D, 
                c=split_data_mid, edgecolor="k", cmap=cm, s=75)
    plt.plot(xlim, y, linestyle="--", color="k")
    plt.xlabel(r"$R_{\rm{SP, 3D}} / R_{\rm{200m}}$")
    plt.ylabel(r"$R_{\rm{SP, 2D}} / R_{\rm{200m}}$")
    plt.xlim(xlim)
    #plt.title("Hydro - EM")
    plt.ylim((0.5,2.0))
    cbaxes = fig.add_axes([0.2, 0.72, 0.02, 0.2]) 
    cbar = plt.colorbar(cax=cbaxes, label=label)
    #cbar.set_ticks([1e5, 2e5])
    #cbar.set_ticklabels([r"10$^{15}$", r"2$\times$10$^{15}$"])
    
    #plt.savefig("spb_stacking_hydro_EM_mass_bins.png", dpi=300)
    plt.show()

morphology_path = "../Analysis_files/"
morph_x = np.genfromtxt(morphology_path + "morphology_stats_x.csv", delimiter=",")
morph_y = np.genfromtxt(morphology_path + "morphology_stats_y.csv", delimiter=",")
morph_z = np.genfromtxt(morphology_path + "morphology_stats_z.csv", delimiter=",")

# plt.scatter(morph_x[:,3], accretion, color="blue", edgecolor="k")
# plt.scatter(morph_y[:,3], accretion, color="blue", edgecolor="k")
# plt.scatter(morph_z[:,3], accretion, color="blue", edgecolor="k")
# plt.xscale("log")
# plt.xlabel("$c$")
# plt.ylabel("$\Gamma$")
# plt.show()

c = np.hstack((morph_x[:,0], morph_y[:,0], morph_z[:,0]))
s = np.hstack((morph_x[:,1], morph_y[:,1], morph_z[:,1]))
w = np.hstack((morph_x[:,3], morph_y[:,3], morph_z[:,3]))
acc =  np.hstack((accretion, accretion, accretion))
r_c = spearmanr(c, acc, nan_policy='omit').correlation
r_s = spearmanr(s, acc, nan_policy='omit').correlation
r_w = spearmanr(w, acc, nan_policy='omit').correlation

# fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3,5),
#                        sharex=True)
# ax[0].scatter(acc, c, edgecolor="k", color="royalblue")
# ax[1].scatter(acc, s, edgecolor="k", color="tomato")
# ax[2].scatter(acc, w, edgecolor="k", color="forestgreen")
# ax[2].set_yscale('log')
# plt.xlabel("$\Gamma$")
# ax[0].set_ylabel("$c$")
# ax[1].set_ylabel("$s$")
# ax[2].set_ylabel(r"$\langle w \rangle$")
# plt.subplots_adjust(0.25)
# #plt.savefig("accretion_morphology.png", dpi=300)
# plt.show()

compare_groups(12, accretion_DMO, 
                sim_type="DMO",
                label=r"$\Gamma$")
compare_groups(12, mass_DMO, 
                sim_type="DMO",
                label=r"$M_{200m}$")

compare_groups(12, accretion, 
                sim_type="hydro", 
                hydro_input=mcs.DM_2D_xyz,
                label=r"$\Gamma$")
compare_groups(11, mass, 
                sim_type="hydro", 
                hydro_input=mcs.DM_2D_xyz,
                label=r"$M_{200m}$")

compare_groups(12, accretion, 
                sim_type="hydro", 
                hydro_input=mcs.DM_median,
                label=r"$\Gamma$")
compare_groups(11, mass, 
                sim_type="hydro", 
                hydro_input=mcs.DM_median,
                label=r"$M_{200m}$")

compare_groups(12, accretion, 
                sim_type="hydro", 
                hydro_input=mcs.EM_median,
                label=r"$\Gamma$")
compare_groups(11, mass, 
                sim_type="hydro", 
                hydro_input=mcs.EM_median,
                label=r"$M_{200m}$")

compare_groups(12, c, 
               sim_type="hydro", 
               hydro_input=mcs.EM_median,
               label=r"c")
compare_groups(12, s, 
               sim_type="hydro", 
               hydro_input=mcs.EM_median,
               label=r"s")
compare_groups(12, np.log10(w), 
               sim_type="hydro", 
               hydro_input=mcs.EM_median,
               label=r"w")














