import numpy as np
import matplotlib.pyplot as plt
import splashback as sp


path = "splashback_data/"

M_200m = np.genfromtxt(path + "M_200m_macsis.csv", delimiter=",")
order_mass = np.argsort(M_200m) #indexes required to order masses
M_200m_xyz = np.hstack((M_200m, M_200m, M_200m))
order_mass_xyz = np.argsort(M_200m_xyz)

accretion = np.genfromtxt(path + "accretion_rates.csv", delimiter=",")
order_accretion = np.argsort(accretion)
accretion_xyz = np.hstack((accretion, accretion, accretion))
order_accretion_xyz = np.argsort(accretion_xyz)

energy_ratio = np.genfromtxt(path + "energy_ratio_200m_macsis.csv", delimiter=",")
order_energy = np.argsort(energy_ratio)
energy_xyz = np.hstack((energy_ratio, energy_ratio, energy_ratio))
order_energy_xyz = np.argsort(energy_xyz)

N_bins = 40
N_macsis = 1170 #for now only use independent clusters rather than xyz

log_radii = np.linspace(-1, 0.6, N_bins+1)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

macsis = sp.macsis()

x = 1
bin_number = np.array([])
while x <= 39:
    bin_number = np.append(bin_number, x)
    y = N_macsis * x / (N_macsis - x)
    y_new = np.rint(y)
    if y_new == x:
        x = y_new + 1
    else:
        x = y_new


part2 = np.rint(N_macsis/np.arange(29,2,-1))
bin_number = np.append(bin_number, part2) #number of bins to go over


for i in bin_number:
    i = int(i)
    splits = np.array_split(order_mass_xyz, i)
    Rsp_EM = np.zeros(i)
    Rsp_DM = np.zeros(i)

    for j in range(i):
        EM_median = np.zeros(N_bins)
        DM_median = np.zeros(N_bins)
        
        mass_bin = splits[j] #indexes of a mass bin
        
        for k in range(N_bins):
            EM_median[k] = np.nanmedian(macsis.EM_median[mass_bin,k])
            DM_median[k] = np.nanmedian(macsis.DM_median[mass_bin,k])
        
        try:
            Rsp_EM[j] = sp.R_SP_finding(rad_mid, EM_median)
            Rsp_DM[j] = sp.R_SP_finding(rad_mid, DM_median)
        except ValueError:
            Rsp_EM[j] = np.nan
            Rsp_DM[j] = np.nan
        
    plt.figure()
    plt.scatter(Rsp_DM, Rsp_EM, edgecolor="k", color="b")
    plt.title("Number of bins: " + str(i) + ". Clusters per bin (approx): " + str(np.rint(N_macsis/i)))
    plt.xlabel("$R_{SP,DM}$")
    plt.ylabel("$R_{SP,EM}$")
    plt.show()


