import splashback as sp
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("mnras.mplstyle")

def error_calculation(data1, data2):
    error1 = data1.R_DM_gap / data2.R_DM_gap
    error2 = data1.error_R_DM_gap / data1.R_DM_gap
    error3 = data2.error_R_DM_gap / data2.R_DM_gap
    error = error1 * np.sqrt(error2**2 + error3**2)
    return error


box = "L1000N1800"
flm = sp.flamingo(box, "HF")
flm.read_properties()
flm.read_magnitude_gap()

flm_high = sp.flamingo("L1000N3600", "HF")
flm_high.read_properties()
flm_high.read_magnitude_gap()

N_bins = 10
gap_bins = np.linspace(0, 3, N_bins+1)
sp.stack_and_find_3D(flm, "gap", gap_bins, bootstrap=True)
sp.stack_and_find_3D(flm_high, "gap", gap_bins, bootstrap=True)


fig, ax = plt.subplots(nrows=2, ncols=1,
                        sharex = True, sharey=True,
                        figsize=(3.321,4),
                        gridspec_kw={'hspace' : 0, 'wspace' : 0})
cm = plt.cm.copper(np.linspace(0,1,N_bins))
for i in range(N_bins):
    label = str(np.round(gap_bins[i],2)) \
            + "$< \\rm{M14} <$" \
            + str(np.round(gap_bins[i+1],2))
    ax[0].semilogx(flm.rad_mid, flm.gap_log_DM[i,:], color=cm[i],
                  linewidth=0.8, label=label)
    ax[1].semilogx(flm_high.rad_mid, flm_high.gap_log_DM[i,:], color=cm[i],
                  linewidth=0.8, label=label)
ax[0].legend()
ax[1].set_xlabel("$r/R_{\\rm{200m}}$")
fig.text(0.02, 0.45, "$d \log \\rho_{\\rm{DM}} / d \log r$", 
          transform=fig.transFigure, rotation='vertical')
ax[0].text(0.03, 0.05, "L1_m9", transform=ax[0].transAxes)
ax[1].text(0.03, 0.05, "L1_m8", transform=ax[1].transAxes)
# filename = "splashback_data/flamingo/plots/gap_res_compare.png"
# plt.savefig(filename, dpi=300)
plt.show()


gap_mids = (gap_bins[1:] + gap_bins[:-1]) / 2

plt.figure()
plt.errorbar(gap_mids, flm.R_DM_gap, yerr=flm.error_R_DM_gap, 
             capsize=2, label="L1_m9")
plt.errorbar(gap_mids, flm_high.R_DM_gap, yerr=flm_high.error_R_DM_gap, 
             capsize=2, label="L1_m8")
plt.legend()
plt.xlabel("$M14$")
plt.ylabel("$R_{\\rm{SP}}$")
filename = "splashback_data/flamingo/plots/gap_res_compare_R.png"
plt.savefig(filename, dpi=300)
plt.show()

