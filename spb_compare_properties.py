import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import determine_radius as dr
import corner

plt.style.use("mnras.mplstyle")

box = "L1000N1800"
flm = sp.flamingo(box, "HF")

N_bins = 45
log_radii = np.linspace(-1, 0.7, N_bins)
rad_mid = (10**log_radii[1:] + 10**log_radii[:-1]) / 2

log_DM_density = sp.log_gradients(rad_mid, flm.DM_density_3D)
log_gas_density = sp.log_gradients(rad_mid, flm.gas_density_3D)

Rsp_DM_3D = dr.standard(rad_mid, log_DM_density)
Rsp_gas_3D = dr.DM_prior_new(rad_mid, log_gas_density, Rsp_DM_3D)

flm.read_properties()

plot_data = np.vstack((np.log10(flm.M200m), flm.accretion, flm.energy,
                       flm.hot_gas_fraction, flm.baryon_fraction)).T
list_good = np.intersect1d(
    np.intersect1d(np.intersect1d(np.where(np.isfinite(plot_data[:,0]))[0], np.where(np.isfinite(plot_data[:,1]))[0]),
    np.intersect1d(np.where(np.isfinite(plot_data[:,2]))[0], np.where(np.isfinite(plot_data[:,3]))[0])),
    np.where(np.isfinite(plot_data[:,4]))[0])

plot_data = plot_data[list_good]

fig, ax = plt.subplots(5,5, figsize=(5,5), dpi=300,
                       gridspec_kw={'hspace' : 0, 'wspace' : 0})
fig.set_facecolor('w')
from matplotlib.lines import Line2D

fig= corner.corner(
    plot_data,
    fig=fig,
    plot_contours=True, plot_density=False, plot_datapoints=True,
    labels=[r'$\log M_{\rm{200m}}$', r'$\Gamma$', r'$E_{\rm{kin}}/E_{\rm{therm}}$',
            r'$f_{\rm{gas}}$', r'$f_{\rm{baryon}}$'],
    data_kwargs=dict(alpha=0.8, color='cornflowerblue'),
    hist_kwargs=dict(linewidth=0.8, color='cornflowerblue', histtype='stepfilled', density=True),
    hist2d_kwargs=dict(linewidth=0.8),
    label_kwargs=dict(fontsize=10),
    contour_kwargs=dict(linewidths=0.7, colors='cornflowerblue'),
    contourf_kwargs=dict(colors=['w', '#ffd1a9', '#bcb1a5', '#7991a1', '#36729d']),
    max_n_ticks=3,
    # quantiles=[0.16, 0.5, 0.84],
    # show_titles=True,
    smooth=1.6,
    # smooth1d=1.6,
    bins=40,
    hist_bin_factor=1,
    fill_contours=False
    # range=[(1, 2), 
    #   (0, 3), 
    #   (0, 0.5), 
    #   (0, 0.45)],    
)

for i in range(5):
    ax[i, i].yaxis.set_label_position('right')
    ax[i, i].yaxis.set_ticks_position('right')
    #ax[i, 0].set_xscale('log')
ax[0, 0].set_ylabel(r'$P(\log M_{\rm{200m}})$')
ax[1, 1].set_ylabel(r'$P(\Gamma)$')
ax[2, 2].set_ylabel(r'$P(E_{\rm{kin}}/E_{\rm{therm}})$')
ax[3, 3].set_ylabel(r'$P(f_{\rm{gas}})$')
ax[4, 4].set_ylabel(r'$P(f_{\rm{baryon}})$')

# # Legend
# handles=[
#     Line2D([0], [0], label=r'0.2 $R_{500}$', color='cornflowerblue', ls='-'),
#     Line2D([0], [0], label=r'$R_{500}$', color='grey', ls='-'),
# ]

ax[0, 0].legend(loc='upper right')
ax[0,0].set_xlim((14.0,15.55))
ax[1,0].set_xlim((14.0,15.55))
ax[2,0].set_xlim((14.0,15.55))
ax[3,0].set_xlim((14.0,15.55))
ax[4,0].set_xlim((14.0,15.55))

ax[1,1].set_xlim((0,8))
ax[2,1].set_xlim((0,8))
ax[3,1].set_xlim((0,8))
ax[4,1].set_ylim((0,8))

ax[2,2].set_xlim((0,0.5))
ax[3,2].set_xlim((0,0.5))
ax[4,2].set_xlim((0,0.5))

ax[3,3].set_xlim((0.96,1.001))
ax[4,3].set_xlim((0.96,1.001))

ax[4,4].set_xlim((0.09,0.15))

ax[1,0].set_ylim((0,8))

ax[2,0].set_ylim((0,0.5))
ax[2,1].set_ylim((0,0.5))

ax[3,0].set_ylim((0.96,1.001))
ax[3,1].set_ylim((0.96,1.001))
ax[3,2].set_ylim((0.96,1.001))

ax[4,0].set_ylim((0.09,0.15))
ax[4,1].set_ylim((0.09,0.15))
ax[4,2].set_ylim((0.09,0.15))
ax[4,3].set_ylim((0.09,0.15))

# plt.savefig('splashback_data/flamingo/plots/cornerplot.png')
plt.show()


# fig, ax = plt.subplots(nrows=5, ncols=5, 
#                        gridspec_kw={'hspace' : 0.1, 'wspace' : 0.1})

# mass_range = (1e13,1e15)
# energy_range = (0, 0.8)
# acc_range = (0,5)
# gas_range = (0.6,2.6)
# DM_range = (0.6, 2.6)

# ec = "k"
# size = 5
# lw=0.4

# ax[4,0].scatter(Rsp_DM_3D, flm.M200m, edgecolor=ec, s=size, linewidth=lw)
# ax[4,1].scatter(Rsp_gas_3D, flm.M200m, edgecolor=ec, s=size, linewidth=lw)
# ax[4,2].scatter(flm.accretion, flm.M200m, edgecolor=ec, s=size, linewidth=lw)
# ax[4,3].scatter(flm.energy, flm.M200m, edgecolor=ec, s=size, linewidth=lw)

# ax[4,0].set_yscale('log')
# ax[4,1].set_yscale('log')
# ax[4,2].set_yscale('log')
# ax[4,3].set_yscale('log')
  
# ax[3,0].scatter(Rsp_DM_3D, flm.energy, edgecolor=ec, s=size, linewidth=lw)
# ax[3,1].scatter(Rsp_gas_3D, flm.energy, edgecolor=ec, s=size, linewidth=lw)
# ax[3,2].scatter(flm.accretion, flm.energy, edgecolor=ec, s=size, linewidth=lw)
    
# ax[2,0].scatter(Rsp_DM_3D, flm.accretion, edgecolor=ec, s=size, linewidth=lw)
# ax[2,1].scatter(Rsp_gas_3D, flm.accretion, edgecolor=ec, s=size, linewidth=lw)
    
# ax[1,0].scatter(Rsp_DM_3D, Rsp_gas_3D, edgecolor=ec, s=size, linewidth=lw)

# mass_bins = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 30)
# ax[4,4].hist(flm.M200m, bins=mass_bins, range=mass_range, color="forestgreen", density=True)
# ax[4,4].set_xscale('log')
# ax[3,3].hist(flm.energy, bins=30, range=energy_range, color="crimson", density=True)
# ax[2,2].hist(flm.accretion, bins=30, range=acc_range, color="cornflowerblue", density=True)
# ax[1,1].hist(Rsp_gas_3D, bins=30, range=gas_range, color="goldenrod", density=True)
# ax[0,0].hist(Rsp_DM_3D, bins=30, range=DM_range, color="#434343", density=True)
        
# #remove axis ticks
        
# ax[4,3].set_yticks([])
# ax[4,2].set_yticks([])
# ax[4,1].set_yticks([])

# ax[3,1].set_yticks([])
# ax[3,2].set_yticks([])
# #ax[3,3].set_yticks([])

# ax[2,1].set_yticks([])


# ax[3,0].set_xticks([])
# ax[2,0].set_xticks([])
# ax[1,0].set_xticks([])
# ax[0,0].set_xticks([])

# ax[3,3].set_xticks([])
# ax[3,2].set_xticks([])
# ax[3,1].set_xticks([])

# ax[2,2].set_xticks([])
# ax[2,1].set_xticks([])

# ax[1,1].set_xticks([])

# #set plot limits  
    
# for i in range(4):
#     ax[4,i].set_ylim(mass_range)
#     ax[i+1,0].set_xlim(DM_range)       

# for i in range(3):
#     ax[3,i].set_ylim(energy_range)
#     ax[i+2,1].set_xlim(gas_range)
    
# ax[3,2].set_xlim(acc_range)
# ax[4,2].set_xlim(acc_range)
# ax[4,3].set_xlim(energy_range)

# ax[2,0].set_ylim(acc_range)
# ax[2,1].set_ylim(acc_range)
# ax[1,0].set_ylim(gas_range)
    
# #add axis labels
    
# ax[4,0].set_xlabel("$R_{\\rm{SP,DM}}$")
# ax[4,1].set_xlabel("$R_{\\rm{SP,gas}}$")
# ax[4,2].set_xlabel("$\Gamma$")
# ax[4,3].set_xlabel("$E_{\\rm{kin}} / E_{\\rm{therm}}$")

# ax[4,0].set_ylabel("$M_{\\rm{200m}}$")
# ax[3,0].set_ylabel("$E_{\\rm{kin}} / E_{\\rm{therm}}$")
# ax[2,0].set_ylabel("$\Gamma$")
# ax[1,0].set_ylabel("$R_{\\rm{SP, gas}}$")
# #ax[0,0].set_ylabel("$\sigma_{A,EM}$")


# #move x axis labels and ticks to top
# for i in range(5):
#     #ax[0,i].xaxis.set_label_position('top')
#     #ax[0,i].xaxis.tick_top()
    
#     ax[i,i].yaxis.set_label_position('right')
#     ax[i,i].yaxis.tick_right()
    
# ax[4,4].set_ylabel("$P(M_{\\rm{200m}})$")
# ax[3,3].set_ylabel("$P(E_{\\rm{kin}} / E_{\\rm{therm}})$")
# ax[2,2].set_ylabel("$P(\Gamma)$")
# ax[1,1].set_ylabel("$P(R_{\\rm{SP, gas}})$")
# ax[0,0].set_ylabel("$P(R_{\\rm{SP, DM}})$")

# ax[3,4].axis('off')
# ax[2,3].axis('off')
# ax[2,4].axis('off')
# ax[1,2].axis('off')
# ax[1,3].axis('off')
# ax[1,4].axis('off')
# ax[0,1].axis('off')
# ax[0,2].axis('off')
# ax[0,3].axis('off')
# ax[0,4].axis('off')

# file_name = "splashback_triangle.pdf"    
# #plt.savefig(file_name, dpi=300)
    
# plt.show()