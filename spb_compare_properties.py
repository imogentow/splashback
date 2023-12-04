import splashback as sp
import numpy as np
import matplotlib.pyplot as plt
import corner

plt.style.use("mnras.mplstyle")

box = "L1000N1800"
flm = sp.flamingo(box, "HF")
flm.read_properties()
flm.read_2D_properties()
flm.read_magnitude_gap()

log_DM_density = sp.log_gradients(flm.rad_mid, flm.DM_density_3D)
log_gas_density = sp.log_gradients(flm.rad_mid, flm.gas_density_3D)

N_clusters = len(flm.M200m)
flm.concentration = flm.concentration[:N_clusters]
flm.symmetry = flm.symmetry[:N_clusters]
flm.alignment = flm.alignment[:N_clusters]
flm.centroid = flm.centroid[:N_clusters]

properties_all = np.vstack((flm.accretion, np.log10(flm.M200m), flm.energy, flm.gap,
                       flm.concentration, flm.symmetry, flm.alignment, flm.centroid))
list_good1 = np.intersect1d(
    np.intersect1d(np.intersect1d(np.where(np.isfinite(properties_all[0,:]))[0], 
                                  np.where(np.isfinite(properties_all[1,:]))[0]),
    np.intersect1d(np.where(np.isfinite(properties_all[2,:]))[0], np.where(np.isfinite(properties_all[3,:]))[0])),
    np.where(np.isfinite(properties_all[4,:]))[0])
list_good2 = np.intersect1d(
    np.intersect1d(np.where(np.isfinite(properties_all[5,:]))[0], 
                                  np.where(np.isfinite(properties_all[6,:]))[0]),
    np.where(np.isfinite(properties_all[7,:]))[0])
list_good = np.intersect1d(list_good1, list_good2)

plot_data = properties_all[:,list_good].T
labels = [r'$\Gamma$', r'$\log \left( M_{\rm{200m}}/M_{\odot}\right)$', r'$X_{\rm{E}}$',
            '$M14$', '$c$', '$s$', '$a$', '$\log \langle w \\rangle$']
labels_p = [r'$\Gamma$', r'$M_{\rm{200m}}$', r'$X_{\rm{E}}$',
            '$M14$', '$c$', '$s$', '$a$', '$\log \langle w \\rangle$']
N_properties = len(labels)

fig, ax = plt.subplots(N_properties, N_properties, figsize=(6.5,6), dpi=300,
                       gridspec_kw={'hspace' : 0, 'wspace' : 0})
fig.set_facecolor('w')
fig= corner.corner(
    plot_data,
    fig=fig,
    plot_contours=True, plot_density=False, plot_datapoints=True,
    #labels=labels,
    data_kwargs=dict(alpha=0.8, color='cornflowerblue'),
    hist_kwargs=dict(linewidth=0.8, color='cornflowerblue', histtype='stepfilled', density=True),
    hist2d_kwargs=dict(linewidth=0.8),
    label_kwargs=dict(fontsize=10),
    contour_kwargs=dict(linewidths=0.7, colors='cornflowerblue'),
    contourf_kwargs=dict(colors=['w', '#ffd1a9', '#bcb1a5', '#7991a1', '#36729d']),
    max_n_ticks=3,
    smooth=1.6,
    bins=40,
    hist_bin_factor=1,
    fill_contours=False  
)

for i in range(N_properties):
    ax[i, i].yaxis.set_label_position('right')
    ax[i, i].yaxis.set_ticks_position('right')
    ax[i, i].set_ylabel(r'$P(${}$)$'.format(labels_p[i]))
    if i >= 1:
        ax[i,0].set_ylabel(labels[i], labelpad=5)
    ax[N_properties-1,i].set_xlabel(labels[i], labelpad=5)

M_lim = (14,15.55)
Gamma_lim = (0,8)
E_lim = (0,0.5)
mag_lim = (0.0,3.5)
c_lim =(0,0.6)
s_lim = (0,1.6)
a_lim = (0,2)
w_lim = (-3, -1)

for i in range(N_properties):
    ax[i,0].set_xlim(Gamma_lim)
    
for i in range(N_properties-1):
    ax[i+1,1].set_xlim(M_lim)
    ax[N_properties-1,i].set_ylim(w_lim)

for i in range(N_properties-2):
    ax[i+2,2].set_xlim(E_lim)
    ax[N_properties-2,i].set_ylim(a_lim)

for i in range(N_properties-3):
    ax[i+3,3].set_xlim(mag_lim)
    ax[N_properties-3,i].set_ylim(s_lim)

for i in range(N_properties-4):
    ax[i+4,4].set_xlim(c_lim)
    ax[N_properties-4,i].set_ylim(c_lim)

ax[5,5].set_xlim(s_lim)
ax[6,5].set_xlim(s_lim)
ax[7,5].set_xlim(s_lim)

ax[6,6].set_xlim(a_lim)
ax[7,6].set_xlim(a_lim)

ax[7,7].set_xlim(w_lim)

ax[1,0].set_ylim(M_lim)

ax[2,0].set_ylim(E_lim)
ax[2,1].set_ylim(E_lim)

ax[3,0].set_ylim(mag_lim)
ax[3,1].set_ylim(mag_lim)
ax[3,2].set_ylim(mag_lim)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
plt.savefig('splashback_data/flamingo/plots/cornerplot.png')
plt.show()


