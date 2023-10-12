import numpy as np
import splashback as sp
import matplotlib.pyplot as plt

plt.style.use("mnras.mplstyle")

def plot_profiles():
    lw = 1
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.semilogx(flm.rad_mid, flm.mass_log_DM,
              color="darkviolet", 
              label="$\\rho_{\\rm{DM}}$",
              linewidth=lw)
    ax.semilogx(flm.rad_mid, flm.mass_log_gas,
              color="cyan", 
              label="$\\rho_{\\rm{gas}}$",
              linewidth=lw)
    ax.semilogx(flm.rad_mid, flm.mass_log_K,
              color="cornflowerblue", 
              label="$K$",
              linewidth=lw)
    ax.semilogx(flm.rad_mid, flm.mass_log_P,
              color="firebrick", 
              label="$P$",
              linewidth=lw)
    ax.semilogx(flm.rad_mid, flm.mass_log_T,
              color="gold", 
              label="$T$",
              linewidth=lw)
    ax.semilogx(flm.rad_mid, flm.mass_log_v,
              color="mediumseagreen", 
              label="$v$",
              linewidth=lw)
    ylim = ax.get_ylim()
    ax.plot((flm.R_DM_mass, flm.R_DM_mass), ylim,
            color="darkviolet", linestyle="--",
            linewidth=lw)
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel("$r/R_{\\rm{200m}}$")
    plt.ylabel("$d \log y / d \log r$")
    filename = "splashback_data/flamingo/plots/compare_3D_profiles.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
if __name__ == "__main__":
    box = "L1000N1800"
    
    flm = sp.flamingo(box, "HF")
    flm.read_properties()
    flm.read_entropy()
    flm.read_pressure()
    flm.read_temperature()
    flm.read_velocity()
    
    mass_bin = np.array([14.35,14.4])
    
    sp.stack_and_find_3D(flm, "mass", mass_bin, print_data=True)
    plot_profiles()