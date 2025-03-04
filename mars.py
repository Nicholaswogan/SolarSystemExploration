import numpy as np
from matplotlib import pyplot as plt
from utils import add_data_to_figure
from utils import EvoAtmosphereRobust
from photochem.clima import AdiabatClimate
import utils
import yaml
from scipy import constants as const

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def get_zTKzzmix():
    # all from Zahnle et al (2008)

    z = np.arange(0,111.0,1)
    T = 211 - 1.4*z
    ind = np.argmin(np.abs(z - 50))
    T[ind:] = T[ind]

    z = z*1e5
    mubar = np.ones(len(z))*(0.95*44 + 4.8e-2*28)

    planet_mass = 6.39e26
    planet_radius = 3.3895e8
    P_surf = 6.3e-3*1e6

    P = utils.compute_pressure_of_ZT(z, T, mubar, planet_radius, planet_mass, P_surf)

    n = P/(const.k*1e7*T)

    ind = np.argmin(np.abs(z - 20e5))
    Kzz = 1e6*(n/n[ind])**-0.5
    Kzz[:ind] = 1e6

    surf_mix = {
        'CO2': 0.95,
        'N2': 0.048
    }
    mix = {a : np.ones(len(z))*surf_mix[a] for a in surf_mix}
    return z, T, Kzz, mix, P_surf

def initialize(atmosphere_file=None):
    
    pc = utils.EvoAtmosphereRobust(
        'input/zahnle_earth_HNOC.yaml',
        'input/Mars/settings.yaml',
        'input/SunNow.txt',
        atmosphere_file='results/Mars/atmosphere.txt',
        data_dir='photochem_data'
    )

    if atmosphere_file is None:
        # Construct atmosphere
        z, T, Kzz, mix, P_surf = get_zTKzzmix()
        pc.initialize_to_zT(z, T, Kzz, mix, P_surf)

    pc.set_particle_parameters(0.5, 100, 10)
    pc.set_particle_radii({'H2Oaer': 1.0e-3})
    pc.var.atol = 1e-18

    return pc

def climate(pc, P_top=1.0, dust_factor=0.0, dust_radii=1.5e-4):

    c = AdiabatClimate(
        'input/Mars/species_climate.yaml', 
        'input/Mars/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir='photochem_data'
    )
    
    # Composition
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-15
    for i,sp in enumerate(c.species_names):
        if sp not in sol:
            continue
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    # Dust
    # Read profile
    z, pdensities = np.loadtxt('input/Mars/DSTprof_MarsREF.txt',skiprows=7).T
    z = z[::-1].copy()
    pdensities = pdensities[::-1].copy()

    # Get pressures of altitudes
    sol = pc.mole_fraction_dict()
    Pr = 10.0**np.interp(z,sol['alt']/1e5,np.log10(sol['pressure']))

    # Set densities and radii
    pdensities = pdensities.reshape((len(Pr),len(c.particle_names)))
    pradii = np.ones_like(pdensities)*dust_radii

    c.solve_for_T_trop = True
    c.xtol_rc = 1e-8
    c.P_top = P_top
    c.max_rc_iters = 30

    # Compute guess
    c.T_trop = 150
    c.set_particle_density_and_radii(Pr, pdensities*0, pradii)
    c.surface_temperature(P_i, 200)

    # Compute climate
    c.set_particle_density_and_radii(Pr, pdensities*dust_factor, pradii)
    assert c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below*False, custom_dry_mix)

    return c

def plot(pc, c_clear, c_small, c_mid, c_large):

    plt.rcParams.update({'font.size': 13.5})
    fig,axs = plt.subplots(1,2,figsize=[10,3.5],sharex=False,sharey=False)
    fig.patch.set_facecolor("w")

    with open('planetary_atmosphere_observations/Mars.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    # Plot VMR and data
    ax = axs[0]
    sol = pc.mole_fraction_dict()
    species = ['CO','O2','O3','CO2','H2O2','NO','H2O']
    colors = ['C4','C5','C6','C2','C7','C1','C0']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=2)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=2,ms=3,elinewidth=0.9, capthick=0.9, alpha=0.9)

    # SVP
    ind = pc.dat.species_names.index('H2Oaer')
    saturation = pc.dat.particle_sat[ind].sat_pressure
    mix = [saturation(T)/pc.wrk.pressure[i] for i,T in enumerate(pc.var.temperature)]
    ax.plot(mix,pc.var.z/1e5,c='C0', ls='--', alpha=0.7,label='H$_2$O\nsat.')

    # Settings
    ax.legend(ncol=3,bbox_to_anchor=(0.5,1.01),loc='lower center',fontsize=12)
    ax.set_xlim(5e-12,1.2)
    ax.grid(alpha=0.4)
    ax.set_xscale('log')
    ax.set_ylim(0,110)
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Mixing Ratio')

    # Climate
    ax = axs[1]
    utils.plot_PT(c_clear, ax, lwc=2, color='0.6', lw=2, ls='--', label='Predicted (no dust)')
    utils.plot_PT(c_small, ax, lwc=2, color='0.4', lw=2, ls='--', label=r'Predicted ($0.1\times$dust)')
    utils.plot_PT(c_mid, ax, lwc=2, color='0.2', lw=2, ls='--', label=r'Predicted ($1\times$dust)')
    utils.plot_PT(c_large, ax, lwc=2, color='0.0', lw=2, ls='--', label=r'Predicted ($10\times$dust)')

    # Settings
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(c_clear.P_surf/1e6,4e-6)
    ax.set_xlim(135,235)
    ax.set_xticks(np.arange(140,225,20))
    ax.set_yticks(10.0**np.arange(-5,-2,1))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.legend(ncol=2,bbox_to_anchor=(0.5,1.01),loc='lower center',fontsize=12)

    # Put altitude on other axis
    c = c_clear
    ax1 = ax.twinx()
    ax1.set_yscale('log')
    ax1.set_ylim(*ax.get_ylim())
    ax1.minorticks_off()
    ax1.set_yticks(ax.get_yticks())
    ticks = ['%i'%np.interp(np.log10(a), np.log10(c.P/1e6)[::-1], c.z[::-1]/1e5) for a in ax.get_yticks()]
    ax1.set_yticklabels(ticks)
    ax1.set_ylabel('Approximate altitude (km)')

    plt.subplots_adjust(wspace=0.3)
        
    plt.savefig('figures/mars.png',dpi=300,bbox_inches = 'tight')

def main():

    # Photochemistry
    pc = initialize('results/Mars/atmosphere.txt')
    assert pc.find_steady_state()
    # pc.out2atmosphere_txt('results/Mars/atmosphere.txt',overwrite=True)

    # Climate
    c_clear = climate(pc, dust_factor=0.0, P_top=3)
    c_small = climate(pc, dust_factor=1.0e-1, P_top=3)
    c_mid = climate(pc, dust_factor=1.0, P_top=3)
    c_large = climate(pc, dust_factor=1.0e1, P_top=3)

    plot(pc, c_clear, c_small, c_mid, c_large)

if __name__ == '__main__':
    main()