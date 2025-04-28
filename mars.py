import numpy as np
from matplotlib import pyplot as plt
from utils import add_data_to_figure
from utils import EvoAtmosphereRobust
# from photochem.clima import AdiabatClimate
from clima import AdiabatClimate
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

def climate(pc, dust_case, P_top=1.0):

    c = AdiabatClimate(
        'input/species_climate.yaml', 
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
    z, pdensities1, pradii1, T_glob, T_glob1 = np.loadtxt('input/Mars/DSTprof_MarsREF_v2.txt',skiprows=7).T
    z = z[::-1].copy()
    pdensities1 = pdensities1[::-1].copy()
    pradii1 = pradii1[::-1].copy()/1e4 # to cm

    # Get pressures of altitudes
    sol = pc.mole_fraction_dict()
    Pr = 10.0**np.interp(z,sol['alt']/1e5,np.log10(sol['pressure']))

    # Set densities and radii
    ind = c.particle_names.index('Dust')
    pdensities = np.zeros((len(Pr),len(c.particle_names)))
    pdensities[:,ind] = pdensities1
    pradii = np.ones((len(Pr),len(c.particle_names)))*1e-4
    pradii[:,ind] = pradii1

    c.solve_for_T_trop = True
    c.xtol_rc = 1e-6
    c.P_top = P_top
    c.max_rc_iters = 30

    # Compute guess
    c.T_trop = 150
    c.set_particle_density_and_radii(Pr, pdensities*0, pradii)
    c.surface_temperature(P_i, 200)

    # Dust optical depths at 9.3 microns, characteristic of different seasons
    # in the Martian year. All of these come from Fig 21 in Montabone et al. (2015). 
    dust_opd = {
        'low': 0.075, # Charateristic of the not so dusty time of the year (solar longitude ~40)
        'mid': 0.375, # Characteristic of the dusty season (solar longitude 240)
        'high': 1.0 # Characteristic of a global dust storm.
    }
    # These factors are the number I need to multiply the particle density by
    # so that the 9.3 dust optical depth is equal to `dust_opd`.
    dust_factors = {
        'low': 0.04002026301870318,
        'mid': 0.2001013150935159,
        'high': 0.5336035069160424
    }

    # Compute climate
    c.set_particle_density_and_radii(Pr, pdensities*dust_factors[dust_case], pradii)
    assert c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below*False, custom_dry_mix)

    return c

def plot(pc, c_low, c_mid, c_high):

    plt.rcParams.update({'font.size': 13.5})
    fig,axs = plt.subplots(1,2,figsize=[10,3.0],sharex=False,sharey=False)
    fig.patch.set_facecolor("w")

    with open('planetary_atmosphere_observations/Mars.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    # Plot VMR and data
    # Kasnopolski 2001 suggests 15 ppm H2 at surface. Should note this in text and compare.
    ax = axs[0]
    sol = pc.mole_fraction_dict()
    species = ['CO','O2','O3','CO2','H2O2','HO2','NO','H2O']
    colors = ['C4','C5','C6','C2','C7','C9','C1','C0']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=2)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=2,ms=3,elinewidth=0.9, capthick=0.9, alpha=0.9)

    # SVP
    ind = pc.dat.species_names.index('H2Oaer')
    saturation = pc.dat.particle_sat[ind].sat_pressure
    mix = [saturation(T)/pc.wrk.pressure[i] for i,T in enumerate(pc.var.temperature)]
    ax.plot(mix,pc.var.z/1e5,c='C0', ls='--', alpha=0.7,label='H$_2$O sat.')

    # Settings
    ax.legend(ncol=3,bbox_to_anchor=(0.5,1.015),loc='lower center',fontsize=11)
    ax.set_xlim(5e-12,1.2)
    ax.grid(alpha=0.4)
    ax.set_xscale('log')
    ax.set_ylim(0,110)
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Mixing Ratio')

    # Climate
    ax = axs[1]
    utils.plot_PT(c_low, ax, lwc=4, color='C0', lw=2, ls='-', label='Predicted\n(clear season)')
    utils.plot_PT(c_mid, ax, lwc=4, color='C5', lw=2, ls='-', label='Predicted\n(dusty season)')
    utils.plot_PT(c_high, ax, lwc=4, color='k', lw=2, ls='-', label='Predicted\n(global dust storm)')

    z, _,_,_, T_glob = np.loadtxt('input/Mars/DSTprof_MarsREF_v2.txt',skiprows=7).T
    z = z[::-1].copy()
    T_glob = T_glob[::-1].copy()
    c = c_mid
    P = 10.0**np.interp(z,c.z/1e5,np.log10(c.P))
    ax.plot(T_glob, P/1e6, 'C3', lw=3, ls=':', label='Kahre+2023\n'+r'$\pm45^\circ$ avg.')

    # Settings
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(c_mid.P_surf/1e6,1e-6)
    ax.set_xlim(135,235)
    ax.set_xticks(np.arange(140,225,20))
    ax.set_yticks(10.0**np.arange(-6,-2,1))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.legend(ncol=2,bbox_to_anchor=(0.5,1.015),loc='lower center',fontsize=11)

    # Put altitude on other axis
    c = c_mid
    ax1 = ax.twinx()
    ax1.set_yscale('log')
    ax1.set_ylim(*ax.get_ylim())
    ax1.minorticks_off()
    ax1.set_yticks(ax.get_yticks())
    ticks = ['%i'%np.interp(np.log10(a), np.log10(c.P/1e6)[::-1], c.z[::-1]/1e5) for a in ax.get_yticks()]
    ax1.set_yticklabels(ticks)
    ax1.set_ylabel('Approximate altitude (km)')

    plt.subplots_adjust(wspace=0.3)

    plt.savefig('figures/mars.pdf',bbox_inches = 'tight')

def main():

    # Photochemistry
    pc = initialize()
    assert pc.find_steady_state()
    pc.out2atmosphere_txt('results/Mars/atmosphere.txt',overwrite=True)

    # Climate
    c_low = climate(pc, dust_case='low', P_top=5e-1)
    c_mid = climate(pc, dust_case='mid', P_top=5e-1)
    c_high = climate(pc, dust_case='high', P_top=5e-1)

    plot(pc, c_low, c_mid, c_high)

if __name__ == '__main__':
    main()