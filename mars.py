import numpy as np
from matplotlib import pyplot as plt
from utils import add_data_to_figure
from utils import EvoAtmosphereRobust
import utils
import yaml
from scipy import constants as const

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
        'input/Mars/zahnle_earth_HNOC.yaml',
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

def plot(pc):
    with open('planetary_atmosphere_observations/Mars.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    sol = pc.mole_fraction_dict()

    plt.rcParams.update({'font.size': 13})
    fig,ax = plt.subplots(1,1,figsize=[5,4],sharex=False,sharey=True)
    fig.patch.set_facecolor("w")

    species = ['CO','O2','O3','CO2','H2O2','NO','H2O']
    colors = ['C4','C5','C6','C2','C7','C1','C0']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=2)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=2,ms=3,elinewidth=0.9, capthick=0.9, alpha=0.9)
        # ax.text(*coords[i],utils.species_to_latex(sp),size = 12, 
        #     ha='center', va='center',color=colors[i])
    ax.set_xlim(5e-12,1.2)
    # ax.text(0.02, .96, '(a)', size = 20, ha='left', va='top',transform=ax.transAxes,color='k')

    ind = pc.dat.species_names.index('H2Oaer')
    saturation = pc.dat.particle_sat[ind].sat_pressure
    mix = [saturation(T)/pc.wrk.pressure[i] for i,T in enumerate(pc.var.temperature)]
    # mix = np.array([pc.dat.particle_sat[ind].sat_pressure(T)/1e6 for T in sol['temperature']])/sol['pressure']/1e6
    ax.plot(mix,pc.var.z/1e5,c='C0', ls='--', alpha=0.7,label='H$_2$O\nsat.')

    ax.legend(ncol=1,bbox_to_anchor=(1.01,1.01),loc='upper left',fontsize=12)

    ax.grid(alpha=0.4)
    ax.set_xscale('log')
    ax.set_ylim(0,110)
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Mixing Ratio')
        
    plt.savefig('figures/mars.png',dpi=300,bbox_inches = 'tight')

def main():
    pc = initialize()
    assert pc.find_steady_state()
    pc.out2atmosphere_txt('results/Mars/atmosphere.txt',overwrite=True)

    plot(pc)

if __name__ == '__main__':
    main()