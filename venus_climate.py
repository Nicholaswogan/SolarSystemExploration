import numpy as np
from matplotlib import pyplot as plt

from utils import plot_PT
import venus
from photochem.clima import AdiabatClimate

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def climate(pc, clouds=True):

    c = AdiabatClimate(
        'input/species_climate.yaml', 
        'input/Venus/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir='photochem_data'
    )

    # Mixing ratios
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-10
    for i,sp in enumerate(c.species_names):
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    # Particles
    Pr = sol['pressure']
    ind1 = pc.dat.species_names.index('H2SO4aer')
    ind2 = c.particle_names.index('H2SO4aer')
    pdensities = np.zeros((len(Pr),len(c.particle_names)))
    pdensities[:,ind2] = pc.wrk.densities[ind1,:]
    pradii = np.ones((len(Pr),len(c.particle_names)))*0.1e-4
    pradii[:,ind2] = pc.var.particle_radius[ind1,:]

    if not clouds:
        pdensities *= 0.0

    c.solve_for_T_trop = True
    c.T_trop = 200
    c.xtol_rc = 1e-8
    c.P_top = 1
    c.max_rc_iters = 30
    c.max_rc_iters_convection = 5

    c.set_particle_density_and_radii(Pr, pdensities*0, pradii)
    c.surface_temperature(P_i, 700)

    c.set_particle_density_and_radii(Pr, pdensities, pradii)
    assert c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below, custom_dry_mix)

    return c

def plot(c1, c2):

    plt.rcParams.update({'font.size': 14})
    fig,ax = plt.subplots(1,1,figsize=[5,4])

    c = c1
    plot_PT(c, ax, lwc=2, color='k', lw=2, ls='--', label='Predicted')

    c = c2
    plot_PT(c, ax, lwc=2, color='0.6', lw=2, ls='--', label='Predicted (w/o clouds)')

    z, T, P = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    ax.plot(T, P , color='C3', lw=2, ls=':', label='VIRA')

    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(93,2e-6)
    ax.set_xticks(np.arange(200,900,100))
    ax.set_yticks(10.0**np.arange(-5,2,1))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.legend(ncol=1,bbox_to_anchor=(.98, 1.02), loc='upper right')

    # Put altitude on other axis
    ax1 = ax.twinx()
    ax1.set_yscale('log')
    ax1.set_ylim(*ax.get_ylim())
    ax1.minorticks_off()
    ax1.set_yticks(ax.get_yticks())
    c = c1
    ticks = ['%i'%np.interp(np.log10(a), np.log10(c.P/1e6)[::-1], c.z[::-1]/1e5) for a in ax.get_yticks()]
    ax1.set_yticklabels(ticks)
    ax1.set_ylabel('Approximate altitude (km)')

    plt.savefig('figures/venus_climate.pdf',bbox_inches='tight')

def main():
    
    pc = venus.initialize(
        reaction_file='input/zahnle_earth.yaml',
        atmosphere_file='results/Venus/atmosphere.txt',
        clouds=False
    )

    c1 = climate(pc, True)
    c2 = climate(pc, False)

    plot(c1, c2)

if __name__ == '__main__':
    main()

