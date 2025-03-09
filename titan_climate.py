import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as const
import utils
import titan

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def climate(pc, P_top=1.0, c_guess=None, haze=True, remove_C2H6=False):

    c = utils.AdiabatClimateRobust(
        'input/species_climate.yaml', 
        'input/Titan/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir='photochem_data'
    )

    if remove_C2H6:
        c.reinit_without_species_opacity('C2H6','k-distributions')
        c.reinit_without_species_opacity('C2H6','photolysis-xs')
        c.reinit_without_species_opacity('C2H2','k-distributions')
        c.reinit_without_species_opacity('C2H2','photolysis-xs')

    # Mixing ratios
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-15
    for i,sp in enumerate(c.species_names):
        if sp not in sol:
            continue
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    # Particles
    Pr = sol['pressure']
    ind = pc.dat.species_names.index('HCaer1')
    pdensities1 = pc.wrk.densities[ind,:]

    ind = pc.dat.species_names.index('HCaer2')
    pdensities1 += pc.wrk.densities[ind,:]

    ind = pc.dat.species_names.index('HCaer3')
    pdensities1 += pc.wrk.densities[ind,:]

    ind1 = c.particle_names.index('HCaer')
    pdensities = np.zeros((len(Pr),len(c.particle_names)))
    pdensities[:,ind1] = pdensities1

    if not haze:
        pdensities *= 0.0

    ind = pc.dat.species_names.index('HCaer1')
    pradii = np.ones((len(Pr),len(c.particle_names)))*1e-4
    pradii[:,ind1] = pc.var.particle_radius[ind,:]

    c.xtol_rc = 1e-5
    c.P_top = P_top
    c.max_rc_iters = 30
    c.max_rc_iters_convection = 5

    if c_guess is None:
        T_surf = 200
        T = np.ones(c.T.shape[0])*T_surf
        convecting_with_below = None
    else:
        T_surf = c_guess.T_surf
        T = c_guess.T
        convecting_with_below = c_guess.convecting_with_below

    c.set_particle_density_and_radii(Pr, pdensities, pradii)
    converged = c.RCE(P_i, T_surf, T, convecting_with_below, custom_dry_mix)
    assert converged
    return c

def get_P_from_T_rho(T, rho):
    rho = rho.copy()
    # kg/m^3 * g/kg * m^3/cm^3 = g/cm^3
    rho *= 1e3*(1/1e6)
    # g/cm^3 * mol/g * molecules/mol
    n = rho*(1/27.8)*const.N_A
    P = n*const.k*1e7*T
    return P

def plot(c1, c2, c3):
    plt.rcParams.update({'font.size': 14})
    fig,ax = plt.subplots(1,1,figsize=[5,4])

    c = c1
    utils.plot_PT(c, ax, lwc=2, color='k', lw=2, ls='--', label='Predicted')

    c = c2
    utils.plot_PT(c , ax, lwc=2, color='0.6', lw=2, ls='--', label='Predicted (w/o haze)')

    c = c3
    utils.plot_PT(c , ax, lwc=2, color='C0', lw=2, ls='--', label='Predicted\n(w/o C$_2$H$_2$ & C$_2$H$_6$)')

    dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
    rho = dat[:,3]
    T_mid = dat[:,6]
    T_min = dat[:,5]
    T_max = dat[:,7]
    P_mid = get_P_from_T_rho(T_mid, rho)/1e6
    ax.fill_betweenx(P_mid, T_min, T_max, fc='C3', alpha=0.3, label='Waite+2023')

    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(1.5,1e-6)
    ax.set_xticks(np.arange(60,211,30))
    ax.set_yticks(10.0**np.arange(-6,1,1))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.legend(ncol=1,bbox_to_anchor=(.01, 0.99), loc='upper left',fontsize=10)

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

    plt.savefig('figures/titan_climate.pdf',bbox_inches='tight')

def main():
    pc = titan.initialize(atmosphere_file='results/Titan/atmosphere.txt')

    c1 = climate(pc, P_top=1, c_guess=None, haze=True, remove_C2H6=False)
    c2 = climate(pc, P_top=1, c_guess=None, haze=False, remove_C2H6=False)
    c3 = climate(pc, P_top=1, c_guess=None, haze=True, remove_C2H6=True)

    plot(c1, c2, c3)

if __name__ == "__main__":
    main()