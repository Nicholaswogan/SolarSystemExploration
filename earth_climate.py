import numpy as np
from matplotlib import pyplot as plt
# from photochem.clima import AdiabatClimate
from clima import AdiabatClimate
import utils
import earth

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def us_standard_atmosphere(z):
    T0 = 288
    zs = np.array([0, 11, 20, 32, 47, 51, 71, 86.0, 91.0, 110])
    slopes = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0, 0.0])
    T = [T0]
    for i in range(len(slopes)):
        intercept = T[-1] - zs[i]*slopes[i]
        T.append(intercept + zs[i+1]*slopes[i])
    T = np.array(T)
    return np.interp(z, zs, T)

def climate(pc, data_dir='photochem_data', fCO2=None):

    c = AdiabatClimate(
        'input/species_climate.yaml', 
        'input/Earth/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir=data_dir
    )

    # Mixing ratios
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-10
    for i,sp in enumerate(c.species_names):
        if sp == 'H2O':
            continue
        tmp = sol[sp]
        if sp == 'CO2' and fCO2 is not None:
            tmp[:] = fCO2
        custom_dry_mix[sp] = np.maximum(tmp,1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(tmp[0],1e-30)*sol['pressure'][0]
    P_i[c.species_names.index('H2O')] = 260e6

    # Particles
    Pr = sol['pressure']
    ind1 = pc.dat.species_names.index('H2SO4aer')
    ind2 = c.particle_names.index('H2SO4aer')
    pdensities = np.zeros((len(Pr),len(c.particle_names)))
    pdensities[:,ind2] = pc.wrk.densities[ind1,:]
    pradii = np.ones((len(Pr),len(c.particle_names)))*0.1e-4
    pradii[:,ind2] = pc.var.particle_radius[ind1,:]

    c.RH = np.ones(len(c.species_names))*0.4
    c.solve_for_T_trop = True
    c.T_trop = 200
    c.xtol_rc = 1e-5
    c.P_top = 10
    c.max_rc_iters = 30
    c.max_rc_iters_convection = 5

    c.set_particle_density_and_radii(Pr, pdensities*0, pradii)
    c.surface_temperature(P_i, 280)

    c.set_particle_density_and_radii(Pr, pdensities, pradii)
    converged = c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below, custom_dry_mix)
    assert converged
    return c

def plot(c1):
    plt.rcParams.update({'font.size': 14})
    fig,ax = plt.subplots(1,1,figsize=[5,4])

    utils.plot_PT(c1, ax, lwc=4, color='0.0', lw=2, ls='-', label='Predicted')

    z, P, T = np.loadtxt('input/Earth/PT_CIRA-86.txt',skiprows=2).T
    ax.plot(T, P , color='C3', lw=3, ls=':', label='CIRA-86\n(Equator Jan.)')

    z1, P1, T1 = np.loadtxt('input/Earth/PT_CIRA-86_45N.txt',skiprows=2).T
    ax.plot(T1, P1 , color='C9', lw=3, ls=':', label='CIRA-86\n'+r'(Equator 45$^{\circ}$ N)')

    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(1.013,1.5e-5)
    ax.set_xlim(180,305)
    ax.set_xticks(np.arange(180,310,20))
    ax.set_yticks(10.0**np.arange(-4,1,1))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.legend(ncol=1,bbox_to_anchor=(1.0,.52), loc='upper right',fontsize=9.5)

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

    plt.savefig('figures/earth_climate.pdf',bbox_inches = 'tight')

def main():
    pc = earth.initialize(atmosphere_file='results/Earth/atmosphere.txt')
    c1 = climate(pc)
    plot(c1)

if __name__ == "__main__":
    main()
