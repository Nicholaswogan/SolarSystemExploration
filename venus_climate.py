import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from utils import plot_PT
import venus
from utils import AdiabatClimateRobust

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)


def initial_guess(c, P_i):

    P1 = np.logspace(np.log10(np.sum(P_i)), np.log10(c.P_top), len(c.z)*2+1)
    P_surf_c = P1[0]
    P_c = P1[1:-1:2]
    z, T, P = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    P *= 1e6
    T_surf_c = np.interp(np.log10(P_surf_c), np.log10(P)[::-1], T[::-1]).copy()
    T_c = np.interp(np.log10(P_c)[::-1], np.log10(P)[::-1], T[::-1])[::-1].copy()

    convecting_with_below = np.empty_like(c.convecting_with_below)
    ind = np.argmin(np.abs(3e-1 - P_c/1e6))
    convecting_with_below[:] = False
    convecting_with_below[:ind] = True

    return T_surf_c, T_c, convecting_with_below

def climate(pc, clouds='crisp', SO2_correction=False):

    c = AdiabatClimateRobust(
        'input/species_climate.yaml', 
        'input/Venus/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir=None
    )

    # Mixing ratios
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-10
    for i,sp in enumerate(c.species_names):
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    # SO2 correction
    if SO2_correction:
        ind1 = np.argmin(np.abs(pc.var.z/1e5 - 50))
        SO2_1 = np.log10(sol['SO2'][ind1])
        z1 = pc.var.z[ind1]/1e5
        slope = (SO2_1 - (-9))/(z1 - 100)
        b = -slope*(z1) + (SO2_1)
        SO2_new = 10.0**(pc.var.z[ind1:]/1e5*slope + b)
        custom_dry_mix['SO2'][ind1:] = SO2_new

    # Particles
    Pr = sol['pressure']
    ind1 = pc.dat.species_names.index('H2SO4aer')
    ind2 = c.particle_names.index('H2SO4aer')
    pdensities = np.zeros((len(Pr),len(c.particle_names)))
    pdensities[:,ind2] = pc.wrk.densities[ind1,:]
    pradii = np.ones((len(Pr),len(c.particle_names)))*0.1e-4
    pradii[:,ind2] = pc.var.particle_radius[ind1,:]

    if clouds == 'photochem':
        venus.apply_custom_opacity_clima(c, crisp_cloud=False, uv='rimmer')
    elif clouds == 'crisp':
        pdensities *= 0.0
        venus.apply_custom_opacity_clima(c, crisp_cloud=True, uv='rimmer')
    elif clouds == None:
        pdensities *= 0.0
        venus.apply_custom_opacity_clima(c, crisp_cloud=False, uv='rimmer')
    else:
        raise Exception()

    c.solve_for_T_trop = True
    c.T_trop = 200
    c.xtol_rc = 1e-8
    c.P_top = 1
    c.max_rc_iters = 30
    c.max_rc_iters_convection = 5

    # c.set_particle_density_and_radii(Pr, pdensities*0, pradii)
    # c.surface_temperature(P_i, 700)
    T_surf_guess, T_guess, convecting_with_below_guess = initial_guess(c, P_i)

    c.set_particle_density_and_radii(Pr, pdensities, pradii)
    assert c.RCE(P_i, T_surf_guess, T_guess, convecting_with_below_guess, custom_dry_mix)

    return c

def plot(c1, c2, c3):

    plt.rcParams.update({'font.size': 13.75})
    fig,axs = plt.subplots(2,3,figsize=[14,7])

    labels = 'Predicted (predicted clouds & predicted SO$_2$)'

    colors = ['k','0.6','C0']
    labels = [
        'Model w/\npredicted clouds &\npredicted SO$_2$',
        'Model w/\nCrisp 1986 clouds &\npredicted SO$_2$',
        'Model w/\nCrisp 1986 clouds &\nobserved SO$_2$',
    ]
    keys = ['(a)','(b)','(c)']

    for i,c in enumerate([c1,c2,c3]):
        ax = axs[0,i]

        ax.text(.5, 1.03, labels[i], \
                    size = 15, ha='center', va='bottom',transform=ax.transAxes)

        ax.text(-0.3, 1.2, keys[i], \
                    size = 25, ha='left', va='bottom',transform=ax.transAxes)

        plot_PT(c, ax, lwc=4, color=colors[i], lw=2, ls='-', label='Predicted')
        
        z, T, P = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
        ax.plot(T, P , color='C3', lw=3, ls=':', label='VIRA')
        
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylim(93,1e-5)
        ax.set_xticks(np.arange(200,900,100))
        ax.set_yticks(10.0**np.arange(-5,2,1))
        ax.grid(alpha=0.4)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (bar)')
        ax.legend(ncol=1,bbox_to_anchor=(0.99, 0.99), loc='upper right', fontsize=12)

        ax = axs[1,i]
        z = np.append(0,c.z+c.dz/2)
        P = 10.0**np.interp(z, np.append(0,c.z), np.log10(np.append(c.P_surf,c.P)))
        F = -(c.rad.wrk_ir.fdn_n - c.rad.wrk_ir.fup_n)/1e3
        F = np.append(F[0:-1:2],F[-1])
        ax.plot(F, P/1e6, c=colors[i], lw=2, ls='--', label='Thermal (predicted)')
        
        z = np.append(0,c.z+c.dz/2)
        P = 10.0**np.interp(z, np.append(0,c.z), np.log10(np.append(c.P_surf,c.P)))
        F = (c.rad.wrk_sol.fdn_n - c.rad.wrk_sol.fup_n)/1e3
        F = np.append(F[0:-1:2],F[-1])
        ax.plot(F, P/1e6, c=colors[i], lw=2, label='Solar (predicted)')

        z, F = np.loadtxt('input/Venus/TomaskoGlobalSolarFluxes.txt',skiprows=2).T
        z1, _, P1 = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
        P = 10.0**np.interp(z, z1, np.log10(P1))
        ax.plot(F, P, c='C4', ls=':', label='Measure Solar\n(Tomasko+1980)',lw=3,zorder=0)

        fdn = c.rad.wrk_sol.fdn_n[-1]/1e3*4
        fup = c.rad.wrk_sol.fup_n[-1]/1e3*4
        albedo = fup/fdn
        
        ax.text(.98, .02, 'Bond albedo = %.2f'%(albedo), \
                size = 14, ha='right', va='bottom',transform=ax.transAxes)

        ax.legend(ncol=1,bbox_to_anchor=(0.01, 0.99), loc='upper left', fontsize=11)
        ax.set_ylabel('Pressure (bar)')
        ax.set_xlabel('Net flux (W m$^{-2}$)')
        
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylim(93,1e-5)
        ax.set_yticks(10.0**np.arange(-5,2,1))
        ax.set_xlim(0,155)
        ax.grid(alpha=0.4)

        rec = matplotlib.patches.Rectangle((-.33,-.3), 1.45, 3, fill=False, lw=1.5, clip_on=False,transform=ax.transAxes)
        rec = ax.add_patch(rec)

    plt.subplots_adjust(wspace=0.5,hspace=0.3)
    plt.savefig('figures/venus_climate.pdf',bbox_inches='tight')

def main():
    
    pc = venus.initialize(
        reaction_file='input/zahnle_earth.yaml',
        atmosphere_file='results/Venus/atmosphere.txt',
        crisp_clouds=False
    )

    c1 = climate(pc, clouds='photochem', SO2_correction=False)
    c2 = climate(pc, clouds='crisp', SO2_correction=False)
    c3 = climate(pc, clouds='crisp', SO2_correction=True)

    plot(c1, c2, c3)

if __name__ == '__main__':
    main()

