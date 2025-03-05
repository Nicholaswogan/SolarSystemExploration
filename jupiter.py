import numpy as np
from matplotlib import pyplot as plt
from photochem.extensions import gasgiants
import utils
from photochem.clima import AdiabatClimate
import numba as nb
import yaml
import pickle
from astropy import constants
import numba as nb

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

@nb.cfunc(nb.double(nb.double, nb.double, nb.double))
def custom_binary_diffusion_fcn(mu_i, mubar, T):
    # Equation 6 in Gladstone et al. (1996)
    if np.isclose(mu_i, 16.04288):
        # Methane
        b = 2.2965e17*T**0.765
    else:
        # Gases that are not methane
        b = 3.64e-5*T**(1.75-1.0)*7.3439e21*np.sqrt(2.01594/mu_i)
    return b

def initialize(special_CH4_diffusion=True):

    pc = gasgiants.EvoAtmosphereGasGiant(
        'input/zahnle_earth_HNOCHe.yaml',
        'input/SunNow.txt',
        constants.M_jup.cgs.value,
        constants.R_jup.cgs.value,
        nz=200,
        photon_scale_factor=0.039,
        thermo_file='input/zahnle_earth_HNOCHe_thermo.yaml',
        data_dir='photochem_data'
    )
    pc.gdat.TOA_pressure_avg = 3e-3

    # Set elemental composition
    gas = pc.gdat.gas
    comp = {
        'H': 1.0,
        'He': 0.0785, # Atreya et al. 2020
        'C': 1.19e-3, # Atreya et al. 2020
        'O': 3.29e-04*0.8, # Cavalie et al. (2023), See note below
        'N': 1.97e-04, # Moeckel et al. (2023), See note below
    }

    # Solar O/H = 6.587302e-04 according to Lodders et al (2021)
    # Cavalie et al. (2023) finds a O/H of ~0.5x solar based on kinetics modeling
    # (reconciling the quenched species like CO). So, this puts us at O/H = 3.293651e-04
    # Or rounded to 3.29e-04. I further multiply by 0.8 to account for rocks 
    # condensing and sequestering O.
    # 
    # Apparently O/H isn't really known on Jupiter. The uncertainty from Juno is really big.

    # Moeckel et al. (2023) used Juno to derive deep atmosphere NH3. They find 2.66x solar N/H, where
    # Solar N/H is taken to be 7.4e-05. So, then you have a resulting N/H of 1.968400e-04
    # which I round to 1.97e-4. This is consistent with Atreya et al. (2020).
    
    molfracs_atoms_jupiter = np.ones(len(gas.atoms_names))*1e-20
    for sp in comp:
        molfracs_atoms_jupiter[gas.atoms_names.index(sp)] = comp[sp]
    molfracs_atoms_jupiter = molfracs_atoms_jupiter/np.sum(molfracs_atoms_jupiter)
    pc.gdat.gas.molfracs_atoms_sun = molfracs_atoms_jupiter

    # Upper BCs following Moses et al. (2005)
    pc.set_upper_bc('H2O',bc_type='flux',flux=-4e4)
    pc.set_upper_bc('CO',bc_type='flux',flux=-4e6)
    pc.set_upper_bc('CO2',bc_type='flux',flux=-1e4)

    # P, T, Kzz from Tsai et al. (2021)
    P, T, Kzz = np.loadtxt('input/Jupiter/Jupiter_deep_top.txt',skiprows=2).T
    inds = np.where(P > 1e-2)
    P = P[inds].copy()
    T = T[inds].copy()
    Kzz = Kzz[inds].copy()

    # Initialize the atmosphere
    pc.initialize_to_climate_equilibrium_PT(
        P,
        T,
        Kzz,
        1.0,
        1.0
    )

    # Make sure condensation will work
    for i in range(pc.dat.np):
        pc.var.cond_params[i].smooth_factor = 1
        pc.var.cond_params[i].k_cond = 1000
        pc.var.cond_params[i].k_evap = 10

    # 10 microns for all particles
    pc.var.particle_radius = np.ones_like(pc.var.particle_radius)*1e-3
    pc.update_vertical_grid(TOA_alt=pc.var.top_atmos)

    if special_CH4_diffusion:
        pc.var.custom_binary_diffusion_fcn = custom_binary_diffusion_fcn

    return pc

def climate(pc):
    c = AdiabatClimate(
        'input/species_climate.yaml',
        'input/Jupiter/settings_climate.yaml',
        'input/SunNow.txt',
        data_dir='photochem_data'
    )

    # Mixing ratios
    sol = pc.return_atmosphere()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-10
    for i,sp in enumerate(c.species_names):
        if sp not in sol:
            continue
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    c.xtol_rc = 1e-6
    c.P_top = 1
    c.max_rc_iters = 30
    c.max_rc_iters_convection = 5
    c.surface_heat_flow = 7.485e3 # Li et al. (2018)

    sol = pc.return_atmosphere()
    P = np.logspace(np.log10(np.sum(P_i)), np.log10(c.P_top),len(c.z))
    T = np.interp(np.log10(P)[::-1].copy(), np.log10(sol['pressure'])[::-1].copy(), sol['temperature'][::-1].copy())[::-1].copy()
    T_surf = T[0].copy()

    convecting_with_below = np.zeros_like(c.convecting_with_below)
    convecting_with_below[:] = False
    convecting_with_below[0] = True
    convecting_with_below[1] = True

    c.RCE(P_i, T_surf, T, convecting_with_below, custom_dry_mix=custom_dry_mix)

    return c

def add_data_to_figure_p(sp, dat, ax, default_error = None, **kwargs):
    if sp in dat:
        entry = utils.retrieve_species(sp, dat)
        for j,en in enumerate(entry):
            for jj in range(len(en['mix'])):  
                mix = en['mix'][jj]
                alt = en['P'][jj]
                xerr = en['mix-err'][:,jj].reshape((2,1))
                yerr = en['P-err'][:,jj].reshape((2,1))
                if xerr[0,0] == mix:
                    ax.errorbar(mix, alt, yerr=yerr, xerr=10.0**(np.log10(mix)-0.1), xuplims=[True],**kwargs)
                elif np.all(xerr.flatten() == np.array([0,0])) and default_error is not None:
                    low = mix - mix*default_error
                    high = low
                    xerr = np.array([low, high]).reshape((2,1))
                    ax.errorbar(mix,alt,xerr=xerr,yerr=yerr,**kwargs)
                else:
                    ax.errorbar(mix,alt,xerr=xerr,yerr=yerr,**kwargs)

def plot(pc, c):
    sol = pc.return_atmosphere()

    with open('planetary_atmosphere_observations/Jupiter.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    plt.rcParams.update({'font.size': 13})
    fig,axs = plt.subplots(1,2,figsize=[10,3.5],sharex=False,sharey=True)
    fig.patch.set_facecolor("w")

    ax = axs[0]
    species = ['CH4','C2H2','C2H4','C2H6','CO','HCN']
    colors = ['C1','C5','C7','C8','C4','C6']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['pressure']/1e6,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        add_data_to_figure_p(sp, dat, ax, default_error=None, c=colors[i],marker='o',ls='',capsize=1.5,ms=4,elinewidth=0.7, capthick=0.7, alpha=1)
    ax.set_xlim(2e-11,5e-3)
    # ax.legend(ncol=1,bbox_to_anchor=(1.01,1.01),loc='upper right',fontsize=10)
    ax.text(0.98, .98, '(a)', size = 20, ha='right', va='top',transform=ax.transAxes,color='k')
    ax.set_ylabel('Pressure (bar)')
    ax.set_yscale('log')

    species = ['H2O','H2Oaer','NH3','NH3aer']
    colors = ['C0','C0','C9','C9']
    ls = ['-','--','-','--']
    labels = ['H$_2$O', 'H$_2$O cloud', 'NH$_3$', 'NH$_3$ cloud']
    alphas = [1,0.7,1,0.7]
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['pressure']/1e6,label=labels[i], c=colors[i], lw=1.5, ls=ls[i], alpha=alphas[i])
        add_data_to_figure_p(sp, dat, ax, default_error=None, c=colors[i],marker='o',ls='',capsize=1.5,ms=4,elinewidth=0.7, capthick=0.7, alpha=1)
    # ax.set_xlim(1e-10,1)
    ax.legend(ncol=4,bbox_to_anchor=(-0.03,1.01),loc='lower left',fontsize=12)
    # ax.text(0.02, .96, '(b)', size = 20, ha='left', va='top',transform=ax.transAxes,color='k')

    ax.grid(alpha=0.4)
    ax.set_xscale('log')
    ax.set_ylim(1e4,1e-8)
    ax.set_xlabel('Mixing Ratio')
    ax.set_yticks(10.0**np.arange(-8,5,2))
    ax.set_xticks(10.0**np.arange(-10,-2,1))

    ax = axs[1]
    utils.plot_PT(c, ax, lwc=2, color='k', lw=2, ls='--', label='Predicted',zorder=500)

    P, T, Kzz = np.loadtxt('input/Jupiter/Jupiter_deep_top.txt',skiprows=2).T
    inds1 = np.where(P/1e6 < 400e-3)
    inds2 = np.where(P/1e6 >= 400e-3)
    ax.plot(T[inds1], P[inds1]/1e6, 'r', lw=4, label='Moses+2005', alpha=0.3)
    ax.plot(T[inds2], P[inds2]/1e6, 'b', lw=4, label='Dry adiabat', alpha=0.3)
    # ax.plot(T, P/1e6, 'r', lw=3, label='Temperature')

    ax.legend(ncol=1,bbox_to_anchor=(0.98,0.6),loc='upper right',fontsize=12)

    ax.text(0.98, .98, '(b)', size = 20, ha='right', va='top',transform=ax.transAxes,color='k')
    ax.set_xlim(50,2100)
    ax.set_xticks(np.arange(100,2100,300))
    ax.grid(alpha=0.4)
    ax.set_xlabel('Temperature (K)')

    plt.subplots_adjust(wspace=0.05)

    plt.savefig('figures/jupiter.png',dpi=300,bbox_inches='tight')

def main():
    pc = initialize()
    assert pc.find_steady_state()

    # with open('results/Jupiter/atmosphere.pkl','rb') as f:
    #     res = pickle.load(f)
    # pc.initialize_from_dict(res)
    
    # Save photochem result
    res = pc.model_state_to_dict()
    with open('results/Jupiter/atmosphere.pkl','wb') as f:
        pickle.dump(res, f)

    c = climate(pc)

    plot(pc, c)

if __name__ == "__main__":
    main()