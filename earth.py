import numpy as np
from matplotlib import pyplot as plt
from utils import add_data_to_figure
from utils import EvoAtmosphereRobust
import utils
import yaml

def get_zTKzzmix():

    z0, P0, T0 = np.loadtxt('input/Earth/PT_CIRA-86.txt',skiprows=2).T
    z = np.arange(0,101.0,1)
    T = np.interp(z, z0, T0)
    P = 10.0**np.interp(z, z0, np.log10(P0))

    z1, Kzz1 = np.loadtxt('input/Earth/eddy_massie.txt',skiprows=2).T
    Kzz = 10.0**np.interp(z,z1,np.log10(Kzz1))
    z = z*1e5
    P = P*1e6

    P_surf = P[0]

    surf_mix = {
        'N2': 0.78,
        'O2': 0.21
    }
    mix = {a : np.ones(len(z))*surf_mix[a] for a in surf_mix}

    return z, T, Kzz, mix, P_surf

def initialize(atmosphere_file=None):
    
    pc = EvoAtmosphereRobust(
        'input/zahnle_earth.yaml',
        'input/Earth/settings.yaml',
        'input/SunNow.txt',
        atmosphere_file,
        data_dir='photochem_data'
    )

    if atmosphere_file is None:
        # Construct atmosphere
        z, T, Kzz, mix, P_surf = get_zTKzzmix()
        pc.initialize_to_zT(z, T, Kzz, mix, P_surf)

    pc.set_particle_parameters(0.1, 1000, 10)
    pc.set_particle_radii({'H2SO4aer': 1.0e-4, 'H2Oaer': 1.0e-3})
    pc.var.atol = 1e-23

    return pc

def plot(pc):

    with open('planetary_atmosphere_observations/Earth.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    sol = pc.mole_fraction_dict()

    plt.rcParams.update({'font.size': 13})
    fig,axs = plt.subplots(1,3,figsize=[13,3.5],sharex=False,sharey=True)
    fig.patch.set_facecolor("w")

    ax = axs[0]
    species = ['H2O','O3','NO','NO2','N2O','HNO3']
    colors = ['C0','C6','C7','C8','C9','C5']
    # coords = [(1e-3, 20),(1e-3, 20)]
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=1.5,ms=1.5,elinewidth=0.7, capthick=0.7, alpha=0.7)
        # ax.text(*coords[i],utils.species_to_latex(sp),size = 12, 
        #     ha='center', va='center',color=colors[i])
    ax.set_xlim(1e-14,1)
    ax.legend(ncol=1,bbox_to_anchor=(1.01,1.01),loc='upper right',fontsize=10)
    ax.text(0.02, .96, '(a)', size = 20, ha='left', va='top',transform=ax.transAxes,color='k')
    ax.set_ylabel('Altitude (km)')

    ax = axs[1]
    species = ['CH4','CO','CO2','H2','OH']
    colors = ['C1','C4','C2','C3','C6']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=1.5,ms=1.5,elinewidth=0.7, capthick=0.7, alpha=0.7)
    ax.set_xlim(1e-15,9e-4)
    ax.legend(ncol=1,bbox_to_anchor=(0,.8),loc='upper left',fontsize=10)
    ax.text(0.02, .96, '(b)', size = 20, ha='left', va='top',transform=ax.transAxes,color='k')

    ax = axs[2]
    species = ['OCS','SO2','H2SO4']
    colors = ['C6','C5','C1']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        add_data_to_figure(sp, dat, ax, default_error=0.5, c=colors[i],marker='o',ls='',capsize=1.5,ms=1.5,elinewidth=0.7, capthick=0.7, alpha=0.7)
    ax.set_xlim(1e-15,3e-9)
    ax.legend(ncol=1,bbox_to_anchor=(0,.85),loc='upper left',fontsize=10)
    ax.text(0.02, .96, '(c)', size = 20, ha='left', va='top',transform=ax.transAxes,color='k')

    for ax in axs:
        ax.grid(alpha=0.4)
        ax.set_xscale('log')
        ax.set_ylim(0,100)
        ax.set_xlabel('Mixing Ratio')
        
    plt.subplots_adjust(hspace=.03, wspace=0.05)
    plt.savefig('figures/earth.png',dpi=300,bbox_inches = 'tight')

def main():
    pc = initialize()
    assert pc.find_steady_state()
    pc.out2atmosphere_txt('results/Earth/atmosphere.txt',overwrite=True)

    plot(pc)

if __name__ == '__main__':
    main()