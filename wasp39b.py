import numpy as np
from matplotlib import pyplot as plt
from photochem.extensions import gasgiants
import utils
import numba as nb
import yaml
import pickle
from astropy import constants
import numba as nb

def initialize(mechanism_file, data_dir, model_state_file=None):

    pc = gasgiants.EvoAtmosphereGasGiant(
        mechanism_file,
        'input/WASP39b/WASP39_flux.txt',
        planet_mass=0.28*constants.M_jup.cgs.value, # Matches VULCAN gravity at 50 bar.
        planet_radius=1.279*7.1492e9, # What VULCAN uses
        solar_zenith_angle=83, # Used in VULCAN.
        P_ref=50e6, # matches VULCAN
        nz=150, # matches vulcan
        data_dir=data_dir
    )
    pc.gdat.verbose = True
    pc.var.diurnal_fac = 1.0 # matches VULCAN
    pc.gdat.TOA_pressure_avg = 3e-3

    # Set atomic composition to VULCAN simulation
    comp = {
        'O': 5.37E-4,
        'C': 2.95E-4,
        'N': 7.08E-5,
        'S': 1.41E-5,
        'He': 0.0838,
        'H': 1
    }
    # comp = get_vulcan_composition()
    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot
    molfracs_atoms_sun = np.ones(len(pc.gdat.gas.atoms_names))*1e-10
    for i,atom in enumerate(pc.gdat.gas.atoms_names):
        molfracs_atoms_sun[i] = comp[atom]
    pc.gdat.gas.molfracs_atoms_sun = molfracs_atoms_sun

    # P and T
    P, T = np.loadtxt('input/WASP39b/atm_W39b_10Xsolar_Twhole_evening_TP_20deg.txt',skiprows=2).T

    # Assumed Kzz (cm^2/s) in Tsai et al. (2023)
    Kzz = np.ones(P.shape[0])
    for i in range(P.shape[0]):
        if P[i]/1e6 > 5.0:
            Kzz[i] = 5e7
        else:
            Kzz[i] = 5e7*(5/(P[i]/1e6))**0.4

    # Initialize
    pc.initialize_to_climate_equilibrium_PT(P, T, Kzz, 10.0, 1.0)

    if model_state_file is not None:
        with open(model_state_file,'rb') as f:
            res = pickle.load(f)
        pc.initialize_from_dict(res)
    
    return pc

def plot(pc1, pc2):

    with open('input/WASP39b/VULCAN/wasp39b_10Xsolar_evening.vul','rb') as f:
        vulcan = pickle.load(f)
    vulcan_species = vulcan['variable']['species']

    plt.rcParams.update({'font.size': 13.5})
    fig,axs = plt.subplots(1,2,figsize=[10,3.5], sharey=True, sharex=True)
    fig.patch.set_facecolor("w")

    ax = axs[0]

    sol1 = pc1.return_atmosphere()

    species = ['H2S','S','S2','SO','SO2']
    colors = ['C8','C4','C6','C13','C5']
    for i,sp in enumerate(species):
        ax.plot(sol1[sp],sol1['pressure']/1e6,c=colors[i], ls='--', lw=2)
        ax.plot([],[],label=utils.species_to_latex(sp),c=colors[i], ls='-', lw=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-8,1e-3)
    ax.set_xticks(10.0**np.arange(-8,-3,1))
    ax.set_ylim(1e1,1e-7)
    ax.set_yticks(10.0**np.arange(-7,2,1))
    ax.grid(alpha=0.4)
    ax.legend(ncol=10,bbox_to_anchor=(1.025,1.01),loc='lower center')
    ax.set_xlabel('Mixing Ratio')
    ax.set_ylabel('Pressure (bar)')

    ax1 = ax.twinx()
    ax1.set_yticks([])
    ax1.plot([],[],c='k',ls='--',lw=2,label='Photochem w/ Photochem chem.')
    ax1.legend(ncol=10,bbox_to_anchor=(0.48,0.35),loc='upper center',fontsize=10)

    sol2 = pc2.return_atmosphere()

    ax = axs[1]
    for i,sp in enumerate(species):
        ax.plot(sol2[sp],sol2['pressure']/1e6,c=colors[i], ls='-', marker='', ms=3, lw=2,label=sp)
        ind = vulcan_species.index(sp)
        ax.plot(vulcan['variable']['ymix'][:,ind],vulcan['atm']['pco']/1e6,c=colors[i], ls=':', lw=2)

    ax.grid(alpha=0.4)
    ax.set_xlabel('Mixing Ratio')

    ax1 = ax.twinx()
    ax1.set_yticks([])
    ax1.plot([],[],c='k',ls='-',lw=2,label='Photochem w/ VULCAN chem.')
    ax1.plot([],[],c='k',ls=':',lw=2,label='VULCAN w/ VULCAN chem.')
    ax1.legend(ncol=1,bbox_to_anchor=(0.5,0.35),loc='upper center',fontsize=10)

    plt.subplots_adjust(wspace=0.05)

    plt.savefig('figures/wasp39b.pdf',bbox_inches='tight')


def main():

    # Nominal case
    pc1 = initialize('input/zahnle_earth_HNOCHeS.yaml', 'photochem_data')
    assert pc1.find_steady_state()
    with open('results/WASP39b/atmosphere.pkl','wb') as f:
        pickle.dump(pc1.model_state_to_dict(), f)

    # Using VULCAN network
    pc2 = initialize('input/WASP39b/SNCHO_photo_network.yaml', 'vulcandata')
    assert pc2.find_steady_state()
    with open('results/WASP39b/atmosphere_vulcan.pkl','wb') as f:
        pickle.dump(pc2.model_state_to_dict(), f)

    # Plot
    plot(pc1, pc2)

if __name__ == "__main__":
    main()






