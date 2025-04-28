import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from utils import add_data_to_figure
import utils
import numba as nb
import yaml

def Kzz_titan(z):
    # Loison et al. (2015), Fig. 4
    slope1 = (np.log10(3e6) - np.log10(2e3))/(300 - 80)
    slope2 = (np.log10(5e7) - np.log10(3e6))/(820 - 300)
    y = np.log10(3e6)
    x = 300
    intercept1 = y - slope1*x
    intercept2 = y - slope2*x
    
    if z < 80:
        Kzz = slope1*80 + intercept1
    elif z >= 80 and z < 300:
        Kzz = slope1*z + intercept1
    elif z >= 300 and z < 820:
        Kzz = slope2*z + intercept2
    else:
        Kzz = slope2*820 + intercept2

    return 10.0**Kzz

def get_zTKzzmix():

    # The recommended temperature from Figure 9 in Waite et al. (2013)
    dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
    z1 = dat[:,0]
    T1 = dat[:,6]

    # Regrid
    z = np.linspace(0,1300,1000)
    T = np.interp(z, z1, T1)
    Kzz = np.array([Kzz_titan(a) for a in z])
    z = z*1e5

    P_surf = 1.5e6

    surf_mix = {
    'N2': 1-1.6e-2-5.1e-5,
    'CH4': 1.6e-2,
    'CO': 5.1e-5
    }
    mix = {a : np.ones(len(z))*surf_mix[a] for a in surf_mix}

    return z, T, Kzz, mix, P_surf

def make_rate_fcn(rate1):
    @nb.cfunc(nb.types.void(nb.types.double,nb.types.int32, nb.types.CPointer(nb.types.double)))
    def tmp(tn, nz, rate):
        for i in range(nz):
            rate[i] = rate1[i]
    return tmp

def initialize(atmosphere_file=None):

    pc = utils.EvoAtmosphereRobust(
        'input/zahnle_earth_HNOC.yaml',
        'input/Titan/settings.yaml',
        'input/SunNow.txt',
        atmosphere_file,
        data_dir='photochem_data'
    )

    if atmosphere_file is None:
        z, T, Kzz, mix, P_surf = get_zTKzzmix()
        pc.initialize_to_zT(z, T, Kzz, mix, P_surf)

    # Apply GCR destruction of N2 and production of N and N2D, based on
    # Figure 3 in Lavvas et al. (2008)
    z1, rate1 = np.loadtxt('input/Titan/Lavvas2008GCR.txt',skiprows=2).T
    rate = 10.0**interp1d(z1,np.log10(rate1),fill_value='extrapolate')(pc.var.z/1e5)

    # Here, I assume that paths that make ions ultimately end up as N.
    # Following the yields above Figure 3 in Lavvas et al. (2008), this means
    # 82.5% N and 17.5% N2D.
    rate_N2 = make_rate_fcn(-rate)
    rate_N = make_rate_fcn(82.5e-2*2*rate)
    rate_N2D = make_rate_fcn(17.5e-2*2*rate)

    pc.set_rate_fcn('N2',rate_N2)
    pc.set_rate_fcn('N',rate_N)
    pc.set_rate_fcn('N2D',rate_N2D)

    # 10 micron for everything, except haze.
    radii = {sp: 1.0e-3 for sp in pc.dat.species_names[:pc.dat.np]}
    radii['HCaer1'] = 0.5e-4 # Loosely based on Lavvas et al. (2010)
    radii['HCaer2'] = 0.5e-4
    radii['HCaer3'] = 0.5e-4
    pc.set_particle_radii(radii)
    pc.set_particle_parameters(1, 100, 0) # No evaporation

    pc.var.equilibrium_time = 1e17
    pc.var.atol = 1e-23

    return pc

def plot(pc):

    with open('planetary_atmosphere_observations/Titan.yaml','r') as f:
        dat1 = yaml.load(f,Loader=utils.MyLoader)

    sol = pc.mole_fraction_dict()

    plt.rcParams.update({'font.size': 13})
    fig,axs = plt.subplots(1,3,figsize=[13,3.5],sharex=False,sharey=True)
    fig.patch.set_facecolor("w")

    for ax in axs:
        ax.set_xlabel('Mixing ratio')
        ax.set_xscale('log')
        ax.grid(alpha=0.4)

    ax = axs[0]
    species = ['CH4','H2']
    colors = ['C1','C3']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)

    dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
    z = dat[:,0]
    CH4_low = dat[:,8]
    CH4_mid = dat[:,9]
    CH4_high = dat[:,10]
    ax.fill_betweenx(z, CH4_low, CH4_high, fc='C1', alpha=0.2)

    H2_low = dat[:,14]
    H2_mid = dat[:,15]
    H2_high = dat[:,16]
    ax.fill_betweenx(z, H2_low, H2_high, fc='C3', alpha=0.2)

    ax.legend(ncol=1,bbox_to_anchor=(1.01,1.01),loc='upper right',fontsize=10)
    ax.set_xlim(5e-4,9e-1)
    ax.set_ylim(0,pc.var.top_atmos/1e5)
    ax.set_ylabel('Altitude (km)')
    ax.text(0.92, .02, '(a)', size = 20, ha='right', va='bottom',transform=ax.transAxes,color='k')

    ax = axs[1]
    species = ['C2H2','C2H4','C2H6','HCN','HCCCN','CH3CN']
    colors = ['C5','C7','C8','C6','C3','C2']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        utils.add_data_to_figure(sp, dat1, ax, c=colors[i],marker='o',ls='',capsize=1.5,ms=3.5,elinewidth=0.7, capthick=0.7, alpha=0.7)

    ax.legend(ncol=1,bbox_to_anchor=(0.0,1.01),loc='upper left',fontsize=10)
    ax.set_xlim(1e-10,2e-2)
    ax.set_xticks(10.0**np.arange(-10,0,2))
    ax.text(0.92, .02, '(b)', size = 20, ha='right', va='bottom',transform=ax.transAxes,color='k')

    ax = axs[2]
    species = ['H2O','CO2','CO','NH3']
    colors = ['C0','C2','C4','C9']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],sol['alt']/1e5,label=utils.species_to_latex(sp), c=colors[i], lw=1.5)
        utils.add_data_to_figure(sp, dat1, ax, c=colors[i],marker='o',ls='',capsize=1.5,ms=3.5,elinewidth=0.7, capthick=0.7, alpha=0.7)

    ax.legend(ncol=1,bbox_to_anchor=(0.5,0.1),loc='lower left',fontsize=10)
    ax.set_xlim(1e-11,1e-4)
    ax.set_xticks(10.0**np.arange(-10,-3,2))
    ax.text(0.92, .02, '(c)', size = 20, ha='right', va='bottom',transform=ax.transAxes,color='k')

    plt.subplots_adjust(wspace=0.05)
    plt.savefig('figures/titan.pdf',bbox_inches = 'tight')

def main():
    pc = initialize()
    assert pc.find_steady_state()
    pc.out2atmosphere_txt('results/Titan/atmosphere.txt',overwrite=True)

    plot(pc)

if __name__ == "__main__":
    main()
