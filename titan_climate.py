import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy import constants as const
import utils
import titan
from photochem.utils import stars

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=4)

def haze_optical_depth(wv, z, P, slope_fudge_factor):

    z1 = stars.make_bins(z)
    dz = z1[1:] - z1[:-1]
    
    # nm, cm, dynes/cm^2
    dtau_0 = (6.270e2*wv**-0.9706) # below 30 km
    dtau_1 = (2.029e4*wv**-1.409) # 30 to 80 km
    dtau_2 = (1.012e7*wv**-2.339) # Above 80 km. 543 km is top of model domain

    # Get a altitude dependence of the opacity
    ind = np.argmin(np.abs(543e5 - z))
    slope = (np.log10(P[0]) - np.log10(P[ind]))/(z[0] - z[ind])

    # We don't know the altitude dependence very well. So I had to
    # play around with it to get a good fit.
    slope = slope*slope_fudge_factor

    # First, do wavelength and altitude dependence
    dtau = np.ones((len(P),len(wv)))*1e-200
    for i in range(len(z)):
        if z[i] < 30e5:
            dtau[i,:] = 10.0**(np.log10(dtau_0) + slope*z[i])
        elif 30e5 <= z[i] < 80e5:
            dtau[i,:] = 10.0**(np.log10(dtau_1) + slope*z[i])
        elif 80e5 <= z[i] < 543e5:
            dtau[i,:] = 10.0**(np.log10(dtau_2) + slope*z[i])
    
    # Now renormalize the opacity so that it has the correct totals
    ind0 = np.argmin(np.abs(30e5 - z))
    ind1 = np.argmin(np.abs(80e5 - z))
    ind2 = np.argmin(np.abs(543e5 - z))
    
    for i in range(len(wv)):
        factor = np.sum(dtau[:ind0,i]*dz[:ind0])/dtau_0[i]
        dtau[:ind0,i] /= factor
    
        factor = np.sum(dtau[ind0:ind1,i]*dz[ind0:ind1])/dtau_1[i]
        dtau[ind0:ind1,i] /= factor
    
        factor = np.sum(dtau[ind1:ind2,i]*dz[ind1:ind2])/dtau_2[i]
        dtau[ind1:ind2,i] /= factor
    
    return dtau

def single_scattering_albedo(wv, z, P):

    wv_tmp, w0_2, w0_1, w0_0 = np.loadtxt('input/Titan/Tomasko2008_w0.txt',skiprows=2).T
    
    w0 = np.empty((len(P),len(wv)))
    for i in range(len(z)):
        if z[i] < 30e5:
            w0[i,:] = np.interp(wv, wv_tmp, w0_0)
        elif 30e5 <= z[i] < 80e5:
            w0[i,:] = np.interp(wv, wv_tmp, w0_1)
        elif 80e5 <= z[i] < 144e5:
            tmp = np.empty(len(wv_tmp))
            for j in range(len(wv_tmp)):
                tmp[j] = np.interp(z[i], [80e5,144e5], [w0_1[j], w0_2[j]])
            w0[i,:] = np.interp(wv, wv_tmp, tmp)
        else:
            w0[i,:] = np.interp(wv, wv_tmp, w0_2)

    return w0

def asymmetry_parameter(wv, z, P):
    wv_tmp, g_1, g_0 = np.loadtxt('input/Titan/Tomasko2008_asymmetry.txt',skiprows=2).T

    g = np.empty((len(P),len(wv)))
    for i in range(len(z)):
        if z[i] < 80e5:
            g[i,:] = np.interp(wv, wv_tmp, g_0)
        else:
            g[i,:] = np.interp(wv, wv_tmp, g_1)

    return g

def tomasko_clouds(slope_fudge_factor=0.6):

    dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
    z = dat[:,0]*1e5
    rho = dat[:,3]
    T_mid = dat[:,6]
    P = get_P_from_T_rho(T_mid, rho)

    wv = np.linspace(350,5000,100) # this impacts bond albedo

    dtau = haze_optical_depth(wv, z, P, slope_fudge_factor)
    w0 = single_scattering_albedo(wv, z, P)
    g = asymmetry_parameter(wv, z, P)

    # Zero out optical depth at edges
    dtau[:,0] = 0
    dtau[:,-1] = 0

    return z.copy(), wv.copy(), P.copy(), dtau.copy(), w0.copy(), g.copy()

def climate(pc, P_top=1.0, c_guess=None, haze=True, tomasko_solar_haze=False, C2H6_C2H2_factor=1.0):

    c = utils.AdiabatClimateRobust(
        'input/species_climate.yaml', 
        'input/Titan/settings_climate.yaml', 
        'input/SunNow.txt',
        data_dir=None
    )
    
    # Mixing ratios
    sol = pc.mole_fraction_dict()
    custom_dry_mix = {'pressure': sol['pressure']}
    P_i = np.ones(len(c.species_names))*1e-15
    for i,sp in enumerate(c.species_names):
        if sp not in sol:
            continue
        custom_dry_mix[sp] = np.maximum(sol[sp],1e-200)
        P_i[c.species_names.index(sp)] = np.maximum(sol[sp][0],1e-30)*sol['pressure'][0]

    custom_dry_mix['C2H6'] *= C2H6_C2H2_factor
    custom_dry_mix['C2H2'] *= C2H6_C2H2_factor

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

    if tomasko_solar_haze:
        # Remove solar haze optical properties
        c.reinit_without_species_opacity('HCaer', 'particle-xs', ['solar'])

        # Layer on the Tomasko haze
        z, wv, P, dtau, w0, g = tomasko_clouds()
        c.rad.set_custom_optical_properties(wv, P, dtau, w0, g)

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
    plt.rcParams.update({'font.size': 13.75})
    fig,axs = plt.subplots(2,3,figsize=[14,7])

    # labels = 'Predicted (predicted clouds & predicted SO$_2$)'

    colors = ['k','0.6','C0']
    labels = [
        'Model w/ predicted haze',
        'Model w/\nTomasko+2008b haze',
        'Model w/o haze',
    ]
    keys = ['(a)','(b)','(c)']

    for i,c in enumerate([c1,c3,c2]):
        ax = axs[0,i]

        ax.text(.5, 1.12, labels[i], \
                    size = 15, ha='center', va='center',transform=ax.transAxes)

        ax.text(-0.3, 1.1, keys[i], \
                    size = 25, ha='left', va='bottom',transform=ax.transAxes)

        utils.plot_PT(c, ax, lwc=4, color=colors[i], lw=2, ls='-', label='Predicted')
        
        dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
        rho = dat[:,3]
        T_mid = dat[:,6]
        T_min = dat[:,5]
        T_max = dat[:,7]
        P_mid = get_P_from_T_rho(T_mid, rho)/1e6
        ax.fill_betweenx(P_mid, T_min, T_max, fc='C3', alpha=0.3, label='Waite+2013')
            
        ax.set_yscale('log')
        ax.set_xticks(np.arange(60,211,30))
        ax.set_ylim(1.5,1e-6)
        ax.set_yticks(10.0**np.arange(-6,1,1))
        ax.grid(alpha=0.4)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (bar)')
        ax.legend(ncol=1,bbox_to_anchor=(0.0, 0.99), loc='upper left', fontsize=10.5,frameon=False)

        ax = axs[1,i]
        z = np.append(0,c.z+c.dz/2)
        P = 10.0**np.interp(z, np.append(0,c.z), np.log10(np.append(c.P_surf,c.P)))
        F = -(c.rad.wrk_ir.fdn_n - c.rad.wrk_ir.fup_n)/1e3
        F = np.append(F[0:-1:2],F[-1])
        ax.plot(F, P/1e6, c=colors[i], lw=2, ls='--', label='Thermal (simulated)')
        
        z = np.append(0,c.z+c.dz/2)
        P = 10.0**np.interp(z, np.append(0,c.z), np.log10(np.append(c.P_surf,c.P)))
        F = (c.rad.wrk_sol.fdn_n - c.rad.wrk_sol.fup_n)/1e3
        F = np.append(F[0:-1:2],F[-1])
        ax.plot(F, P/1e6, c=colors[i], lw=2, label='Solar (simulated)')

        z, F = np.loadtxt('input/Titan/Tomasko2008Solar.txt',skiprows=2).T
        dat = np.loadtxt('input/Titan/Waite2013Model.txt',skiprows=1)
        rho = dat[:,3]
        T_mid = dat[:,6]
        P1 = get_P_from_T_rho(T_mid, rho)/1e6
        z1 = dat[:,0]
        P = 10.0**np.interp(z, z1, np.log10(P1))
        ax.plot(F, P, c='C4', ls=':', label='Measure Solar\n(Tomasko+2008a)',lw=3,zorder=0)

        fdn = c.rad.wrk_sol.fdn_n[-1]/1e3*4
        fup = c.rad.wrk_sol.fup_n[-1]/1e3*4
        albedo = fup/fdn
        
        ax.text(.98, .02, 'Bond alb. = %.2f'%(albedo), \
                size = 12, ha='right', va='bottom',transform=ax.transAxes)

        ax.legend(ncol=1,bbox_to_anchor=(0.01, 0.99), loc='upper left', fontsize=10,frameon=False)
        ax.set_ylabel('Pressure (bar)')
        ax.set_xlabel('Net flux (W m$^{-2}$)')
        
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylim(1.5,1e-6)
        ax.set_yticks(10.0**np.arange(-6,1,1))
        ax.set_xlim(0,3.5)
        ax.grid(alpha=0.4)
        print('Surface Temperature = %.1f K'%c.T_surf)

        rec = matplotlib.patches.Rectangle((-.33,-.3), 1.45, 2.9, fill=False, lw=1.5, clip_on=False,transform=ax.transAxes)
        rec = ax.add_patch(rec)

    plt.subplots_adjust(wspace=0.5,hspace=0.3)

    plt.savefig('figures/titan_climate.pdf',bbox_inches='tight')

def main():
    pc = titan.initialize(atmosphere_file='results/Titan/atmosphere.txt')

    c1 = climate(pc, P_top=1, c_guess=None, haze=True, tomasko_solar_haze=False, C2H6_C2H2_factor=1.0)
    c2 = climate(pc, P_top=1.05, c_guess=None, haze=False, tomasko_solar_haze=False, C2H6_C2H2_factor=1.0)
    c3 = climate(pc, P_top=1, c_guess=None, haze=True, tomasko_solar_haze=True, C2H6_C2H2_factor=1.0)

    plot(c1, c2, c3)

if __name__ == "__main__":
    main()