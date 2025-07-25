import numpy as np
from matplotlib import pyplot as plt
from utils import EvoAtmosphereRobust, add_data_to_figure
from photochem.equilibrate import ChemEquiAnalysis
import yaml
from photochem.utils import stars
from photochem.clima import rebin
from scipy import constants as const

def read_crisp_cloud_optical_properties():
    """Reads in the Crisp cloud optical properties
    and puts them on a vertical altitude grid (cm).
    """

    # Get z-P relationship from VIRA
    z, _, P = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    P *= 1e6
    z *= 1e5

    z1 = z # cm
    P1 = P/1e6 # from dynes/cm^2 to bar

    # Flip for interpolation
    z1 = z1[::-1]
    log10P1 = np.log10(P1[::-1])
    
    # Load up David Crisp cloud properties
    tmp = np.loadtxt('input/Venus/clouds/crisp_1986.cld',skiprows=8)
    
    # Get edges of grid in altitude
    Pe = tmp[:,0][::-1]
    ze = np.interp(np. log10(Pe)[::-1], log10P1, z1)[::-1]
    
    # Thickness of each layer
    dz = ze[1:] - ze[:-1]
    
    # Optical depth divided by layer thicknesses
    dtau_ref = {}
    dtau_ref['m1'] = tmp[1:,1][::-1]/dz # tau per cm
    dtau_ref['m2'] = tmp[1:,5][::-1]/dz
    dtau_ref['m2p'] = tmp[1:,7][::-1]/dz
    dtau_ref['m3'] = tmp[1:,9][::-1]/dz
    
    # Use reference optical depth to get optical properties
    # at all wavelengths
    dtau = {}
    dtaus = {}
    qext = {}
    gt = {}
    for j,key in enumerate(dtau_ref):
        
        tmp = np.loadtxt('input/Venus/clouds/venus_h2so4_'+key+'.mie',skiprows=19)
        if j > 0:
            assert np.allclose(wv[1:], tmp[:,0]*1e3)
        wv = np.append(1,tmp[:,0]*1e3)
        ind = np.argmin(np.abs(wv-630))
    
        w0 = tmp[:,9]
        w0 = np.append(w0[0],w0)
        g = tmp[:,10]
        g = np.append(g[0],g)
        qext = tmp[:,6]
        qext = np.append(qext[0],qext)
        
        dtau1 = np.empty((len(dtau_ref[key]),len(wv)))
        for i in range(len(wv)):
            # Equation on pg. 493 of Crisp (1986)
            dtau1[:,i] = (qext[i]/qext[ind])*dtau_ref[key]
        dtau[key] = dtau1
            
        dtaus1 = np.empty((len(dtau_ref[key]),len(wv)))
        g1 = np.empty((len(dtau_ref[key]),len(wv)))
        for i in range(len(ze)-1):
            dtaus1[i,:] = dtau1[i,:]*w0
            g1[i,:] = g
        dtaus[key] = dtaus1
        gt[key] = g1

    # Sum up the optical depths
    dtau_tot = np.zeros_like(dtau['m1'])
    dtaus_tot = np.zeros_like(dtaus['m1'])
    for key in dtaus:
        dtau_tot += dtau[key]
        dtaus_tot += dtaus[key]
    
    # Asymmetry factor
    gt_tot = np.zeros_like(dtaus['m1'])
    for key in dtaus:
        gt_tot += gt[key]*dtaus[key]/np.maximum(dtaus_tot,1e-200)

    res = {
        'ze': ze, # edges of grid in cm
        'wv': wv, # nm
        'dtau': dtau_tot, # tau/cm
        'dtaus': dtaus_tot, # taus/cm
        'gt': gt_tot
    }
    return res

def crisp_clouds_for_clima():
    """Get cloud optical properties for clima.
    """

    z1, _, P1 = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    P1 *= 1e6
    z1 *= 1e5

    res = read_crisp_cloud_optical_properties()
    
    wv = res['wv']
    ze = res['ze']
    dtau_dz = res['dtau']
    w0 = res['dtaus']/np.maximum(res['dtau'],1.0e-200)
    g0 = res['gt']
    
    z = np.empty(dtau_dz.shape[0]*2)
    dtau_dz_1 = np.empty((dtau_dz.shape[0]*2,dtau_dz.shape[1]))
    w0_1 = np.empty((dtau_dz.shape[0]*2,dtau_dz.shape[1]))
    g0_1 = np.empty((dtau_dz.shape[0]*2,dtau_dz.shape[1]))
    
    for i in range(dtau_dz.shape[0]):
        j = i*2
        dtau_dz_1[j,:] = dtau_dz[i,:]
        dtau_dz_1[j+1,:] = dtau_dz[i,:]
    
        w0_1[j,:] = w0[i,:]
        w0_1[j+1,:] = w0[i,:]
    
        g0_1[j,:] = g0[i,:]
        g0_1[j+1,:] = g0[i,:]
    
        z[j] = ze[i]
        z[j+1] = ze[i+1]*0.9999
    
    P = 10.0**np.interp(z,z1,np.log10(P1))

    return z, wv, P, dtau_dz_1, w0_1, g0_1

def crisp_clouds_for_photochem(z, wv):
    """Interpolates the Crisp cloud optical properties to a Photochem altitude
    grid (`z` in cm), and a wavelength grid (`wv` in nm).
    """
    
    # Get optical properties
    res = read_crisp_cloud_optical_properties()
    
    # Make z grid
    ze = stars.make_bins(z)
    dz = ze[1:] - ze[:-1]
    
    # Interpolate to wavelength
    dtau = np.empty((res['dtau'].shape[0],len(wv)))
    dtaus = np.empty((res['dtau'].shape[0],len(wv)))
    gt = np.empty((res['dtau'].shape[0],len(wv)))
    for i in range(dtau.shape[0]):
        dtau[i,:] = np.interp(wv, res['wv'], res['dtau'][i,:])
        dtaus[i,:] = np.interp(wv, res['wv'], res['dtaus'][i,:])
        gt[i,:] = np.interp(wv, res['wv'], res['gt'][i,:])
    
    # Rebin to altitude
    dtau1 = np.empty((len(ze)-1,len(wv)))
    dtaus1 = np.empty((len(ze)-1,len(wv)))
    gt1 = np.empty((len(ze)-1,len(wv)))
    for i in range(len(wv)):
        dtau1[:,i] = rebin(res['ze'].copy(), dtau[:,i].copy(), ze.copy())
        dtaus1[:,i] = rebin(res['ze'].copy(), dtaus[:,i].copy(), ze.copy())
        gt1[:,i] = rebin(res['ze'].copy(), gt[:,i].copy(), ze.copy())
    
    # Multiply by layer thickness
    tau = np.empty_like(dtau1)
    taus = np.empty_like(dtau1)
    for i in range(len(wv)):
        tau[:,i] = dtau1[:,i]*dz
        taus[:,i] = dtaus1[:,i]*dz
    
    # no change here
    g = gt1

    return tau, taus, g

def haus_unknown_uv_absorber(z, wv, uvcase='low'):
    """Haus et al. (2016) unknown UV absorber opacities (tau/cm) on an altitude and
    wavelength grid (`z` in cm and `wv` in nm)
    """

    z1, T1, P1 = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    P1 *= 1e6
    z1 *= 1e5
    density1 = P1/(const.k*1e7*T1)

    # low xs
    if uvcase == 'low':
        with open('input/Venus/clouds/haus_uv_xsec_low.dat','r') as f:
            lines = f.readlines()
    elif uvcase == 'high':
        with open('input/Venus/clouds/haus_uv_xsec_high.dat','r') as f:
            lines = f.readlines()
    else:
        raise Exception()
    wv2 = []
    xs2 = []
    for line in lines[7:]:
        wv2.append(float(line.split()[0]))
        xs2.append(float(line.split()[1]))
    wv2 = np.array(wv2)
    xs2 = np.array(xs2)
    xs2 = np.maximum(xs2,1e-100)

    xs = np.interp(wv,wv2*1e3,np.log10(xs2))
    xs = 10.0**xs
    
    _,_, z2, mix_low,mix_high = np.loadtxt('input/Venus/clouds/venus_haus_uv.cld',skiprows=5).T
    if uvcase == 'low':
        mix = mix_low
    elif uvcase == 'high':
        mix = mix_high
    else:
        raise Exception()
    z2 = z2[::-1].copy()
    mix = mix[::-1].copy()

    density = 10.0**np.interp(z,z1,np.log10(density1))
    
    mix = np.interp(z,z2*1e5,np.log10(mix+1e-200))
    mix = 10.0**mix
    den = density*mix
    
    dtau_dz = np.empty((len(z),len(wv)))
    for i in range(den.shape[0]):
        for j in range(wv.shape[0]):
            dtau_dz[i,j] = den[i]*xs[j]

    return dtau_dz # tau/cm

def unknown_uv_absorber(z, wv, max_wv=400.0):
    """Opacity per cm of the unknown UV absorber from Rimmer et al. (2021).
    `z` in cm, `wv` in nm
    """

    tauc_dz = np.zeros((len(z),len(wv)))
    for i in range(len(z)):
        if z[i] > 67.0e5:
            tauc_dz[i,:] = 0.056e-5*np.exp(-((z[i] - 67.0e5)/3.0e5) - (wv - 360.0)/100.0)
        elif 58.0e5 < z[i] <= 67.0e5:
            tauc_dz[i,:] = 0.056e-5*np.exp(-(wv - 360.0)/100.0)
        else:
            pass

    inds = np.where(wv > max_wv)
    tauc_dz[:,inds] = 0.0

    return tauc_dz # tau/cm

def unknown_uv_absorber_photochem(pc, max_wv=400.0):

    wv = (pc.dat.wavl[1:] + pc.dat.wavl[:-1])/2
    dz = pc.var.z[1] - pc.var.z[0]

    tauc_dz = unknown_uv_absorber(pc.var.z, wv, max_wv)
    tauc = tauc_dz*dz

    return tauc # tau

def apply_custom_opacity_clima(c, crisp_cloud=True, uv=None, max_uv=400):
    "Set the crips cloud optical properties in clima"

    if not crisp_cloud and uv is None:
        # Don't set anything
        # c.rad.unset_custom_optical_properties()
        raise Exception()

    # Crisp Clouds
    z, wv, P, dtau_dz_1, w0_1, g0_1 = crisp_clouds_for_clima()

    # UV absorber
    if uv in ['low','high']:
        # Haus
        dtau_dz_uv = haus_unknown_uv_absorber(z, wv, uvcase=uv)
    elif uv == 'rimmer':
        # Rimmer
        dtau_dz_uv = unknown_uv_absorber(z, wv, max_uv)
    else:
        raise Exception()
    
    if crisp_cloud and uv is not None:
        for i in range(len(wv)):
            dtaus_dz_1 = dtau_dz_1[:,i]*w0_1[:,i]
            dtau_dz_1[:,i] += dtau_dz_uv[:,i]
            w0_1[:,i] = dtaus_dz_1/np.maximum(dtau_dz_1[:,i],1e-100)
    elif crisp_cloud and uv is None:
        pass
    elif not crisp_cloud and uv is not None:
        dtau_dz_1 = dtau_dz_uv
        w0_1 = w0_1*0.0
        g0_1 = g0_1*0.0

    c.rad.set_custom_optical_properties(wv.copy(), P.copy(), dtau_dz_1.copy(), w0_1.copy(), g0_1.copy())

    return z, wv, P, dtau_dz_1, w0_1, g0_1

def apply_custom_opacity_photochem(pc, uv='rimmer', max_wv_uv=400, crisp_clouds=False):

    wv = (pc.dat.wavl[1:] + pc.dat.wavl[:-1])/2

    if uv == 'rimmer':
        # Rimmer et al. UV absorber
        tau_uv = unknown_uv_absorber_photochem(pc, max_wv_uv)
    elif uv in ['low','high']:
        dz = pc.var.z[1] - pc.var.z[0]
        dtau_dz_uv = haus_unknown_uv_absorber(pc.var.z, wv, uvcase=uv)
        tau_uv = dtau_dz_uv*dz

    # Crisp clouds?
    wv = (pc.dat.wavl[1:] + pc.dat.wavl[:-1])/2
    tau_c, taus_c, g0_c = crisp_clouds_for_photochem(pc.var.z, wv)

    if crisp_clouds:
        tau = tau_uv + tau_c
        taus = taus_c
        w0 = taus/np.maximum(tau,1e-200)
        g0 = g0_c
    else:
        tau = tau_uv
        w0 = np.zeros_like(tau)
        g0 = np.zeros_like(tau)
    
    pc.var.tauc = tau
    pc.var.w0c = w0
    pc.var.g0c = g0

    return 

def get_zTKzzmix():
    z, T, P = np.loadtxt('input/venus/venus_seiff1985.txt',skiprows=2).T
    # Extrapolate upward
    z = np.append(z, 110)
    T = np.append(T, T[-1])
    _, _, _, Kzz1, z1 = np.loadtxt('input/venus/venus_rimmer2021.dat',skiprows=2).T
    Kzz = 10.0**np.interp(z, z1/1e5, np.log10(Kzz1))

    z = z*1e5
    P = P*1e6
    P_surf = P[0] # Surface P

    surf_mix = {
        'CO2': 0.965,
        'N2': 3.5e-2,
        'SO2': 100e-6,
        'H2O': 35e-6,
        'OCS': 5.0e-6,
        'CO': 10e-6,
        'HCl': 500e-9,
        'H2': 4.5e-9,
    }
    mix = {a: surf_mix[a]*np.ones(len(z)) for a in surf_mix}

    return z, T, Kzz, mix, P_surf

def initialize(reaction_file='input/zahnle_earth.yaml', atmosphere_file=None, uv='rimmer', max_wv_uv=400, crisp_clouds=False, atol=1e-17):
    
    pc = EvoAtmosphereRobust(
        reaction_file,
        'input/Venus/settings.yaml',
        'input/SunNow.txt',
        atmosphere_file,
        data_dir='photochem_data'
    )

    if atmosphere_file is None:
        # Construct atmosphere
        z, T, Kzz, mix, P_surf = get_zTKzzmix()
        pc.initialize_to_zT(z, T, Kzz, mix, P_surf)

    apply_custom_opacity_photochem(pc, uv=uv, max_wv_uv=max_wv_uv, crisp_clouds=crisp_clouds)
    pc.set_particle_parameters(5, 100, 100)
    pc.set_particle_radii({'H2SO4aer': 2.0e-4, 'H2Oaer': 1.0e-3})

    pc.var.equilibrium_time = 3.15e13
    pc.var.atol = atol

    return pc

def equilibrate_layers(pc):
    cea = ChemEquiAnalysis('input/zahnle_earth_thermo.yaml')

    # Get species composition
    with open('input/zahnle_earth.yaml','r') as f:
        data = yaml.load(f,Loader=yaml.Loader)
    species_composition = {}
    for i,sp in enumerate(data['species']):
        species_composition[sp['name']] = sp['composition']
    for i,sp in enumerate(data['particles']):
        species_composition[sp['name']] = sp['composition']

    dz = pc.var.z[1] - pc.var.z[0]
    eqmix = {}
    for sp in cea.gas_names:
        eqmix[sp] = np.empty(pc.var.nz)
        
    for jj in range(pc.var.nz):

        # Get the column abundance of atoms in a layer
        atom_cols = np.zeros(len(pc.dat.atoms_names))
        for i,sp in enumerate(pc.dat.species_names[:-2-pc.dat.nsl]):
            col = np.sum(pc.wrk.usol[i,jj]*dz)
            for j,atom in enumerate(pc.dat.atoms_names):
                if atom in species_composition[sp]:
                    atom_cols[j] += col*species_composition[sp][atom]

        # Compute the atomic composition
        X = atom_cols/np.sum(atom_cols)

        # Map to photochemical model
        molfracs_atoms = np.zeros(X.shape)
        for i,atom in enumerate(pc.dat.atoms_names):
            ind = cea.atoms_names.index(atom)
            molfracs_atoms[ind] = X[i]
            if atom == 'He':
                molfracs_atoms[i] = 1e-15

        # Equilibrate
        assert cea.solve(pc.wrk.pressure[jj], pc.var.temperature[jj], molfracs_atoms=molfracs_atoms)

        # Save results
        for j,sp in enumerate(cea.gas_names):
            eqmix[sp][jj] = cea.molfracs_species_gas[j]

    return eqmix

def plot(pc, pc1):

    with open('planetary_atmosphere_observations/Venus.yaml','r') as f:
        dat = yaml.load(f,Loader=yaml.Loader)

    sol = pc.mole_fraction_dict()
    sol1 = pc1.mole_fraction_dict()
    eqmix = equilibrate_layers(pc)

    plt.rcParams.update({'font.size': 12})
    fig,axs = plt.subplots(3,4,figsize=[12,6],sharex=True,sharey=True)
    fig.patch.set_facecolor("w")

    axx = []
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axx.append(axs[i,j])

    species = ['SO2','H2O','CO','OCS','HCl','H2S','S3','S4','O2','H2SO4','H2SO4aer','SO']
    labels = ['SO$_2$','H$_2$O','CO','OCS','HCl','H$_2$S','S$_3$','S$_4$','O$_2$','H$_2$SO$_4$','H$_2$SO$_4$ cloud','SO']
    colors = ['C5', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9','C10','C5','C11','C12','C13']
    fig_letter = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
    xlabel = [0.02]*6 + [0.85]*6
    ylabel = [.98]*12
    ylabel[-3] = 0.85
    ylabel[-2] = 0.85
    for i,sp in enumerate(species):
        if sp not in pc.dat.species_names:
            continue
        ax = axx[i]
        ax.plot(sol[sp],sol['alt']/1e5,label=labels[i], c=colors[i], lw=2, alpha=1)
        ax.plot(sol1[sp],sol1['alt']/1e5,label=labels[i], c=colors[i], lw=2, alpha=1,ls='--')
        if sp in eqmix:
            ax.plot(eqmix[sp],sol['alt']/1e5,label=sp, c=colors[i], lw=2, ls=':', alpha=1)
        
        ax.text(0.02, ylabel[i], labels[i], \
                size = 15, ha='left', va='top',transform=ax.transAxes,color=colors[i])
        ax.text(xlabel[i], 0.02, fig_letter[i], \
                size = 15, ha='left', va='bottom',transform=ax.transAxes,color='k')
        
        if sp not in ['O2']:
            add_data_to_figure(sp, dat, ax, c='k',marker='o',ls='',capsize=1.5,ms=1.5,elinewidth=0.7, capthick=0.7, alpha=0.7)
        
    for ax in axx:
        ax.grid(alpha=0.4)
        ax.set_xscale('log')
        ax.set_xlim(1e-13,1e-3)
        ax.set_ylim(0,110)
        ax.set_xticks(10.0**np.arange(-13,-3,2))
        ax.set_yticks(np.arange(0,110,25))

    for i in range(axs.shape[0]):
        axs[i,0].set_ylabel('Altitude (km)')
    for i in range(axs.shape[1]):
        axs[-1,i].set_xlabel('Mixing Ratio')

    ax = axx[0]
    ax1 = ax.twinx()
    ax1.plot([],[],lw=2,ls='-',c='k',label='Photochemistry (nominal)')
    ax1.plot([],[],lw=2,ls='--',c='k',label='Photochemistry (w/ extra Cl chem.)')
    ax1.plot([],[],lw=2,ls=':',c='k',label='Thermochemical Equilibrium')
    ax1.set_yticks([])
    ax1.legend(ncol=3,bbox_to_anchor=(-0.04,1.0),loc='lower left',fontsize=11.5)
        
    plt.subplots_adjust(hspace=.03, wspace=0.03)
    plt.savefig('figures/venus.pdf',bbox_inches = 'tight')

def main():
    # Nominal model
    pc = initialize(
        reaction_file='input/zahnle_earth.yaml',
        atmosphere_file=None,
        crisp_clouds=False
    )

    # First converge with these parameters
    pc.set_particle_radii({'H2SO4aer': 1.0e-4, 'H2Oaer': 1.0e-3})
    apply_custom_opacity_photochem(pc, uv='rimmer', max_wv_uv=1e5, crisp_clouds=False)
    assert pc.find_steady_state()

    # Then converge with these
    apply_custom_opacity_photochem(pc, uv='rimmer', max_wv_uv=400, crisp_clouds=False)
    assert pc.find_steady_state()

    # Then converge with these
    pc.set_particle_radii({'H2SO4aer': 2.0e-4, 'H2Oaer': 1.0e-3})
    assert pc.find_steady_state()

    # Finally converge with these
    pc.var.equilibrium_time = 1e17
    pc.var.atol = 1e-15
    pc.var.nsteps_before_reinit = 100000
    assert pc.find_steady_state()

    pc.out2atmosphere_txt('results/Venus/atmosphere.txt',overwrite=True)

    # With Rimmer et al. chem.
    pc1 = initialize(
        reaction_file='input/Venus/zahnle_earth_w_rimmer2021.yaml',
        atmosphere_file='results/Venus/atmosphere.txt',
        crisp_clouds=False
    )
    pc1.var.atol = 1e-17
    pc1.var.equilibrium_time = 1e11
    assert pc1.find_steady_state()
    pc1.var.nsteps_before_reinit = 100000
    pc1.var.atol = 1e-18
    pc1.var.equilibrium_time = 1e13
    assert pc1.find_steady_state()
    pc1.var.equilibrium_time = 1e16
    pc1.var.atol = 1e-16
    pc1.var.nsteps_before_reinit = 100000
    assert pc1.find_steady_state()
    pc1.out2atmosphere_txt('results/Venus/atmosphere_rimmer.txt',overwrite=True)

    # Plot 
    plot(pc, pc1)

    # Crisp cloud for comparison
    pc1 = initialize(
        reaction_file='input/Venus/zahnle_earth_no_cloud_opacity.yaml',
        atmosphere_file='results/Venus/atmosphere.txt',
        crisp_clouds=True,
        atol=1e-19
    )
    assert pc1.find_steady_state()
    pc1.out2atmosphere_txt('results/Venus/atmosphere_crisp_cloud.txt',overwrite=True)

if __name__ == '__main__':
    main()

