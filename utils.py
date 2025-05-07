import numpy as np
import numba as nb
from numba import types
from scipy import constants as const
from scipy import integrate
from copy import deepcopy
from photochem import EvoAtmosphere, PhotoException
from tempfile import NamedTemporaryFile
# from photochem.clima import AdiabatClimate
from clima import AdiabatClimate
import yaml
import re

MyLoader = yaml.SafeLoader
MyLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class RobustData():
    
    def __init__(self):
        self.min_mix_reset = -1e-13
        self.nsteps_total = None
        self.nerrors_total = None
        self.nerrors_before_giveup = 10

class EvoAtmosphereRobust(EvoAtmosphere):

    def __init__(self, mechanism_file, settings_file, flux_file, atmosphere_file=None, data_dir=None):

        with NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(ATMOSPHERE_INIT)
            f.flush()
            name = atmosphere_file
            if name is None:
                name = f.name
            super().__init__(
                mechanism_file, 
                settings_file, 
                flux_file,
                name,
                data_dir
            )

        self.rdat = RobustData()

        # Values in photochem to adjust
        self.var.verbose = 1
        self.var.upwind_molec_diff = True
        self.var.autodiff = True
        self.var.atol = 1.0e-23
        self.var.equilibrium_time = 1e17
        self.set_particle_parameters(1, 100, 10)

    def set_particle_parameters(self, smooth_factor, k_cond, k_evap):
        for i in range(len(self.var.cond_params)):
            self.var.cond_params[i].smooth_factor = smooth_factor
            self.var.cond_params[i].k_cond = k_cond
            self.var.cond_params[i].k_evap = k_evap

    def set_particle_radii(self, radii):
        particle_radius = self.var.particle_radius
        for key in radii:
            ind = self.dat.species_names.index(key)
            particle_radius[ind,:] = radii[key]
        self.var.particle_radius = particle_radius
        self.update_vertical_grid(TOA_alt=self.var.top_atmos)

    def initialize_to_zT(self, z, T, Kzz, mix, P_surf):

        # Copy everything
        z, T, Kzz, mix = deepcopy(z), deepcopy(T), deepcopy(Kzz), deepcopy(mix)

        # Ensure mix sums to 1
        ftot = np.zeros(z.shape[0])
        for key in mix:
            ftot += mix[key]
        for key in mix:
            mix[key] = mix[key]/ftot

        # Compute mubar at all heights
        mu = {}
        for i,sp in enumerate(self.dat.species_names[:-2]):
            mu[sp] = self.dat.species_mass[i]
        mubar = np.zeros(z.shape[0])
        for key in mix:
            mubar += mix[key]*mu[key]

        # Compute pressure from hydrostatic equation
        P = compute_pressure_of_ZT(z, T, mubar, self.dat.planet_radius, self.dat.planet_mass, P_surf)

        # Calculate the photochemical grid
        z_top = z[-1]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z, np.log10(P))
        T_p = np.interp(z_p, z, T)
        Kzz_p = 10.0**np.interp(z_p, z, np.log10(Kzz))
        mix_p = {}
        for sp in mix:
            mix_p[sp] = 10.0**np.interp(z_p, z, np.log10(mix[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Update photochemical model grid
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        # prep the atmosphere
        self.prep_atmosphere(self.wrk.usol)

    def initialize_robust_stepper(self, usol):
        rdat = self.rdat
        rdat.nsteps_total = 0
        rdat.nerrors_total = 0
        self.initialize_stepper(usol)
    
    def robust_step(self):

        rdat = self.rdat

        converged = False
        give_up = False

        if rdat.nsteps_total is None:
            raise PhotoException("You must first initialize a robust stepper with 'initialize_robust_stepper'")
        if self.var.nsteps_before_conv_check >= self.var.nsteps_before_reinit:
            raise PhotoException("`nsteps_before_conv_check` should be < `nsteps_before_reinit`")

        for i in range(1):
            try:
                tn = self.step()
                rdat.nsteps_total += 1
            except PhotoException as e:
                self.initialize_stepper(np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf))
                if rdat.nerrors_total > rdat.nerrors_before_giveup:
                    give_up = True
                    break

            # If converged, then return
            if self.wrk.tn > self.var.equilibrium_time:
                converged = True
                break

            # If converged, then return
            converged = self.check_for_convergence()
            if converged:
                break
            
            # Reinit if time to do that
            if self.wrk.nsteps > self.var.nsteps_before_reinit:
                self.initialize_stepper(np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf))
                break

            # Reinit if negative numbers
            if np.min(self.wrk.mix_history[:,:,0]) < rdat.min_mix_reset:
                self.initialize_stepper(np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf))
                break

        return give_up, converged

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation_z(z, u, planet_radius, planet_mass, ptm):
    P = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(z)
    k_boltz = const.Boltzmann*1e7
    dP_dz = -(mubar*grav*P)/(k_boltz*T*const.Avogadro)
    return np.array([dP_dz])

@nb.experimental.jitclass()
class TempAltMubar:

    z : types.double[:] # type: ignore
    T : types.double[:] # type: ignore
    mubar : types.double[:] # type: ignore

    def __init__(self, z, T, mubar):
        self.z = z.copy()
        self.T = T.copy()
        self.mubar = mubar.copy()

    def temperature_mubar(self, z):
        T = np.interp(z, self.z, self.T)
        mubar = np.interp(z, self.z, self.mubar)
        return T, mubar

def compute_pressure_of_ZT(z, T, mubar, planet_radius, planet_mass, P_surf):

    ptm = TempAltMubar(z, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    # Integrate to TOA
    out = integrate.solve_ivp(hydrostatic_equation_z, [z[0], z[-1]], np.array([P_surf]), t_eval=z, args=args, rtol=1e-6)
    assert out.success

    # Stitch together
    P = out.y[0]

    return P

class AdiabatClimateRobust(AdiabatClimate):

    def __init__(self, species_file, settings_file, star_file, data_dir=None, **kwargs):

        super().__init__(
            species_file, 
            settings_file, 
            star_file,
            data_dir,
            **kwargs
        )

        self.kwargs = kwargs

        with open(species_file,'r') as f:
            self.species_file = yaml.load(f, yaml.Loader)
        with open(settings_file,'r') as f:
            self.settings_file = yaml.load(f, yaml.Loader)
        with open(star_file,'r') as f:
            self.star_file = f.read()
        self.data_dir = data_dir

    def reinit(self):
        with NamedTemporaryFile('w',suffix='.yaml') as f1:
            with NamedTemporaryFile('w',suffix='.yaml') as f2:
                with NamedTemporaryFile('w',suffix='.txt') as f3:
                    yaml.dump(self.species_file, f1, yaml.Dumper)
                    yaml.dump(self.settings_file, f2, yaml.Dumper)
                    f3.write(self.star_file)
                    f3.flush()
                    
                    super().__init__(
                        f1.name, 
                        f2.name, 
                        f3.name,
                        self.data_dir,
                        **self.kwargs
                    )
    
    def reinit_with_op(self, op):
        settings_file = deepcopy(self.settings_file)
        settings_file['optical-properties'] = op

        with NamedTemporaryFile('w',suffix='.yaml') as f1:
            with NamedTemporaryFile('w',suffix='.yaml') as f2:
                with NamedTemporaryFile('w',suffix='.txt') as f3:
                    yaml.dump(self.species_file, f1, yaml.Dumper)
                    yaml.dump(settings_file, f2, yaml.Dumper)
                    f3.write(self.star_file)
                    f3.flush()
                    
                    super().__init__(
                        f1.name, 
                        f2.name, 
                        f3.name,
                        self.data_dir,
                        **self.kwargs
                    )

    def reinit_without_species_opacity(self, species, optype):

        settings_file = deepcopy(self.settings_file)
        optical_properties = yaml.safe_load(self.rad.opacities2yaml())['optical-properties']
        
        for key in ['ir','solar']:
            if optype == 'particle-xs':
                names = [a['name'] for a in optical_properties[key]['opacities'][optype]]
                tmp = [optical_properties[key]['opacities'][optype][i] for i in range(len(names)) if names[i] not in species]
                optical_properties[key]['opacities'][optype] = tmp
            elif optype == 'CIA':
                tmp = []
                if '-' in species:
                    for a in optical_properties[key]['opacities'][optype]:
                        if species != a:
                            tmp.append(a)
                else:
                    for a in optical_properties[key]['opacities'][optype]:
                        b = a.split('-')
                        if species not in b:
                            tmp.append(a)
                optical_properties[key]['opacities'][optype] = tmp
            else:
                tmp = [a for a in optical_properties[key]['opacities'][optype] if a not in species]
                optical_properties[key]['opacities'][optype] = tmp

        settings_file['optical-properties'] = optical_properties

        with NamedTemporaryFile('w',suffix='.yaml') as f1:
            with NamedTemporaryFile('w',suffix='.yaml') as f2:
                with NamedTemporaryFile('w',suffix='.txt') as f3:
                    yaml.dump(self.species_file, f1, yaml.Dumper)
                    yaml.dump(settings_file, f2, yaml.Dumper)
                    f3.write(self.star_file)
                    f3.flush()
                    
                    super().__init__(
                        f1.name, 
                        f2.name, 
                        f3.name,
                        self.data_dir,
                        **self.kwargs
                    )

def species_to_latex(sp):
    sp1 = re.sub(r'([0-9]+)', r"_\1", sp)
    sp1 = r'$\mathrm{'+sp1+'}$'
    if sp == 'O1D':
        sp1 = r'$\mathrm{O(^1D)}$'
    elif sp == 'N2D':
        sp1 = r'$\mathrm{N(^2D)}$'
    elif sp == '1CH2':
        sp1 = r'$\mathrm{^1CH_2}$'
    elif sp == 'H2SO4aer':
        sp1 = r'H$_2$SO$_2$ cloud'
    return sp1

def add_data_to_figure(sp, dat, ax, default_error = None, **kwargs):
    if sp in dat:
        entry = retrieve_species(sp, dat)
        for j,en in enumerate(entry):
            for jj in range(len(en['mix'])):  
                mix = en['mix'][jj]
                alt = en['alt'][jj]
                xerr = en['mix-err'][:,jj].reshape((2,1))
                yerr = en['alt-err'][:,jj].reshape((2,1))
                if xerr[0,0] == mix:
                    ax.errorbar(mix, alt, yerr=yerr, xerr=10.0**(np.log10(mix)-0.1), xuplims=[True],**kwargs)
                elif np.all(xerr.flatten() == np.array([0,0])) and default_error is not None:
                    low = mix - 10.0**(np.log10(mix)-default_error)
                    high = 10.0**(np.log10(mix)+default_error) - mix
                    xerr = np.array([low, high]).reshape((2,1))
                    ax.errorbar(mix,alt,xerr=xerr,yerr=yerr,**kwargs)
                else:
                    ax.errorbar(mix,alt,xerr=xerr,yerr=yerr,**kwargs)

def plot_PT(c, ax, lwc=3, **kwargs):
    T = np.append(c.T_surf,c.T)
    z = np.append(c.P_surf,c.P)/1e6
    p = ax.plot(T, z, **kwargs)
    for a in ['lw','ls','color','label']:
        if a in kwargs:
            kwargs.pop(a)
    color = p[0].get_color()
    for i in range(len(c.convecting_with_below)):
        j = i+1
        if c.convecting_with_below[i]:
            ax.plot(T[j-1:j+1],z[j-1:j+1], lw=lwc, ls='-', color=color, **kwargs)

def retrieve_species(sp, data):
    out = []
    for i in range(len(data[sp])):
        entry = {}
        entry['citation'] = data[sp][i]['citation']
        keys = set([key for key in data[sp][i]['data'][0].keys()])
        P_keys = set(['P', 'P-low', 'P-high'])
        alt_keys = set(['alt', 'alt-low', 'alt-high'])

        if bool(keys & P_keys):
            # We have P vs mix
            entry['P'] = []
            entry['P-err'] = []
            for j in range(len(data[sp][i]['data'])):

                P_low_p = True
                if 'P-low' in data[sp][i]['data'][j]:
                    P_low = data[sp][i]['data'][j]['P-low']
                else:
                    P_low_p = False 
                P_high_p = True
                if 'P-high' in data[sp][i]['data'][j]:
                    P_high = data[sp][i]['data'][j]['P-high']
                else:
                    P_high_p = False
                P_p = True
                if 'P' in data[sp][i]['data'][j]:
                    P = data[sp][i]['data'][j]['P']
                else:
                    P_p = False

                if P_p and P_low_p and P_high_p:
                    entry['P'].append(P)
                    tmp = [P - P_low, P_high - P]
                    entry['P-err'].append(tmp)
                elif not P_p and P_low_p and P_high_p:
                    P_tmp = np.mean([P_low,P_high])
                    entry['P'].append(P_tmp)
                    tmp = [P_tmp - P_low, P_high - P_tmp]
                    entry['P-err'].append(tmp)
                elif P_p and not P_low_p and not P_high_p:
                    entry['P'].append(P)
                    tmp = [0,0]
                    entry['P-err'].append(tmp)
                else:
                    raise Exception("Problem parsing data file")

            entry['P'] = np.array(entry['P'])
            entry['P-err'] = np.array(entry['P-err']).T
        elif bool(keys & alt_keys):
            # We have alt vs mix
            entry['alt'] = []
            entry['alt-err'] = []
            for j in range(len(data[sp][i]['data'])):

                alt_low_p = True
                if 'alt-low' in data[sp][i]['data'][j]:
                    alt_low = data[sp][i]['data'][j]['alt-low']
                else:
                    alt_low_p = False 
                alt_high_p = True
                if 'alt-high' in data[sp][i]['data'][j]:
                    alt_high = data[sp][i]['data'][j]['alt-high']
                else:
                    alt_high_p = False
                alt_p = True
                if 'alt' in data[sp][i]['data'][j]:
                    alt = data[sp][i]['data'][j]['alt']
                else:
                    alt_p = False

                if alt_p and alt_low_p and alt_high_p:
                    entry['alt'].append(alt)
                    tmp = [alt - alt_low, alt_high - alt]
                    entry['alt-err'].append(tmp)
                elif not alt_p and alt_low_p and alt_high_p:
                    alt_tmp = np.mean([alt_low,alt_high])
                    entry['alt'].append(alt_tmp)
                    tmp = [alt_tmp - alt_low, alt_high - alt_tmp]
                    entry['alt-err'].append(tmp)
                elif alt_p and not alt_low_p and not alt_high_p:
                    entry['alt'].append(alt)
                    tmp = [0,0]
                    entry['alt-err'].append(tmp)
                else:
                    raise Exception("Problem parsing data file")

            entry['alt'] = np.array(entry['alt'])
            entry['alt-err'] = np.array(entry['alt-err']).T
        else:
            raise Exception("Problem parsing data file")

        # We have P vs mix
        entry['mix'] = []
        entry['mix-err'] = []
        for j in range(len(data[sp][i]['data'])):

            mix_low_p = True
            if 'mix-low' in data[sp][i]['data'][j]:
                mix_low = data[sp][i]['data'][j]['mix-low']
            else:
                mix_low_p = False 
            mix_high_p = True
            if 'mix-high' in data[sp][i]['data'][j]:
                mix_high = data[sp][i]['data'][j]['mix-high']
            else:
                mix_high_p = False
            mix_p = True
            if 'mix' in data[sp][i]['data'][j]:
                mix = data[sp][i]['data'][j]['mix']
            else:
                mix_p = False

            if mix_p and mix_low_p and mix_high_p:
                entry['mix'].append(mix)
                tmp = [mix - mix_low, mix_high - mix]
                entry['mix-err'].append(tmp)
            elif not mix_p and mix_low_p and mix_high_p:
                mix_tmp = np.mean([mix_low,mix_high])
                entry['mix'].append(mix_tmp)
                tmp = [mix_tmp - mix_low, mix_high - mix_tmp]
                entry['mix-err'].append(tmp)
            elif mix_p and not mix_low_p and not mix_high_p:
                entry['mix'].append(mix)
                tmp = [0,0]
                entry['mix-err'].append(tmp)
            else:
                raise Exception("Problem parsing data file")

        entry['mix'] = np.array(entry['mix'])
        entry['mix-err'] = np.array(entry['mix-err']).T

        out.append(entry)
    return out