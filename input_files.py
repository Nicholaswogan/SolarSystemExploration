import numpy as np
from astropy import constants
from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files, resave_mechanism_with_atoms
from photochem.utils import vulcan2yaml
from photochem.utils._format import yaml, Loader, MyDumper, FormatReactions_main, mechanism_dict_with_atoms
import requests
import zipfile
import io
import os
import shutil

def create_stellar_fluxes():
    _ = stars.solar_spectrum(
        outputfile='input/SunNow.txt',
        stellar_flux=1367,
    )

    # WASP39b
    wv, F = np.loadtxt('input/WASP39b/sflux-wasp39-frances.txt',skiprows=1).T
    # Scale to where WASP39b is
    F = F*((0.932*constants.R_sun.value)/(constants.au.value*0.04828))**2
    # File
    fmt = '{:20}'
    with open('input/WASP39b/WASP39_flux.txt','w') as f:
        f.write(fmt.format('Wavelength (nm)'))
        f.write(fmt.format('Solar flux (mW/m^2/nm)'))
        f.write('\n')
        for i in range(F.shape[0]):
            f.write(fmt.format('%e'%wv[i]))
            f.write(fmt.format('%e'%F[i]))
            f.write('\n')

def get_solarsystem_observations():
    if os.path.isdir('planetary_atmosphere_observations'):
        shutil.rmtree('planetary_atmosphere_observations')
    commit = 'd9dfe305e81a43be19fe23046790e19daaa28fee'
    url = 'https://github.com/Nicholaswogan/planetary_atmosphere_observations/archive/'+commit+'.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")
    os.rename('planetary_atmosphere_observations-'+commit,'planetary_atmosphere_observations')

def reaction_mechanisms():

    # Venus
    with open('input/zahnle_earth.yaml','r') as f:
        dat = yaml.load(f,Loader=Loader)
    for i in range(len(dat['particles'])):
        if dat['particles'][i]['name'] == 'H2SO4aer':
            dat['particles'][i]['optical-properties'] = "none"
    dat = FormatReactions_main(dat)
    with open('input/Venus/zahnle_earth_no_cloud_opacity.yaml', 'w') as f:
        yaml.dump(dat,f,Dumper=MyDumper,sort_keys=False,width=70)
    
    with open('input/zahnle_earth.yaml','r') as f:
        dat = yaml.load(f,Loader=Loader)
    with open('input/Venus/rimmer2021.yaml','r') as f:
        new = yaml.load(f,Loader=Loader)
    dat['species'] += new['species']
    dat['reactions'] += new['reactions']
    dat = FormatReactions_main(dat)
    with open('input/Venus/zahnle_earth_w_rimmer2021.yaml', 'w') as f:
        yaml.dump(dat,f,Dumper=MyDumper,sort_keys=False,width=70)

    generate_thermo(
        'input/zahnle_earth.yaml',
        'input/condensate_thermo.yaml',
        'input/zahnle_earth_thermo.yaml',
        atoms_names=['H','O','N','C','He','S','Cl']
    )

    # Mars and Titan
    resave_mechanism_with_atoms(
        'input/zahnle_earth.yaml',
        'input/zahnle_earth_HNOC.yaml',
        ['H','O','N','C']
    )

    # Jupiter
    resave_mechanism_with_atoms(
        'input/zahnle_earth.yaml',
        'input/zahnle_earth_HNOCHe.yaml',
        ['H','O','N','C','He'],
        remove_reaction_particles=True
    )
    generate_thermo(
        'input/zahnle_earth.yaml',
        'input/condensate_thermo.yaml',
        'input/zahnle_earth_HNOCHe_thermo.yaml',
        atoms_names=['H','O','N','C','He']
    )

    # WASP-39b
    resave_mechanism_with_atoms(
        'input/zahnle_earth.yaml',
        'input/zahnle_earth_HNOCHeS.yaml',
        ['H','O','N','C','He','S'],
        remove_reaction_particles=True
    )
    generate_thermo(
        'input/zahnle_earth.yaml',
        'input/condensate_thermo.yaml',
        'input/zahnle_earth_HNOCHeS_thermo.yaml',
        atoms_names=['H','O','N','C','He','S']
    )

    # VULCAN stuff for WASP-39b
    if os.path.isdir('VULCAN'):
        shutil.rmtree('VULCAN')
    commit = 'f3d7291d69b356a38f18d70a39c41e143eb85cee'
    url = 'https://github.com/exoclime/VULCAN/archive/'+commit+'.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")
    os.rename('VULCAN-'+commit,'VULCAN')
    # Make reactions file
    vulcan2yaml(
        'input/WASP39b/VULCAN/SNCHO_photo_network.txt',
        'VULCAN/thermo',
        'photochem_data'
    )
    # Correct He duplicate reaction
    with open('SNCHO_photo_network.yaml','r') as f:
        dat1 = yaml.load(f, Loader)
    for i in range(len(dat1['reactions'])):
        if dat1['reactions'][i]['equation'] == 'He <=> He':
            dat1['reactions'][i]['equation'] = 'He => He'
    dat1 = FormatReactions_main(dat1)
    with open('SNCHO_photo_network.yaml','w') as f:
        yaml.dump(dat1, f, MyDumper)
    shutil.move('SNCHO_photo_network.yaml','input/WASP39b/SNCHO_photo_network.yaml')
    # Delete VULCAN
    shutil.rmtree('VULCAN')

def generate_thermo(mechanism_file, thermo_file, outfile, atoms_names=None, exclude_species=[], remove_particles=False):

    with open(mechanism_file,'r') as f:
        dat = yaml.load(f, Loader=Loader)

    with open(thermo_file,'r') as f:
        dat1 = yaml.load(f, Loader=Loader)

    # Delete information that is not needed
    for i,atom in enumerate(dat['atoms']):
        del dat['atoms'][i]['redox'] 
    del dat['particles']
    del dat['reactions']

    if not remove_particles:
        for i,sp in enumerate(dat1['species']):
            dat['species'].append(sp)

    if atoms_names is None:
        atoms_names = [a['name'] for a in dat['atoms']]
        
    dat = mechanism_dict_with_atoms(dat, atoms_names, exclude_species)

    dat = FormatReactions_main(dat)

    with open(outfile, 'w') as f:
        yaml.dump(dat,f,Dumper=MyDumper,sort_keys=False,width=70)

def main():
    create_stellar_fluxes()
    get_solarsystem_observations()
    reaction_mechanisms()

if __name__ == '__main__':
    main()