from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.utils._format import yaml, Loader, MyDumper, FormatReactions_main
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

def get_solarsystem_observations():
    if os.path.isdir('planetary_atmosphere_observations'):
        shutil.rmtree('planetary_atmosphere_observations')
    commit = 'f09a62d30e38b4758ae95994e887decaf9f6ce6b'
    url = 'https://github.com/Nicholaswogan/planetary_atmosphere_observations/archive/'+commit+'.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")
    os.rename('planetary_atmosphere_observations-'+commit,'planetary_atmosphere_observations')

def reaction_mechanisms_venus():
    with open('input/zahnle_earth.yaml','r') as f:
        dat = yaml.load(f,Loader=Loader)

    for i in range(len(dat['particles'])):
        if dat['particles'][i]['name'] == 'H2SO4aer':
            dat['particles'][i]['optical-properties'] = "none"

    dat = FormatReactions_main(dat)

    with open('input/Venus/zahnle_earth_no_cloud_opacity.yaml', 'w') as f:
        yaml.dump(dat,f,Dumper=MyDumper,sort_keys=False,width=70)

def main():
    create_stellar_fluxes()
    get_solarsystem_observations()
    reaction_mechanisms_venus()

if __name__ == '__main__':
    main()