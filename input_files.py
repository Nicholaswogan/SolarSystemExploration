from photochem.utils import stars
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.utils._format import yaml, Loader, MyDumper, FormatReactions_main
import requests
import zipfile
import io
import os

def create_stellar_fluxes():
    _ = stars.solar_spectrum(
        outputfile='input/SunNow.txt',
        stellar_flux=1367,
    )

def get_solarsystem_observations():
    url = 'https://github.com/Nicholaswogan/planetary_atmosphere_observations/archive/c4eb1cb15a45300a555951334ec6e0dfb5f81f5c.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")
    os.rename('planetary_atmosphere_observations-c4eb1cb15a45300a555951334ec6e0dfb5f81f5c','planetary_atmosphere_observations')

def reaction_mechanisms_venus():
    with open('input/zahnle_earth.yaml','r') as f:
        dat = yaml.load(f,Loader=Loader)

    for i in range(len(dat['particles'])):
        if dat['particles'][i]['name'] == 'H2SO4aer':
            dat['particles'][i]['optical-properties'] = "none"

    dat = FormatReactions_main(dat)

    with open('input/Venus/zahnle_earth_no_cloud_opacity.yaml', 'w') as f:
        yaml.dump(dat,f,Dumper=MyDumper,sort_keys=False,width=70)

if __name__ == '__main__':
    # create_stellar_fluxes()
    # get_solarsystem_observations()
    reaction_mechanisms_venus()