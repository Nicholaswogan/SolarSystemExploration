import os
import subprocess
import shutil
import requests
import zipfile
import io

def main():
    
    # Download
    if os.path.isdir('VULCAN'):
        shutil.rmtree('VULCAN')
    commit = 'f3d7291d69b356a38f18d70a39c41e143eb85cee'
    url = 'https://github.com/exoclime/VULCAN/archive/'+commit+'.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("./")
    os.rename('VULCAN-'+commit,'VULCAN')

    # Copy config in
    shutil.copyfile('vulcan_cfg.py','VULCAN/vulcan_cfg.py')

    # Compile fastchem
    subprocess.call('make'.split(), cwd='VULCAN/fastchem_vulcan/')

    # Adjust eddy diffusion so that it is **0.5 instead of **0.4
    # with open('VULCAN/build_atm.py','r') as f:
    #     lines = f.readlines()
    # lines[396] = '            data_atm.Kzz = vulcan_cfg.K_max * (vulcan_cfg.K_p_lev*1e6 /(data_atm.pico[1:-1]))**0.5\n'
    # with open('VULCAN/build_atm.py','w') as f:
    #     for line in lines:
    #         f.write(line)

    # Run VULCAN
    subprocess.call('python vulcan.py'.split(), cwd='VULCAN/')

    # Save results
    shutil.copyfile('VULCAN/output/20deg/wasp39b_10Xsolar_evening.vul','./wasp39b_10Xsolar_evening.vul')

    # # Delete
    shutil.rmtree('VULCAN')

if __name__ == '__main__':
    main()
