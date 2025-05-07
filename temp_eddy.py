import numpy as np
from matplotlib import pyplot as plt

import venus
import earth
import mars
import titan

def main():

    plt.rcParams.update({'font.size': 11})
    fig,axs = plt.subplots(2,3,figsize=[10,6.5])
    fig.patch.set_facecolor("w")

    ax = axs[0,0]
    z, T, Kzz, mix, P_surf = venus.get_zTKzzmix()
    ax.plot(T,z/1e5, c='k')
    ax.set_ylim(0,np.max(z/1e5))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.set_xticks(np.arange(150,751,150))
    ax.grid(alpha=0.4)
    ax.text(.5, 0.98, 'Venus',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.65, 0.4, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.15, .2, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,z/1e5, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')

    ax = axs[0,1]
    z, T, Kzz, mix, P_surf = earth.get_zTKzzmix()
    ax.plot(T,z/1e5, c='k')
    ax.set_ylim(0,np.max(z/1e5))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.set_xticks(np.arange(200,301,25))
    ax.grid(alpha=0.4)
    ax.text(.5, 0.98, 'Earth',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.65, 0.65, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.8, .13, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,z/1e5, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')

    ax = axs[0,2]
    z, T, Kzz, mix, P_surf = mars.get_zTKzzmix()
    ax.plot(T,z/1e5, c='k')
    ax.set_ylim(0,np.max(z/1e5))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(alpha=0.4)
    ax.text(.45, 0.98, 'Mars',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.65, 0.3, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.15, .1, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,z/1e5, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')
    
    ax = axs[1,0]
    z, T, Kzz, mix, P_surf = titan.get_zTKzzmix()
    ax.plot(T,z/1e5, c='k')
    ax.set_ylim(0,np.max(z/1e5))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.set_xticks(np.arange(50,201,50))
    ax.grid(alpha=0.4)
    ax.text(.45, 0.98, 'Titan',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.4, 0.6, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.2, .15, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,z/1e5, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')

    ax = axs[1,1]
    P, T, Kzz = np.loadtxt('input/Jupiter/Jupiter_deep_top.txt',skiprows=2).T
    inds = np.where(P > 1e-2)
    P = P[inds].copy()
    T = T[inds].copy()
    Kzz = Kzz[inds].copy()
    ax.plot(T,P/1e6, c='k')
    ax.set_yscale('log')
    ax.set_ylim(np.max(P/1e6),np.min(P/1e6))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_xticks(np.arange(0,2001,500))
    ax.grid(alpha=0.4)
    ax.text(.55, 0.98, 'Jupiter',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.35, 0.75, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.5, .3, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,P/1e6, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')
    ax1.set_xticks(10.0**np.arange(3,8,2))

    ax = axs[1,2]
    P, T = np.loadtxt('input/WASP39b/atm_W39b_10Xsolar_Twhole_evening_TP_20deg.txt',skiprows=2).T
    Kzz = np.ones(P.shape[0])
    for i in range(P.shape[0]):
        if P[i]/1e6 > 5.0:
            Kzz[i] = 5e7
        else:
            Kzz[i] = 5e7*(5/(P[i]/1e6))**0.4
    ax.plot(T,P/1e6, c='k')
    ax.set_yscale('log')
    ax.set_ylim(np.max(P/1e6),np.min(P/1e6))
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_xticks(np.arange(1000,4001,1000))
    ax.grid(alpha=0.4)
    ax.text(.45, 0.98, 'WASP-39b',size=20, ha='center', va='top',transform=ax.transAxes)
    ax.text(.6, 0.17, 'Temperature',size=11, ha='center', va='bottom',transform=ax.transAxes)
    ax.text(.45, .6, '$K_{zz}$',size=11, ha='center', va='bottom',transform=ax.transAxes, c='C3')
    ax1 = ax.twiny()
    ax1.plot(Kzz,P/1e6, c='C3', ls='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Eddy diffusion (cm$^2$ s$^{-1}$)')

    plt.subplots_adjust(wspace=0.4,hspace=0.55)
    plt.savefig('figures/T_and_Kzz.pdf',bbox_inches='tight')

if __name__ == '__main__':
    main()

