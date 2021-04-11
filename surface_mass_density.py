# this file use to get the cluster sueface brightness density from NFW model
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from numba import vectorize

# get the velocity of light and in unit Km/s
vc = C.c.to(U.km/U.s).value

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
# Test_model = apcy.Planck15.clone(H0 = 67.66, Om0 = 0.3111) ## for test

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass

kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

def sigma_m(Mc, z, N, c):

    Qc = kpc2m / Msun2kg # recrect parameter for rho_c
    Z = z
    M = 10**Mc ## in unit of M_sun / h

    Ez = np.sqrt(Omega_m*(1+Z)**3 + Omega_k*(1+Z)**2 + Omega_lambda )
    Hz = H0 * Ez

    rhoc = Qc * (3 * Hz**2) / (8 * np.pi*G) ## in unit of M_sun / kpc^3
    rho_c = rhoc / h**2 ## here in unit of M_sun * h^2 / kpc^3
    rho_mean = 200 * rho_c * Omega_m

    Deltac = (200 / 3) * (c**3 / (np.log(1+c) - c / (c+1) ) )
    r200m = (3 * M / (4 * np.pi * rho_mean) )**(1/3)
    # r200c = (3 * M / (4 * np.pi * rho_c * 200) )**(1/3)

    R = np.logspace(-3, np.log10(1.5 * r200m), N) ## in unit of kpc / h

    Nbins = len(R)

    rs = r200m / c

    f0 = 2 * Deltac * rhoc * Omega_m * rs / h**2 ## in unit of M_sun * h / kpc^3

    sigma = np.zeros( Nbins, dtype = np.float)

    for k in range(Nbins):
        x = R[k]/rs
        if x < 1: 
            f1 = np.sqrt(1-x**2)
            f2 = np.sqrt((1-x)/(1+x))
            f3 = x**2-1
            sigma[k] = f0*(1-2*np.arctanh(f2)/f1)/f3
        elif x == 1:
            sigma[k] = f0/3
        else:
            f1 = np.sqrt(x**2-1)
            f2 = np.sqrt((x-1)/(1+x))
            f3 = x**2-1
            sigma[k] = f0*(1-2*np.arctan(f2)/f1)/f3

    return r200m, sigma, R

### sigma for given radius (based on NFW)
@vectorize
def sigmam(r, Mc, z, c):

    Qc = kpc2m / Msun2kg
    Z = z
    M = 10**Mc ## in unit of M_sun / h
    R = r

    Ez = np.sqrt(Omega_m*(1+Z)**3 + Omega_k*(1+Z)**2 + Omega_lambda)
    Hz = H0*Ez

    rhoc = Qc * ( 3 * Hz**2 ) / ( 8 * np.pi * G ) ## here in unit of M_sun / kpc^3
    rho_c = rhoc / h**2 ## here in unit of M_sun * h^2 / kpc^3

    rho_mean = 200 * rho_c * Omega_m

    Deltac = ( 200 / 3 ) * ( c**3 / ( np.log(1+c) - c / (c+1) ) ) 

    r200m = ( 3 * M / ( 4 * np.pi * rho_mean ) )**(1/3)  ## in unit of kpc / h
    r200c = ( 3 * M / ( 4 * np.pi * rho_c * 200 ) )**(1/3)

    rs = r200m / c
    f0 = 2 * Deltac * ( rhoc * Omega_m / h**2 ) * rs # in unit of M_sun * h / kpc^3

    x = R / rs

    if x < 1: 
        f1 = np.sqrt(1-x**2)
        f2 = np.sqrt((1-x)/(1+x))
        f3 = x**2-1
        sigma = f0*(1-2*np.arctanh(f2)/f1)/f3

    elif x == 1:
        sigma = f0/3

    else:
        f1 = np.sqrt(x**2-1)
        f2 = np.sqrt((x-1)/(1+x))
        f3 = x**2-1
        sigma = f0*(1-2*np.arctan(f2)/f1)/f3

    return sigma

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy import integrate as integ

    z0 = 0
    r200m, sigma_0, R_0 = sigma_m(15, z0, 101, 5)
    sigma_0_z0 = sigmam(R_0, 15, z0, 5)

    idx_lim = R_0 <= r200m
    integ_f0 = 2 * np.pi * R_0 * sigma_0_z0
    M_0 = np.log10( integ.simps( integ_f0[ idx_lim ], R_0[idx_lim] ) )

    from colossus.cosmology import cosmology
    from colossus.halo import profile_nfw

    cosmos = cosmology.setCosmology( 'planck18' )
    p_nfw = profile_nfw.NFWProfile( M = 1E15, c = 5, z = z0, mdef = '200m')
    p_Sigma = p_nfw.surfaceDensity( R_0 )

    integ_f1 = 2 * np.pi * R_0 * p_Sigma
    M_1 = np.log10( integ.simps( integ_f1[idx_lim], R_0[idx_lim] ) )

    from cluster_toolkit import density
    from cluster_toolkit import deltasigma
    Sigma_nfw = deltasigma.Sigma_nfw_at_R( R_0 / 1e3, 1e15, 5, Omega_m, )
    Sigma_nfw = Sigma_nfw * 1e6


    print( M_0 )
    print( M_1 )

    plt.figure()
    # plt.plot( R_0, sigma_0, 'g--', label = 'z = 0',alpha = 0.5)
    # plt.plot( R_0, sigma_0_z0, 'r-', label = 'mine',alpha = 0.5)

    plt.plot( R_0, sigma_0_z0, 'r-', label = 'mine', alpha = 0.5)

    #plt.plot( R_0, Sigma_nfw, 'g--', label = 'cluster_toolkit', alpha = 0.5)
    plt.plot( R_0, p_Sigma, 'b:', label = 'colossus', alpha = 0.5)

    plt.legend( loc = 1 )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$R[kpc / h]$')
    plt.ylabel('$\Sigma[M_\odot h kpc^{-2}]$')
    plt.xlim(1e-3, 4e3)
    plt.savefig('/home/xkchen/figs/surface_mass_density.png', dpi = 300)
    plt.show()

    pass

