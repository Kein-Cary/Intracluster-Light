# this file use to get the cluster sueface brightness density from NFW model
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import matplotlib.pyplot as plt
# get the velocity of light and in unit Km/s
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311) ##use the cosmology model Plank 2018 to analysis
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
c = 5. # the concentration
c0 = U.kpc.to(U.m)
c1 = U.M_sun.to(U.kg)
def sigma_m(Mc,z,N):
    Qc = c0/c1 # recrect parameter for rho_c
    Z = z
    M = 10**Mc
    Ez = np.sqrt(Omega_m*(1+Z)**3+Omega_k*(1+Z)**2+Omega_lambda)
    Hz = H0*Ez
    rhoc = Qc*(3*Hz**2)/(8*np.pi*G) # in unit Msun/kpc^3
    Deltac = (200/3)*(c**3/(np.log(1+c)-c/(c+1))) 
    r200 = (3*M/(4*np.pi*rhoc*200))**(1/3) # in unit kpc
    R = np.logspace(-3,np.log10(r200),N)
    Nbins = len(R)
    rs = r200/c
    f0 = 2*Deltac*rhoc*rs
    sigma = np.zeros(Nbins,dtype = np.float)
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
    return r200, sigma, R

def sigma_m_c(Mc,N):
    M = 10**Mc
    rho_0 = (c0/c1)*(3*H0**2)/(8*np.pi*G)
    r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3) 
    R = np.logspace(-3,np.log10(r200_c),N)
    Nbins = len(R)
    rs = r200_c/c
    R_c = R*1
    # next similar variables are for comoving coordinate, with simble "_c"
    rho0_c = M/((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
    r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3)
    f0_c = 2*rho0_c*rs # use for test
    sigma_c = np.zeros(Nbins,dtype = np.float) # comoving coordinate
    for k in range(Nbins):
        x = R[k]/rs
        if x < 1: 
            f1 = np.sqrt(1-x**2)
            f2 = np.sqrt((1-x)/(1+x))
            f3 = x**2-1
            sigma_c[k] = f0_c*(1-2*np.arctanh(f2)/f1)/f3
        elif x == 1:
            sigma_c[k] = f0_c/3
        else:
            f1 = np.sqrt(x**2-1)
            f2 = np.sqrt((x-1)/(1+x))
            f3 = x**2-1
            sigma_c[k] = f0_c*(1-2*np.arctan(f2)/f1)/f3
    return  r200_c, sigma_c, R_c

def sigma_m_unlog(Mc,z,N):
    Qc = c0/c1 # recrect parameter for rho_c
    Z = z
    M = 10**Mc
    Ez = np.sqrt(Omega_m*(1+Z)**3+Omega_k*(1+Z)**2+Omega_lambda)
    Hz = H0*Ez
    rhoc = Qc*(3*Hz**2)/(8*np.pi*G)
    Deltac = (200/3)*(c**3/(np.log(1+c)-c/(c+1))) # in unit Msun/kpc^3
    r200 = (3*M/(4*np.pi*rhoc*200))**(1/3) # in unit kpc
    R = np.linspace(1e-3, r200, N)
    Nbins = len(R)
    rs = r200/c
    f0 = 2*Deltac*rhoc*rs
    sigma = np.zeros(Nbins,dtype = np.float)
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
    sigma[0] = sigma[1]
    return r200, sigma, R

def sigma_m_c_unlog(Mc,N):
    M = 10**Mc
    rho_0 = (c0/c1)*(3*H0**2)/(8*np.pi*G)
    r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3) 
    R = np.linspace(1e-3, r200_c, N)
    Nbins = len(R)
    rs = r200_c/c
    R_c = R*1
    # next similar variables are for comoving coordinate, with simble "_c"
    rho0_c = M/((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
    r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3)
    f0_c = 2*rho0_c*rs # use for test
    sigma_c = np.zeros(Nbins,dtype = np.float) # comoving coordinate
    for k in range(Nbins):
        x = R[k]/rs
        if x < 1: 
            f1 = np.sqrt(1-x**2)
            f2 = np.sqrt((1-x)/(1+x))
            f3 = x**2-1
            sigma_c[k] = f0_c*(1-2*np.arctanh(f2)/f1)/f3
        elif x == 1:
            sigma_c[k] = f0_c/3
        else:
            f1 = np.sqrt(x**2-1)
            f2 = np.sqrt((x-1)/(1+x))
            f3 = x**2-1
            sigma_c[k] = f0_c*(1-2*np.arctan(f2)/f1)/f3
    sigma_c[0] = sigma_c[1]
    return  r200_c, sigma_c, R_c

if __name__ == "__main__":
    
    r200, sigma, R = sigma_m(15, 0, 101)
    r200_c, sigma_c, R_c = sigma_m_c(15, 101)
    plt.plot(R,sigma,'r--',label = 'with z',alpha = 0.5)
    plt.plot(R_c,sigma_c,'b-',label = 'comoving',alpha = 0.5)
    
    r200, sigma, R = sigma_m_unlog(15, 0, 101)
    r200_c, sigma_c, R_c = sigma_m_c_unlog(15, 101)
    plt.plot(R,sigma,'r--',label = 'with z',alpha = 0.5)
    plt.plot(R_c,sigma_c,'b-',label = 'comoving',alpha = 0.5)

    plt.legend(loc = 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$R[kpc]$')
    plt.ylabel('$\Sigma[M_\odot kpc^{-2}]$')
    plt.show()
    
    pass
