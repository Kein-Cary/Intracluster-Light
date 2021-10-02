# this file use to get the cluster sueface brightness density from NFW model
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from numba import vectorize
import scipy.interpolate as interp

### constant
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass

kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

def input_cosm_model( get_model = None ):

    global Test_model

    if get_model is not None:

        Test_model = get_model

    else:
        ### cosmology
        Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)

    return Test_model

def cosmos_param():

    global H0, h, Omega_m, Omega_lambda, Omega_k

    ## cosmology params
    H0 = Test_model.H0.value
    h = H0 / 100
    Omega_m = Test_model.Om0
    Omega_lambda = 1.-Omega_m
    Omega_k = 1.- (Omega_lambda + Omega_m)

    return

### universe matter density
def rhom_set( z ):

    Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
    Hz = H0 * Ez

    Qc = kpc2m / Msun2kg
    rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

    rho_c = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

    rho_m = rho_c * Omega_m * ( z + 1 )**3 / Ez**2 ## mean matter density of universe at z

    return rho_c, rho_m

### NFW profile
def rho_nfw_delta_m(r, z, c_mass, lgM, v_m = 200):

    Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
    Hz = H0 * Ez

    Qc = kpc2m / Msun2kg
    rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

    rhoc = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

    omega_mz = Omega_m * (z + 1)**3 / Ez**2
    rhom = rhoc * omega_mz

    delta_c = v_m * c_mass**3 / ( 3 * ( np.log(1 + c_mass) - c_mass / ( 1 + c_mass) ) ) 

    M = 10**lgM # in unit of M_sun / h

    r_200m = ( 3 * M / (4 * np.pi * rhom * v_m) )**(1/3)
    rs = r_200m / c_mass

    rho = delta_c * rhom / ( (r / rs) * (1 + r / rs)**2 ) # in unit of M_sun * h^2 / kpc^3

    return rho

def rho_nfw_delta_c(r, z, c_mass, lgM, v_m = 200):

    Ez = np.sqrt( Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
    Hz = H0 * Ez

    Qc = kpc2m / Msun2kg
    rho_c = Qc * ( 3 * Hz**2 ) / (8 * np.pi * G) # here in unit of M_sun / kpc^3

    rhoc = rho_c / h**2 ## here in unit of M_sun * h^2 / kpc^3

    delta_c = v_m * c_mass**3 / ( 3 * ( np.log(1 + c_mass) - c_mass / ( 1 + c_mass) ) ) 

    M = 10**lgM # in unit of M_sun / h

    r_200c = ( 3 * M / (4 * np.pi * rhoc * v_m) )**(1/3)
    rs = r_200c / c_mass

    rho = delta_c * rhoc / ( (r / rs) * (1 + r / rs)**2 ) # in unit of M_sun * h^2 / kpc^3

    return rho

### === ### miscentering nfw profile (Zu et al. 2021, section 3.)
def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

    NR = len( rp )
    aveg_sigma = np.zeros( NR, dtype = np.float32 )

    tR = rp
    intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

    cumu_mass = np.zeros( NR, )
    lg_r_min = np.log10( np.min( rp ) / 10 )

    for ii in range( NR ):

        new_rp = np.logspace( lg_r_min, np.log10( tR[ii] ), N_grid)
        new_sigma = intep_sigma_F( new_rp )

        cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

        aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

    return aveg_sigma

def mis_p_func( r_off, sigma_off):
    """
    r_off : the offset between cluster center and BCGs
    sigma_off : characteristic offset
    """

    pf0 = r_off / sigma_off**2
    pf1 = np.exp( - r_off / sigma_off )

    return pf0 * pf1

def misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

    theta = np.linspace( 0, 2 * np.pi, 100)
    d_theta = np.diff( theta )
    N_theta = len( theta )

    try:
        NR = len( rp )
    except:
        rp = np.array( [rp] )
        NR = len( rp )

    r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
    off_pdf = mis_p_func( r_off, sigma_off )
    dr_off = np.diff( r_off )

    NR_off = len( r_off )

    surf_dens_off = np.zeros( NR, dtype = np.float32 )

    for ii in range( NR ):

        surf_dens_arr = np.zeros( (NR_off, N_theta), dtype = np.float32 )

        for jj in range( NR_off ):

            r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )
            surf_dens_arr[jj,:] = sigmam( r_cir, lgM, z, c_mass,)

        ## integration on theta
        medi_surf_dens = ( surf_dens_arr[:,1:] + surf_dens_arr[:,:-1] ) / 2
        sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1) / ( 2 * np.pi )

        ## integration on r_off
        integ_f = sum_theta_fdens * off_pdf

        medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

        surf_dens_ii = np.sum( medi_integ_f * dr_off )

        surf_dens_off[ ii ] = surf_dens_ii

    off_sigma = surf_dens_off

    if NR == 1:
        return off_sigma[0]
    return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

    off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
    norm_sigma = sigmam( rp, lgM, z, c_mass)

    obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

    return obs_sigma

### sigma for given radius (based on NFW)
@vectorize
def sigmam(r, Mc, z, c):

    Qc = kpc2m / Msun2kg
    Z = z
    M = 10**Mc ## in unit of M_sun / h

    v_m = 200

    R = r

    Ez = np.sqrt(Omega_m*(1+Z)**3 + Omega_k*(1+Z)**2 + Omega_lambda)
    Hz = H0 * Ez

    rhoc = Qc * ( 3 * Hz**2 ) / ( 8 * np.pi * G ) ## here in unit of M_sun / kpc^3
    rho_c = rhoc / h**2 ## here in unit of M_sun * h^2 / kpc^3

    omega_mz = Omega_m * ( Z + 1 )**3 / Ez**2

    rho_mean = v_m * rho_c * omega_mz

    Deltac = ( v_m / 3 ) * ( c**3 / ( np.log(1+c) - c / (c+1) ) ) 

    r200m = ( 3 * M / ( 4 * np.pi * rho_mean ) )**(1/3)  ## in unit of kpc / h

    rs = r200m / c
    f0 = 2 * Deltac * ( rhoc * omega_mz / h**2 ) * rs # in unit of M_sun * h / kpc^2

    x = R / rs

    if x < 1: 
        f1 = np.sqrt(1-x**2)
        f2 = np.sqrt((1-x)/(1+x))
        f3 = x**2-1
        sigma = f0 * ( 1 - 2 * np.arctanh(f2) / f1 ) / f3

    elif x == 1:
        sigma = f0 / 3

    else:
        f1 = np.sqrt(x**2-1)
        f2 = np.sqrt((x-1)/(1+x))
        f3 = x**2-1
        sigma = f0 * ( 1 - 2 * np.arctan(f2) / f1 ) / f3

    return sigma

@vectorize
def sigmac(r, Mc, z, c):

    Qc = kpc2m / Msun2kg
    Z = z
    M = 10**Mc ## in unit of M_sun / h

    v_m = 200

    R = r

    Ez = np.sqrt(Omega_m*(1+Z)**3 + Omega_k*(1+Z)**2 + Omega_lambda)
    Hz = H0*Ez

    rhoc = Qc * ( 3 * Hz**2 ) / ( 8 * np.pi * G ) ## here in unit of M_sun / kpc^3
    rho_c = rhoc / h**2 ## here in unit of M_sun * h^2 / kpc^3

    Deltac = ( v_m / 3 ) * ( c**3 / ( np.log(1+c) - c / (c+1) ) ) 

    r200c = ( 3 * M / ( 4 * np.pi * rho_c * v_m ) )**(1/3)

    rs = r200c / c
    f0 = 2 * Deltac * ( rhoc / h**2 ) * rs # in unit of M_sun * h / kpc^2

    x = R / rs

    if x < 1: 
        f1 = np.sqrt(1-x**2)
        f2 = np.sqrt((1-x)/(1+x))
        f3 = x**2-1
        sigma = f0 * ( 1 - 2 * np.arctanh(f2) / f1 ) / f3

    elif x == 1:

        sigma = f0 / 3

    else:
        f1 = np.sqrt(x**2-1)
        f2 = np.sqrt((x-1)/(1+x))
        f3 = x**2-1
        sigma = f0 * ( 1 - 2 * np.arctan(f2) / f1 ) / f3

    return sigma

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from colossus.cosmology import cosmology
    from colossus.halo import profile_nfw

    cosmos = cosmology.setCosmology( 'planck18' )
    input_cosm_model( get_model = cosmos )
    cosmos_param()

    # input_cosm_model() ## my previous setting
    # cosmos_param()

    z0 = 0.50
    v_m = 200
    Mh0 = 14
    c0 = 5

    R_0 = np.logspace(0, 3.5, 100)

    sigma_m = sigmam(R_0, Mh0, z0, c0)
    sigma_c = sigmac(R_0, Mh0, z0, c0)


    p_nfw = profile_nfw.NFWProfile( M = 1E14, c = c0, z = z0, mdef = '200m')
    p_Sigma = p_nfw.surfaceDensity( R_0 )

    print( p_Sigma / sigma_m )


    p_nfw_c = profile_nfw.NFWProfile( M = 1E14, c = c0, z = z0, mdef = '200c')
    p_Sigma_c = p_nfw_c.surfaceDensity( R_0 )

    print( p_Sigma_c / sigma_c )


    rho_c, rho_m = rhom_set( z0 )
    c_rho_c = cosmos.rho_c( z0 )
    c_rho_m = cosmos.rho_m( z0 )

    print( c_rho_c / rho_c )
    print( c_rho_m / rho_m )


    plt.figure()

    plt.plot( R_0, sigma_m, 'r-', label = 'mine', alpha = 0.5)
    plt.plot( R_0, p_Sigma, 'b:', label = 'colossus', alpha = 0.5)

    plt.plot( R_0, sigma_c, 'm-', label = 'mine', alpha = 0.5)
    plt.plot( R_0, p_Sigma_c, 'c:', label = 'colossus', alpha = 0.5)

    plt.legend( loc = 1 )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$R[kpc / h]$')
    plt.ylabel('$\Sigma[M_\odot h kpc^{-2}]$')
    plt.xlim(1e0, 4e3)
    plt.savefig('/home/xkchen/figs/surface_mass_density.png', dpi = 300)
    plt.show()

    pass
