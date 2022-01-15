import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import scipy.interpolate as interp
import scipy.stats as sts
from scipy.interpolate import splev, splrep

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy.stats import binned_statistic as binned
from scipy import optimize
from scipy.optimize import fmin

import emcee
import ezgal
from multiprocessing import Pool

# from mpi4py import MPI
# commd = MPI.COMM_WORLD
# rank = commd.Get_rank()
# cpus = commd.Get_size()

## cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['g', 'r', 'i']
l_wave = np.array([4686, 6166, 7480])

Jky = 10**(-23) # erg * s^(-1) * cm^(-2) * Hz^(-1)
F0 = 3.631 * 10**(-6) * Jky

def z_at_lb_time( z_min, z_max, dt, N_grid = 100):
    """
    dt : time interval, in unit of Gyr
    """
    z_arr = np.linspace( z_min, z_max, N_grid)
    t_arr = Test_model.lookback_time( z_arr ).value ## unit Gyr

    lb_time_low = Test_model.lookback_time( z_min ).value
    lb_time_up = Test_model.lookback_time( z_max ).value

    intep_f = splrep( t_arr, z_arr )
    equa_dt = np.arange( lb_time_low, lb_time_up, dt )
    equa_dt_z = splev( equa_dt, intep_f )

    return equa_dt_z

def find_mstar2(model, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr, zfarr, Rabs_guess):

    # eliminate crazy Rabs
    def log_prior(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr, Rabs_guess):
        Rabs = theta[0]
        zf = theta[1]
        zfmin = zs + 0.2
        # if (np.abs(Rabs - Rabs_guess) >= 1.5) or (zf < zfmin) or (zf > 8.0):
        if (np.abs(Rabs - Rabs_guess) >= 1.5) or (zf < zfmin):
            # if (np.abs(Rabs - Rabs_guess) >= 1.5) or (zf < 2.0) or (zf > 6.0):
            return -np.inf
        else:
            return 0.0

    # chi2
    def log_likelihood(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                       Rabs_guess):
        # print theta
        Rabs = theta[0]
        zf = theta[1]
        model.set_normalization('sloan_r', zs, Rabs)
        Gezgal, Rezgal, Iezgal = model.get_apparent_mags(
            zf=zf, filters=['sloan_g', 'sloan_r', 'sloan_i'], zs=zs)[0]
        chi2g = ((Gobs - Gezgal) / Gerr)**2
        chi2r = ((Robs - Rezgal) / Rerr)**2
        chi2i = ((Iobs - Iezgal) / Ierr)**2
        logp = 0.0 - (chi2g + chi2r + chi2i) / 2.
        if np.isnan(logp):
            print("found nan",)
            print(zs,)
            print(zf)
            # print chi2g,
            # print chi2r,
            # print chi2i
        return logp

    # logp
    def log_posterior(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                      Rabs_guess):
        logprior = log_prior(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                             Rabs_guess)
        if logprior == 0.0:
            loglike = log_likelihood(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr,
                                     Ierr, Rabs_guess)
            logp = logprior + loglike
            if np.isnan(logp):
                print(theta,)
                print("bad logp from like and prior:",)
                print(loglike,)
                print(logprior,)
                logp = -np.inf
            return logp
        else:
            return logprior

    def negative_logp(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                      Rabs_guess):
        return (0.0 - log_posterior(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr,
                                    Ierr, Rabs_guess))

    def negative_logp_fixedzf(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                              Rabs_guess, zf):
        theta[1] = zf
        # print theta
        return (0.0 - log_posterior(theta, zs, Gobs, Robs, Iobs, Gerr, Rerr,
                                    Ierr, Rabs_guess))

    if False:

        print( 'to here' )

        x0 = [Rabs_guess, 3.0]
        print(x0)
        xopt, fopt = fmin(negative_logp,
                          x0,
                          args=(zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                                Rabs_guess),
                          full_output=True)[:2]
        print("bestfit:",)
        print(xopt)
        Rabs, zf = xopt

    else:

        print( 'to here' )
        fnow = np.inf
        # for _zf in np.arange(0.0, 8.0, 0.1):
        for _zf in zfarr:
            x0 = [Rabs_guess, _zf]
            xopt, fopt = fmin(negative_logp_fixedzf,
                              x0,
                              args=(zs, Gobs, Robs, Iobs, Gerr, Rerr, Ierr,
                                    Rabs_guess, _zf),
                              full_output=True,
                              disp=False)[:2]
            # print(xopt)

            if fopt < fnow:
                fnow = fopt
                Rabs, zf = xopt
        print("bestfit:",)
        print(Rabs,)
        print(zf)

    model.set_normalization('sloan_r', zs, Rabs, apparent=False)
    Mass = model.get_masses(zf, zs) * model.get_normalization(zf, flux=True)
    # lgms = np.log10(Mass[0][0]) + 2.0 * np.log10(h)
    lgms = np.log10(Mass[0][0])
    print(lgms)

    if False:

        print( 'from here' )

        parray = np.zeros(8)
        parray[0] = zs
        parray[1] = Gobs
        parray[2] = Robs
        parray[3] = Iobs
        parray[4] = Gerr
        parray[5] = Rerr
        parray[6] = Ierr
        parray[7] = Rabs_guess
        ndim = 2  # number of parameters in the model
        nwalkers = 7  # number of MCMC walkers
        nburn = 100  # "burn-in" period to let chains stabilize
        nsteps = 200  # number of MCMC steps to take
        # set theta near the maximum likelihood, with
        np.random.seed(0)
        Rabs_init = np.random.uniform(Rabs_guess - 5. * Rerr,
                                      Rabs_guess + 5. * Rerr, nwalkers)
        zf_init = np.random.uniform(5, 4, nwalkers)
        starting_guesses = np.vstack((Rabs_init, zf_init)).T
        #
        pool = Pool(processes = 4)
        # pool = Pool(processes=1)
        # I usually set processes to 1 less than the max so my computer doesn't overload
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_posterior,
                                        args=parray,
                                        pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps)
        print("done")
        pool.close()
        #
        print(sampler.flatchain.shape)
        Rabs = sampler.flatchain[:, 0]
        zf = sampler.flatchain[:, 1]

    return lgms, zf, Rabs

if __name__ == "__main__":


    modelrich = ezgal.model('bc03_ssp_z_0.02_chab.model')
    modelpoor = ezgal.model('bc03_ssp_z_0.008_chab.model')
    model = ezgal.weight(97) * modelrich
    model += ezgal.weight(3) * modelpoor
    #
    model.set_cosmology( Om = Test_model.Om0, Ol = Test_model.Ode0, h = Test_model.h )
    model.add_filter('sloan_g', grid=True)
    model.add_filter('sloan_r', grid=True)
    model.add_filter('sloan_i', grid=True)

    ncpu = 2

    cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
    path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'
    out_path = '/home/xkchen/mywork/ICL/code/ezgal_files/'

    zs = 0.25
    z_min, z_max = 0.25, 8
    dt = 0.1
    zfarr = z_at_lb_time( z_min, z_max, dt, N_grid = 100)

    mm = 0

    tot_r, tot_sb, tot_err = [], [], []
    ## observed flux
    for kk in range( 3 ):

        with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[ mm ], band[kk]), 'r') as f:
            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        tot_r.append( tt_r )
        tot_sb.append( tt_sb )
        tot_err.append( tt_err )

    tot_r = np.array( tot_r )
    tot_sb = np.array( tot_sb )
    tot_err = np.array( tot_err )

    obs_mag = []
    obs_mag_err = []
    obs_R = []
    for kk in range( 3 ):

        idux = (tot_r[kk] >= 10) & (tot_r[kk] <= 1e3) ## use data points within 1Mpc only

        obs_R.append( tot_r[kk][idux] )

        tt_mag = 22.5 - 2.5 * np.log10( tot_sb[kk] )
        tt_mag_err = 2.5 * tot_err[kk] / ( np.log(10) * tot_sb[kk] )

        obs_mag.append( tt_mag[idux] )
        obs_mag_err.append( tt_mag_err[idux] )

    obs_mag = np.array( obs_mag )
    obs_mag_err = np.array( obs_mag_err )
    obs_R = np.array( obs_R )

    ## r band absolute mag
    Dl_ref = Test_model.luminosity_distance( zs ).value
    abs_Mag = obs_mag[1] - 5 * np.log10( Dl_ref * 10**6 / 10)

    NR = len( obs_R[0] )

    lgMs, zfs, mod_Mag = [], [], []

    for jj in range( NR ):
        _lgms, _zf, _abs_Mag = find_mstar2(model, zs, obs_mag[:,jj][0], obs_mag[:,jj][1], obs_mag[:,jj][2], 
                                obs_mag_err[:,jj][0], obs_mag_err[:,jj][1], obs_mag_err[:,jj][2], zfarr, abs_Mag[jj],)


        lgMs.append( _lgms )
        zfs.append( _zf )
        mod_Mag.append( _abs_Mag )

    

