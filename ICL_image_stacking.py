import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

from resamp import gen
from extinction_redden import A_wave
from light_measure import light_measure, flux_recal
##
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

# sample catalog
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
R0 = 1 # in unit Mpc
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['u', 'g', 'r', 'i', 'z']
mag_add = np.array([-0.04, 0, 0, 0, 0.02])
#zp = np.array([]) # zero point magnitude for differ band
def stack_light():
    x0 = 2427
    y0 = 1765
    bins = 90
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    sum_grid = np.array(np.meshgrid(Nx, Ny))
    # stack cluster
    for ii in range(len(band)):
        get_array = np.zeros((len(Ny), len(Nx)), dtype = np.float) # sum the flux value for each time
        count_array = np.zeros((len(Ny), len(Nx)), dtype = np.float) # sum array but use for pixel count for each time
        p_count_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # how many times of each pixel get value

        for jj in range(100):
            ra_g = ra[jj]
            dec_g = dec[jj]
            z_g = z[jj]
            Da_g = Test_model.angular_diameter_distance(z_g).value
            data = fits.getdata(load + 'mask_data/A_plane/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
            img = data[0]
            wcs = awc.WCS(data[1])
            cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

            Angur = (R0*rad2asec/Da_g)
            Rp = Angur/pixel
            L_ref = Da_ref*pixel/rad2asec
            L_z0 = Da_g*pixel/rad2asec
            b = L_ref/L_z0
            Rref = (R0*rad2asec/Da_ref)/pixel

            f_goal = flux_recal(img, z_g, z_ref)
            xn, yn, resam = gen(f_goal, 1, b, cx, cy)
            xn = np.int(xn)
            yn = np.int(yn)
            if b > 1:
                resam = resam[1:, 1:]
            elif b == 1:
                resam = resam[1:-1, 1:-1]
            else:
                resam = resam
            la0 = np.int(y0 - yn)
            la1 = np.int(y0 - yn + resam.shape[0])
            lb0 = np.int(x0 - xn)
            lb1 = np.int(x0 - xn + resam.shape[1])

            get_array[la0:la1, lb0:lb1] = get_array[la0:la1, lb0:lb1] + resam
            count_array[la0: la1, lb0: lb1] = resam
            ia = np.where(count_array != 0)
            p_count_1[ia[0], ia[1]] = p_count_1[ia[0], ia[1]] + 1
            count_array[la0: la1, lb0: lb1] = 0

        mean_array = get_array/p_count_1
        where_are_nan = np.isnan(mean_array)
        mean_array[where_are_nan] = 0

        Angu_ref = (R0/Da_ref)*rad2asec
        Rpp = Angu_ref/pixel

        SB, R, Ar, error = light_measure(mean_array, bins, 1, Rpp, x0, y0, pixel, z_ref)
        SB_measure = SB[1:] + mag_add[ii]
        R_measure = R[1:]
        Ar_measure = Ar[1:]
        SB_error = error[1:]
        
        #staack sky
        sky_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        sky_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        p_sky_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        for kk in range(100):
            ra_g = ra[kk]
            dec_g = dec[kk]
            z_g = z[kk]
            Da_g = Test_model.angular_diameter_distance(z_g).value
            data = fits.getdata(load + 'mask_data/sky_plane/sky_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
            img = data[0]
            wcs = awc.WCS(data[1])
            cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

            Angur = (R0*rad2asec/Da_g)
            Rp = Angur/pixel
            L_ref = Da_ref*pixel/rad2asec
            L_z0 = Da_g*pixel/rad2asec
            b = L_ref/L_z0
            Rref = (R0*rad2asec/Da_ref)/pixel

            f_goal = flux_recal(img, z_g, z_ref)
            xn, yn, resam = gen(f_goal, 1, b, cx, cy)
            xn = np.int(xn)
            yn = np.int(yn)
            if b > 1:
                resam = resam[1:, 1:]
            elif b == 1:
                resam = resam[1:-1, 1:-1]
            else:
                resam = resam
            la0 = np.int(y0 - yn)
            la1 = np.int(y0 - yn + resam.shape[0])
            lb0 = np.int(x0 - xn)
            lb1 = np.int(x0 - xn + resam.shape[1])

            sky_array[la0: la1, lb0: lb1] = sky_array[la0: la1, lb0: lb1] + resam
            sky_count[la0: la1, lb0: lb1] = resam
            ia = np.where(sky_count != 0)
            p_sky_count[ia[0], ia[1]] = p_sky_count[ia[0], ia[1]] + 1
            sky_count[la0: la1, lb0: lb1] = 0

        mean_sky = sky_array/p_sky_count
        where_are_nan = np.isnan(mean_sky)
        mean_sky[where_are_nan] = 0
        dr = np.sqrt((sum_grid[0,:] - x0)**2 + (sum_grid[1,:] - y0)**2)
        ia = dr >= Rpp
        ib = dr <= 1.1*Rpp
        ic = ia & ib
        sky_set = mean_sky[ic]
        sky_light = np.sum(sky_set[sky_set != 0])/len(sky_set[sky_set != 0])
        sky_mag = 22.5 - 2.5*np.log10(sky_light) + 2.5*np.log10(pixel**2) + mag_add[ii]
        '''
        test_sky = np.zeros(10, dtype = np.float)
        for kk in range(10):

            tx = np.linspace(0, 2047, 2048)
            ty = np.linspace(0, 1488, 1489)
            t_grid = np.array(np.meshgrid(tx, ty))

            ra_g = ra[kk]
            dec_g = dec[kk]
            z_g = z[kk]
            Da_g = Test_model.angular_diameter_distance(z_g).value
            data = fits.getdata(load + 'mask_data/sky_plane/sky_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
            img = data[0]
            wcs = awc.WCS(data[1])
            cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
            Angur = (R0*rad2asec/Da_g)
            Rp = Angur/pixel

            dtr = np.sqrt((cx - t_grid[0,:])**2 + (cy - t_grid[1,:])**2)
            ix = dtr <= 1.1*Rp
            iy = dtr >= Rp*1
            iz = ix & iy
            iu = np.where(iz == True)[0]
            tsky = img[iu[0], iu[1]]
            ttsky = tsky[tsky != 0]
            SBT = 22.5 - 2.5*np.log10(np.mean(ttsky)) + 2.5*np.log10(pixel**2)
            test_sky[kk] = SBT + 10*np.log10((1+z_ref)/(1+z_g))
        sky_mag = np.mean(test_sky) + mag_add[ii]
        '''
        # fig part    
        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(Ar_measure, SB_measure, 'b-', label = '$Stack_{100}$')
        ax1.axhline(sky_mag, ls = '-.', c = 'r', label = '$sky$')
        ax1.set_xscale('log')
        ax1.set_xlabel('$R[arcsec]$')
        ax1.set_ylabel('$M_r[mag/arcsec^2]$')
        ax1.legend(loc = 1)
        ax1.set_title('stacking test in %s band'%band[ii])
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        ax2 = ax1.twiny()
        ax2.plot(R_measure, SB_measure, 'b-')
        ax2.set_xscale('log')
        ax2.set_xlabel('$R[kpc]$')
        ax2.tick_params(axis = 'x', which = 'both', direction = 'in')
        ax1.invert_yaxis()
        plt.savefig('stacking_test_%s.png'%band[ii], dpi = 600)
        plt.close()
       
    return SB_measure, R_measure, Ar_measure, SB_error

def main():
    stack_light()

if __name__ == "__main__":
    main()
