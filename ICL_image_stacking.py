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
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['u', 'g', 'r', 'i', 'z']
mag_add = np.array([-0.04, 0, 0, 0, 0.02])

stack_N = np.int(50)
def stack_light():
    un_mask = 0.15

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

        for jj in range(stack_N):
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

        SB, R, Ar, error = light_measure(mean_array, bins, 1, Rpp, x0, y0, pixel, z_ref)
        SB_measure = SB[1:] + mag_add[ii]
        R_measure = R[1:]
        Ar_measure = Ar[1:]
        SB_error = error[1:]

        # stack the total light
        tot_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        tot_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        p_count_total = np.zeros((len(Ny), len(Nx)), dtype = np.float)

        for jj in range(stack_N):
            ra_g = ra[jj]
            dec_g = dec[jj]
            z_g = z[jj]
            Da_g = Test_model.angular_diameter_distance(z_g).value
            data = fits.getdata(load + 'mask_data/B_plane/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
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

            tot_array[la0:la1, lb0:lb1] = tot_array[la0:la1, lb0:lb1] + resam
            tot_count[la0: la1, lb0: lb1] = resam
            ia = np.where(tot_count != 0)
            p_count_total[ia[0], ia[1]] = p_count_total[ia[0], ia[1]] + 1
            tot_count[la0: la1, lb0: lb1] = 0

        mean_total = tot_array/p_count_total
        where_are_nan = np.isnan(mean_total)
        mean_total[where_are_nan] = 0

        SB_tot, R_tot, Ar_tot, error_tot = light_measure(mean_total, bins, 1, Rpp, x0, y0, pixel, z_ref)
        SB_TT = SB_tot[1:] + mag_add[ii]
        R_TT = R_tot[1:]
        Ar_TT = Ar_tot[1:]
        err_TT = error_tot[1:]

        SB_ICL = SB_measure/(1 - un_mask) - SB_TT * un_mask/(1 - un_mask)

        #staack sky
        sky_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        sky_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        p_sky_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
        for kk in range(stack_N):
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

        # fig part    
        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(Ar_measure, SB_ICL, 'b-', label = '$Stack_{%.0f}$' % stack_N)
        ax1.axhline(sky_mag, ls = '-.', c = 'r', label = '$sky$')
        ax1.set_xscale('log')
        ax1.set_xlabel('$R[arcsec]$')
        ax1.set_ylabel('$M_r[mag/arcsec^2]$')
        ax1.legend(loc = 1)
        ax1.set_title('stacking test in %s band'%band[ii])
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        ax2 = ax1.twiny()
        ax2.plot(R_measure, SB_ICL, 'b-')
        ax2.set_xscale('log')
        ax2.set_xlabel('$R[kpc]$')
        ax2.tick_params(axis = 'x', which = 'both', direction = 'in')
        ax1.invert_yaxis()
        plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stacking_test_%s.png'%band[ii], dpi = 600)
        plt.close()

        # stack image
        plt.figure()
        gf = plt.imshow(mean_array, cmap  ='Greys', origin = 'lower', vmin = 1e-3,  norm = mpl.colors.LogNorm())
        plt.colorbar(gf, fraction = 0.036, pad = 0.01, label = '$f[nmagy]$')
        hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b', ls = '-', lw = 0.5)
        hsc.circles(x0, y0, s = 1.1*Rpp,  fc = '', ec = 'b', ls = '--', lw = 0.5)
        plt.title('stack %.0f mean image in %s band' % (stack_N, band[ii]))
        plt.xlim(x0 - 1.2*Rpp, x0 + 1.2*Rpp)
        plt.ylim(y0 - 1.2*Rpp, y0 + 1.2*Rpp)
        plt.subplots_adjust(left = 0.01, right = 0.85)
        plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack %.0f mean image in %s band.png' % (stack_N, band[ii]), dpi = 600)
        plt.close()

        print(ii)
    return

def main():
    stack_light()

if __name__ == "__main__":
    main()
