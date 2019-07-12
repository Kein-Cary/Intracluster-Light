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

#band = ['u', 'g', 'r', 'i', 'z']
#mag_add = np.array([-0.04, 0, 0, 0, 0.02])
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

#read Redmapper catalog
goal_data = fits.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)

z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]

red_z = z_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_ra = ra_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_dec = dec_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_rich = rich_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]

def stack_light(band_number, stack_number, subz, subra, subdec):
    stack_N = np.int(stack_number)
    ii = np.int(band_number)
    sub_z = subz
    sub_ra = subra
    sub_dec = subdec

    x0 = 2427
    y0 = 1765
    bins = 45
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    sum_grid = np.array(np.meshgrid(Nx, Ny))
    # stack cluster
    get_array = np.zeros((len(Ny), len(Nx)), dtype = np.float) # sum the flux value for each time
    count_array = np.zeros((len(Ny), len(Nx)), dtype = np.float) # sum array but use for pixel count for each time
    p_count_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # how many times of each pixel get value

    for jj in range(stack_N):
        '''
        ra_g = ra[jj]
        dec_g = dec[jj]
        z_g = z[jj]
        '''
        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]
        
        Da_g = Test_model.angular_diameter_distance(z_g).value
        data = fits.getdata(load + 'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
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

        # read the mask_metrix
        mask_A = fits.getdata(load + 'mask_metrx/mask_A/1.5sigma/A_mask_metrx_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
        maskA = mask_A[0]
        xm, ym, res_mask = gen(maskA, 1, b, cx, cy)
        if b > 1:
            res_mask = res_mask[1:, 1:]
        elif b == 1:
            res_mask = res_mask[1:-1, 1:-1]
        else:
            res_mask = res_mask

        i_cut = res_mask <= b**2/2
        res_mask[i_cut] = 0
        i_sav = res_mask != 0
        res_mask[i_sav] = 1

        resam = resam * res_mask

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
    where_are_inf = np.isinf(mean_array)
    mean_array[where_are_inf] = 0

    SB, R, Ar, error = light_measure(mean_array, bins, 1, Rpp, x0, y0, pixel, z_ref)

    SB_diff = SB[1:] + mag_add[ii]
    R_diff = R[1:]
    Ar_diff = Ar[1:]
    err_diff = error[1:]
    # background level
    dr = np.sqrt((sum_grid[0,:] - x0)**2 + (sum_grid[1,:] - y0)**2)
    ia = dr >= Rpp
    ib = dr <= 1.1*Rpp
    ic = ia & ib
    sky_set = mean_array[ic]
    sky_light = np.sum(sky_set[sky_set != 0])/len(sky_set[sky_set != 0])
    sky_mag = 22.5 - 2.5*np.log10(sky_light) + 2.5*np.log10(pixel**2) + mag_add[ii]

    Back_subs_mean = mean_array * 1
    Back_subs_mean[mean_array != 0] = Back_subs_mean[mean_array != 0 ] - sky_light
    SBt, Rt, Art, errort = light_measure(Back_subs_mean, bins, 1, Rpp, x0, y0, pixel, z_ref)
    SBt = SBt[1:] + mag_add[ii]
    Rt = Rt[1:]
    Art = Art[1:]
    errt = errort[1:]

    # fig part    
    plt.figure()
    ax1 = plt.subplot(111)
    ax1.errorbar(Ar_diff, SB_diff, yerr = err_diff, xerr = None, ls = '', fmt = 'ro', label = '$Stack_{%.0f}$' % stack_N)
    ax1.axhline(sky_mag, ls = '-.', c = 'r', label = '$Background$')
    ax1.set_xscale('log')
    ax1.set_xlabel('$R[arcsec]$')
    ax1.set_ylabel('$SB[mag/arcsec^2]$')
    ax1.legend(loc = 1, fontsize = 12)
    ax1.set_title('stacking test in %s band'%band[ii])
    ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
    ax2 = ax1.twiny()
    ax2.plot(R_diff, SB_diff, 'b-')
    ax2.set_xscale('log')
    ax2.set_xlabel('$R[kpc]$')
    ax2.tick_params(axis = 'x', which = 'both', direction = 'in')
    ax1.invert_yaxis()
    plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%.0f_1.5sigma_%s_band.png' % (stack_N, band[ii]), dpi = 600)
    plt.close()

    return SBt, Rt, Art, errt, SB_diff, R_diff, Ar_diff, err_diff, sky_mag

def main():
    import matplotlib.gridspec as gridspec
    ix = (red_rich >= 25) & (red_rich <= 29)
    RichN = red_rich[ix]
    zN = red_z[ix]
    raN = red_ra[ix]
    decN = red_dec[ix]
    stackn = len(zN)

    for ii in range(len(band)):
        id_band = ii

        fig = plt.figure(figsize = (16,9))
        bx = plt.subplot(111)
        bx.set_title('$stack \; light \; test \; in \; %s \; band$' % band[ii])

        SB, R, Ar, err, iner_SB, iner_R, iner_Ar, iner_err, BG_light = stack_light(id_band, stackn, zN, raN, decN)

        bx.errorbar(Ar, SB, yerr = err, xerr = None, color = 'b', marker = 'o', linewidth = 1, markersize = 10, 
            ecolor = 'b', elinewidth = 1, alpha = 0.5, label = '$SB_{stack%.0f}$' % stackn)
        bx.set_xscale('log')
        bx.set_xlabel('$R[arcsec]$')
        bx.set_ylabel('$SB[mag/arcsec^2]$')
        bx.tick_params(axis = 'both', which = 'both', direction = 'in')
        handles, labels = plt.gca().get_legend_handles_labels()

        bx1 = bx.twiny()
        bx1.plot(R, SB, ls = '-', color = 'b', alpha = 0.5)
        bx1.set_xscale('log')
        bx1.set_xlabel('$R[kpc]$')
        bx1.tick_params(axis = 'x', which = 'both', direction = 'in')

        subax = fig.add_axes([0.18, 0.15, 0.35, 0.25])
        subax.errorbar(iner_Ar, iner_SB, yerr = iner_err, xerr = None, ls = '', fmt = 'ro', label = '$Stack_{%.0f}$' % stackn)
        subax.axhline(BG_light, ls = '-.', c = 'r', label = '$Background$')
        subax.set_xscale('log')
        subax.set_xlabel('$R[arcsec]$')
        subax.set_ylabel('$SB[mag/arcsec^2]$')
        subax.legend(loc = 1, fontsize = 10)
        subax.set_title('stacking test in %s band'%band[ii])
        subax.tick_params(axis = 'both', which = 'both', direction = 'in')
        subax.invert_yaxis()

        bx.invert_yaxis()
        bx.legend( loc = 1, fontsize = 15)
        plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/light_test_%s_band.png' % band[ii], dpi = 600)
        plt.close()

if __name__ == "__main__":
    main()
