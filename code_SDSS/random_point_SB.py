import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C

import h5py
import time
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

from scipy import ndimage
from astropy import cosmology as apcy
from light_measure import light_measure, light_measure_Z0

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref)*rad2asec
Rpp = Angu_ref / pixel

home = '/media/xkchen/My Passport/data/SDSS/'
load = '/media/xkchen/My Passport/data/SDSS/'

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def cov_MX(radius, pros):
    flux_array = np.array(pros)
    r_array = np.array(radius)
    Nt = len(flux_array)
    SB_value = []
    R_value = []
    for ll in range(Nt):
        id_nan = np.isnan(flux_array[ll])
        setx = flux_array[ll][id_nan == False]
        setr = r_array[ll][id_nan == False]
        SB_value.append(setx)
        R_value.append(setr)
    SB_value = np.array(SB_value)
    R_value = np.array(R_value)
    R_mean_img = np.nanmean(R_value, axis = 0)

    mean_lit = np.nanmean(SB_value, axis = 0)
    std_lit = np.nanstd(SB_value, axis = 0)
    nx, ny = SB_value.shape[1], SB_value.shape[0]

    cov_tt = np.zeros((nx, nx), dtype = np.float)
    cor_tt = np.zeros((nx, nx), dtype = np.float)

    for qq in range(nx):
        for tt in range(nx):
            cov_tt[qq, tt] = np.sum( (SB_value[:,qq] - mean_lit[qq]) * (SB_value[:,tt] - mean_lit[tt]) ) / ny

    for qq in range(nx):
        for tt in range(nx):
            cor_tt[qq, tt] = cov_tt[qq, tt] / (std_lit[qq] * std_lit[tt])
    cov_MX_img = cov_tt * 1.
    cor_MX_img = cor_tt * 1.
    return R_mean_img, cov_MX_img, cor_MX_img

def SB_pro(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg, band_id):
    kk = band_id
    Intns, Intns_r, Intns_err = light_measure(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg)
    SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    flux0 = Intns + Intns_err
    flux1 = Intns - Intns_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    err0 = SB - dSB0
    err1 = dSB1 - SB
    id_nan = np.isnan(SB)
    SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
    dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
    idx_nan = np.isnan(dSB1)
    out_err1[idx_nan] = 100.

    return R_out, SB_out, out_err0, out_err1, Intns, Intns_r, Intns_err

def SB_pro_0z(img, pix_size, r_lim, R_pix, cx, cy, R_bins, band_id):
    kk = band_id
    Intns, Angl_r, Intns_err = light_measure_Z0(img, pix_size, r_lim, R_pix, cx, cy, R_bins)
    SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    flux0 = Intns + Intns_err
    flux1 = Intns - Intns_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
    err0 = SB - dSB0
    err1 = dSB1 - SB
    id_nan = np.isnan(SB)

    SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Angl_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
    dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
    idx_nan = np.isnan(dSB1)
    out_err1[idx_nan] = 100.

    return R_out, SB_out, out_err0, out_err1, Intns, Angl_r, Intns_err

def mask_adjust():

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
    xn, yn = 1024, 744 ## center-stack

    for kk in range(1):

        ## mask once
        with h5py.File(load + 're_mask/r_band_center-stack_random-img_top1000.h5', 'r') as f:
            random_0 = np.array(f['a'])

        rand_I_0, rand_I_r_0, rand_I_err_0 = SB_pro_0z(random_0, pixel, 1, 3000, x0, y0, np.int(1.5 * bins), kk)[4:]
        rand_I_0, rand_I_err_0 = rand_I_0 / pixel**2, rand_I_err_0 / pixel**2

        ## mask twice
        with h5py.File(load + 'random_cat/angle_stack/r_band_stack_mask-only_adjust-param.h5', 'r') as f:
            random_1 = np.array(f['a'])

        rand_I_1, rand_I_r_1, rand_I_err_1 = SB_pro_0z(random_1, pixel, 1, 3000, x0, y0, np.int(1.5 * bins), kk)[4:]
        rand_I_1, rand_I_err_1 = rand_I_1 / pixel**2, rand_I_err_1 / pixel**2

        plt.figure()
        gs = gridspec.GridSpec(2,1, height_ratios = [4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        ax0.set_title('centered on image center SB [random]')
        ax0.errorbar(rand_I_r_0, rand_I_0, yerr = rand_I_err_0, xerr = None, color = 'r', marker = 'None', 
            ls = '-', linewidth = 1.5, ecolor = 'r', elinewidth = 1.5, alpha = 0.5, label = 'single masking')

        ax0.errorbar(rand_I_r_1, rand_I_1, yerr = rand_I_err_1, xerr = None, color = 'b', marker = 'None', 
            ls = '-', linewidth = 1.5, ecolor = 'b', elinewidth = 1.5, alpha = 0.5, label = 'twice masking')

        ax0.axvline(x = Angu_ref, color = 'k', linestyle = '-', linewidth = 1,)
        ax0.axvline(x = 2 * Angu_ref, color = 'k', linestyle = '--', linewidth = 1,)

        ax0.set_ylim(2e-3, 8e-3)
        ax0.set_xlim(10, 1e3)
        ax0.set_xscale('log')
        ax0.set_ylabel('$SB[nanomaggies / arcsec^2]$')
        ax0.set_xlabel('$ R[arcsec] $')
        ax0.legend(loc = 'upper center', frameon = False)
        ax0.grid(which = 'both', axis = 'both', alpha = 0.35)
        ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

        ax1.plot(rand_I_r_0, rand_I_0 - rand_I_1, color = 'r', alpha = 0.5)
        ax1.fill_between(rand_I_r_0, y1 = rand_I_0 - rand_I_1 - rand_I_err_0, y2 = rand_I_0 - rand_I_1 + rand_I_err_0, color = 'r', alpha = 0.35)

        ax1.plot(rand_I_r_1, rand_I_1, color = 'b', alpha = 0.5)
        ax1.fill_between(rand_I_r_1, y1 = rand_I_1 - rand_I_1 - rand_I_err_1, y2 = rand_I_1 - rand_I_1 + rand_I_err_1, color = 'b', alpha = 0.35)

        ax1.axvline(x = Angu_ref, color = 'k', linestyle = '-', linewidth = 1,)
        ax1.axvline(x = 2 * Angu_ref, color = 'k', linestyle = '--', linewidth = 1,)
        ax1.set_ylim(-1e-3, 1e-3)
        ax1.set_xlim(ax0.get_xlim())
        ax1.set_xscale('log')
        ax1.set_ylabel('$ SB - SB_{twice \; masking} $')
        ax1.set_xlabel('$ R[arcsec] $')
        ax1.grid(which = 'both', axis = 'both', alpha = 0.35)
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        ax0.set_xticks([])

        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = 0.01)
        plt.savefig('img_trend.png', dpi = 300)
        plt.close()

        raise

    return

def main():
    mask_adjust()

if __name__ == "__main__" :
    main()
