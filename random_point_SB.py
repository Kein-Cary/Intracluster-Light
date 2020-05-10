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
from light_measure import light_measure, light_measure_rn
from Mass_rich_radius import rich2R_critical_2019

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def betwn_SB(data, R_low, R_up, cx, cy, pix_size, z0, band_id):

    betwn_r, betwn_Intns, betwn_err = light_measure_rn(data, R_low, R_up, cx, cy, pix_size, z0)
    betwn_lit = 22.5 - 2.5 * np.log10(betwn_Intns) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
    flux0 = betwn_Intns + betwn_err
    flux1 = betwn_Intns - betwn_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
    btn_err0 = betwn_lit - dSB0
    btn_err1 = dSB1 - betwn_lit
    id_nan = np.isnan(dSB1)
    if id_nan == True:
        btn_err1 = 100.

    return betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err

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

def main():

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
    dnoise = 30
    SB_lel = np.arange(27, 31, 1) ## cotour c-label

    for kk in range(rank, rank + 1):
        ## cluster img
        with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_imgs.h5' % band[kk], 'r') as f:
            clust_img = np.array(f['a'])
        ## sky img
        with h5py.File(load + 'random_cat/stack/sky-median-sub_%s_band_img.h5' % band[kk], 'r') as f:
            BCG_sky = np.array(f['a'])
        with h5py.File(load + 'random_cat/stack/M_rndm_sky-median-sub_stack_%s_band.h5' % band[kk], 'r') as f:
            rand_sky = np.array(f['a'])

        differ_img = BCG_sky - rand_sky
        add_img = clust_img + differ_img

        ## SB measurement
        Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro(clust_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

        R_sky, sky_ICL, sky_err0, sky_err1, Intns, Intns_r, Intns_err = SB_pro(differ_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns, Intns_err = Intns / pixel**2, Intns_err / pixel**2

        R_add, SB_add, add_err0, add_err1, Intns_1, Intns_r_1, Intns_err_1 = SB_pro(add_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2

        ## stack imgs
        plt.figure(figsize = (18, 6))
        bx0 = plt.subplot(131)
        bx1 = plt.subplot(132)
        bx2 = plt.subplot(133)

        bx0.set_title('%s band random point stack img' % band[kk])
        clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
        clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
        tf = bx0.imshow(clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e0, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = bx0, fraction = 0.050, pad = 0.01, label = 'flux[nmaggy]')
        ## add contour
        con_img = clust_img * 1.
        kernl_img = ndimage.gaussian_filter(clust_img, sigma = dnoise,  mode = 'nearest')
        SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
        tg = bx0.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
        plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')
        bx0.add_patch(clust00)
        bx0.add_patch(clust01)
        bx0.axis('equal')
        bx0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
        bx0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
        bx0.set_xticks([])
        bx0.set_yticks([])

        bx1.set_title('random point sky difference img')
        clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-',)
        clust11 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'r', ls = '--',)
        tf = bx1.imshow(differ_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4)
        plt.colorbar(tf, ax = bx1, fraction = 0.050, pad = 0.01, label = 'flux[nmaggy]')
        bx1.add_patch(clust10)
        bx1.add_patch(clust11)
        bx1.axis('equal')
        bx1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
        bx1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
        bx1.set_xticks([])
        bx1.set_yticks([])

        bx2.set_title('difference + random point stack img')
        clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
        clust21 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
        tf = bx2.imshow(differ_img + clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e0, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = bx2, fraction = 0.050, pad = 0.01, label = 'flux[nmaggy]')
        ## add contour
        con_img = differ_img + clust_img
        kernl_img = ndimage.gaussian_filter(con_img, sigma = dnoise,  mode = 'nearest')
        SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
        tg = bx2.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
        plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')
        bx2.add_patch(clust20)
        bx2.add_patch(clust21)
        bx2.axis('equal')
        bx2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
        bx2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
        bx2.set_xticks([])
        bx2.set_yticks([])

        plt.tight_layout()
        plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_view.png' % band[kk], dpi = 300) 
        plt.close()

        ## SB pros.
        plt.figure(figsize = (12, 6))
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        ax0.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, label = 'stacking random point imgs', alpha = 0.5)
        ax0.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'g', elinewidth = 1, label = 'residual sky img + random point img', alpha = 0.5)
        #ax0.errorbar(R_sky, sky_ICL, yerr = [sky_err0, sky_err1], xerr = None, color = 'm', marker = 'None', ls = '--', linewidth = 1, 
        #    ecolor = 'm', elinewidth = 1, label = 'stacking residual sky imgs', alpha = 0.5)
        #ax0.plot(R_sky, sky_ICL, color = 'm', ls = '-', alpha = 0.5, label = 'stacking residual sky imgs',)

        ax0.set_xlabel('$R[kpc]$')
        ax0.set_ylabel('$SB[mag / arcsec^2]$')
        ax0.set_xscale('log')
        ax0.set_ylim(28, 30)
        ax0.set_xlim(1, 2e3)
        ax0.legend(loc = 1, frameon = False)
        ax0.invert_yaxis()
        ax0.grid(which = 'both', axis = 'both')
        ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

        ax1.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5)
        ax1.errorbar(Intns_r_1, Intns_1, yerr = Intns_err_1, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'g', elinewidth = 1, alpha = 0.5)
        #ax1.errorbar(Intns_r, Intns, yerr = Intns_err, xerr = None, color = 'm', marker = 'None', ls = '--', linewidth = 1, 
        #    ecolor = 'm', elinewidth = 1, alpha = 0.5)
        #ax1.plot(Intns_r, Intns, color = 'm', ls = '-', alpha = 0.5, )

        ax1.set_xlabel('$R[kpc]$')
        ax1.set_ylabel('$SB[nanomaggies / arcsec^2]$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim(1e-3, 1e-2)
        ax1.set_xlim(1, 2e3)
        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')        

        plt.tight_layout()
        plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_SB.png' % band[kk], dpi = 300) 
        plt.close()

    commd.Barrier()

if __name__ == "__main__" :
    main()
