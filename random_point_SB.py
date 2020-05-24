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

def origin_img_case():

    img_id = 2 ## 0 : center-stack, 2 : centered on catalog, 1 : rand-stack

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
    dnoise = 30

    for kk in range(rank, rank + 1):

        ### centered on img center
        if kk == 0:
            f_lel = np.arange(3.2, 4.2, 0.1) * 1e-3
        elif kk == 1:
            f_lel = np.arange(2.4, 3.2, 0.1) * 1e-3
        else:
            f_lel = np.arange(4.0, 5.6, 0.2) * 1e-3
        '''
        ### random / cat.(ra, dec) stacking case
        if kk == 0:
            f_lel = np.arange(1.0, 3.0, 0.2) * 1e-3
        elif kk == 1:
            f_lel = np.arange(1.0, 2.6, 0.2) * 1e-3
        else:
            f_lel = np.arange(1.4, 4.0, 0.2) * 1e-3
        '''

        if img_id == 0:
            with h5py.File(load + 'random_cat/stack/%s_band_center-stack_origin_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])
        if img_id == 2:
            with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_origin_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])

        Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro_0z(clust_img, pixel, 1, 3000, x0, y0, np.int(1.5 * bins), kk)
        Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

        plt.figure()
        bx0 = plt.subplot(111)
        if img_id == 0:
            bx0.set_title('%s band stacking img [centered on frame center]' % band[kk])
        if img_id == 1:
            bx0.set_title('%s band stacking img [random center]' % band[kk])
        if img_id == 2:
            bx0.set_title('%s band stacking img [centered on catalog]' % band[kk])

        clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'k', ls = '-', linewidth = 1.5, label = '1 Mpc', alpha = 0.5)
        clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'k', ls = '--', linewidth = 1.5, label = '2 Mpc', alpha = 0.5)
        #tf = bx0.imshow(clust_img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 2e-3, vmax = 5e-3, ) #norm = mpl.colors.LogNorm())
        tf = bx0.imshow(clust_img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 5e-5, vmax = 5e-3, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = bx0, fraction = 0.040, pad = 0.01, label = '$ flux[nmaggy / arcsec^2] $')

        kernl_img = ndimage.gaussian_filter(clust_img / pixel**2, sigma = dnoise,  mode = 'nearest')
        tf = bx0.contour(kernl_img, origin = 'lower', cmap = 'rainbow', levels = f_lel, linewidth = 0.5, )
        plt.clabel(tf, inline = False, fontsize = 6.5, colors = 'r', fmt = '%.4f')

        bx0.add_patch(clust00)
        bx0.add_patch(clust01)
        bx0.legend(loc = 1, frameon = False)
        bx0.axis('equal')
        ## centered on img center
        #bx0.set_xlim(x0 - np.int(2 * Rpp), x0 + np.int(2 * Rpp))
        #bx0.set_ylim(y0 - np.int(2 * Rpp), y0 + np.int(2 * Rpp))
        ## centered on cat.
        bx0.set_xlim(x0 - np.int(3 * Rpp), x0 + np.int(3 * Rpp))
        bx0.set_ylim(y0 - np.int(3 * Rpp), y0 + np.int(3 * Rpp))

        bx0.set_xticks([])
        bx0.set_yticks([])
        bx0.set_aspect('equal', 'box')
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_center-stack_2D_flux_origin.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_2D_flux_origin.png' % band[kk], dpi = 300)
        plt.close()

        ## SB pros.
        plt.figure()
        ax1 = plt.subplot(111)
        if img_id == 0:
            ax1.set_title('%s band SB [centered on frame center]' % band[kk] )
        if img_id == 1:
            ax1.set_title('%s band SB [random center]' % band[kk] )
        if img_id == 2:
            ax1.set_title('%s band SB [centered on catalog]' % band[kk] )

        ax1.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5)

        ax1.axvline(x = Angu_ref, color = 'b', linestyle = '--', label = '0.5 img width')
        ax1.axvline(x = 1.3 * Angu_ref, color = 'b', linestyle = '-', label = '0.5 img length') ## angle radius case
        '''
        ### center-stacking case
        if kk == 0:
            ax1.set_ylim(3.0e-3, 9.0e-3)
        elif kk == 1:
            ax1.set_ylim(2.0e-3, 7.0e-3)
        else:
            ax1.set_ylim(3.0e-3, 2.0e-2)
        '''
        ### random / cat.(ra, dec) stacking case
        if kk == 0:
            ax1.set_ylim(7e-5, 4e-3)
        elif kk == 1:
            ax1.set_ylim(5e-5, 3e-3)
        else:
            ax1.set_ylim(8e-5, 6e-3)

        ax1.set_yscale('log')
        ax1.set_ylabel('$SB[nanomaggies / arcsec^2]$')

        ax1.set_xlim(10, 1e3)
        ax1.set_xlabel('$ R[arcsec] $')
        ax1.set_xscale('log')

        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_center-stack_SB_origin.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_SB_origin.png' % band[kk], dpi = 300) 
        plt.close()

    return

def appli_resampling_img():

    img_id = 2 ## 0 : center-stack, 2 : centered on catalog, 1 : rand center (rand-stack)

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
    Ns = 30
    dnoise = 30

    for kk in range(rank, rank + 1):

        ### centered on img center
        if kk == 0:
            f_lel = np.arange(3.2, 4.2, 0.1) * 1e-3
        elif kk == 1:
            f_lel = np.arange(2.4, 3.2, 0.1) * 1e-3
        else:
            f_lel = np.arange(4.0, 5.6, 0.2) * 1e-3
        '''
        ### random / cat.(ra, dec) stacking case
        if kk == 0:
            f_lel = np.arange(1.0, 3.0, 0.2) * 1e-3
        elif kk == 1:
            f_lel = np.arange(1.0, 2.6, 0.2) * 1e-3
        else:
            f_lel = np.arange(1.4, 4.0, 0.2) * 1e-3
        '''
        '''
        ### sub-sample test
        for jj in range(Ns):
            #with h5py.File(load + 'random_cat/stack/sub_sample/%s_band_stack_%d_sub-imgs.h5' % (band[kk], jj), 'r') as f:  ## correlation
            with h5py.File(load + 'random_cat/stack/sub_sample/%s_band_center-stack_%d_sub-imgs.h5' % (band[kk], jj), 'r') as f:
                sub_mean = np.array(f['a'])
            Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro_0z(sub_mean, pixel, 1, 3 * Rpp, x0, y0, bins, kk)
            Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

            plt.figure()
            bx0 = plt.subplot(111)
            bx0.set_title('%s band %d sub-sample [centered on frame center]' % (band[kk], jj) )

            clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
            tf = bx0.imshow(sub_mean, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e-1, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = bx0, fraction = 0.050, pad = 0.01, label = 'flux[nmaggy]')
            bx0.add_patch(clust00)
            bx0.add_patch(clust01)
            bx0.axis('equal')
            bx0.set_xlim(x0 - np.ceil(1.3 * Rpp), x0 + np.ceil(1.3 * Rpp))
            bx0.set_ylim(y0 - np.ceil(Rpp), y0 + np.ceil(Rpp))
            bx0.set_xticks([])
            bx0.set_yticks([])
            plt.savefig(load + 'random_cat/stack/sub_sample/%s_band_%d_sub-center-stack_2D.png' % (band[kk], jj), dpi = 300)
            plt.close()

            plt.figure()
            ax1 = plt.subplot(111)
            ax1.set_title('%s band %d sub-sample SB [centered on frame center]' % (band[kk], jj) )

            ax1.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
                ecolor = 'r', elinewidth = 1, alpha = 0.5)
            ax1.set_ylim(1e-3, 1e-2)
            ax1.set_yscale('log')
            ax1.set_ylabel('$SB[nanomaggies / arcsec^2]$')
            ax1.set_xlim(1, 1e3)
            ax1.set_xlabel('$ R[arcsec] $')
            ax1.set_xscale('log')
            ax1.grid(which = 'both', axis = 'both')
            ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
            plt.savefig(load + 'random_cat/stack/sub_sample/%s_band_%d_sub-sample_SB.png' % (band[kk], jj), dpi = 300)
            plt.close()
        '''

        if img_id == 0:
            with h5py.File(load + 'random_cat/stack/%s_band_center-stack_cluster_imgs.h5' % band[kk], 'r') as f:
            #with h5py.File(load + 'random_cat/stack/sample_test/%s_band_center-stack_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])
        if img_id == 1:
            with h5py.File(load + 'random_cat/stack/%s_band_rand-stack_cluster_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])
        if img_id == 2:
            with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_imgs.h5' % band[kk], 'r') as f:
            #with h5py.File(load + 'random_cat/stack/sample_test/%s_band_stack_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])

        ## SB measurement
        Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro(clust_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

        plt.figure()
        bx0 = plt.subplot(111)
        if img_id == 0:
            bx0.set_title('%s band stacking img [centered on frame center]' % band[kk])
        if img_id == 1:
            bx0.set_title('%s band stacking img [random center]' % band[kk])
        if img_id == 2:
            bx0.set_title('%s band stacking img [centered on catalog]' % band[kk])

        clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'b', ls = '-', alpha = 0.5, label = '1 Mpc')
        clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'b', ls = '--', alpha = 0.5, label = '2 Mpc')
        tf = bx0.imshow(clust_img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 2e-3, vmax = 5e-3, ) #norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = bx0, fraction = 0.040, pad = 0.01, label = '$ flux[nmaggy / arcsec^2] $')

        kernl_img = ndimage.gaussian_filter(clust_img / pixel**2, sigma = dnoise,  mode = 'nearest')
        tf = bx0.contour(kernl_img, origin = 'lower', cmap = 'rainbow', levels = f_lel, )
        plt.clabel(tf, inline = False, fontsize = 6.5, colors = 'r', fmt = '%.4f')

        bx0.add_patch(clust00)
        bx0.add_patch(clust01)
        bx0.legend(loc = 1, frameon = False)
        bx0.axis('equal')

        ## centered on cat.
        #bx0.set_xlim(x0 - np.int(3 * Rpp), x0 + np.int(3 * Rpp))
        #bx0.set_ylim(y0 - np.int(3 * Rpp), y0 + np.int(3 * Rpp))

        ## centered on img center
        bx0.set_xlim(x0 - np.int(1.3 * Rpp), x0 + np.int(1.3 * Rpp))
        bx0.set_ylim(y0 - np.int(Rpp), y0 + np.int(Rpp))

        bx0.set_xticks([])
        bx0.set_yticks([])
        bx0.set_aspect('equal', 'box')
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_center-stack_2D_flux.png' % band[kk], dpi = 300)
            #plt.savefig(load + 'random_cat/stack/sample_test/%s_band_rand-pont_center-stack_2D_flux.png' % band[kk], dpi = 300)
        if img_id == 1:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_rand-stack_2D_flux.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_2D_flux.png' % band[kk], dpi = 300)
            #plt.savefig(load + 'random_cat/stack/sample_test/%s_band_rand-pont_stack_2D_flux.png' % band[kk], dpi = 300)
        plt.close()

        ## SB pros.
        plt.figure()
        ax1 = plt.subplot(111)
        if img_id == 0:
            ax1.set_title('%s band SB [centered on frame center]' % band[kk] )
        if img_id == 1:
            ax1.set_title('%s band SB [random center]' % band[kk] )
        if img_id == 2:
            ax1.set_title('%s band SB [centered on catalog]' % band[kk] )

        ax1.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5)

        ax1.axvline(x = 1e3, color = 'b', linestyle = '--', label = '0.5 img width')
        ax1.axvline(x = 1.3e3, color = 'b', linestyle = '-', label = '0.5 img length') ## physical radius case

        ### centered on img center
        if kk == 0:
            ax1.set_ylim(3e-3, 4.5e-3)
        elif kk == 1:
            ax1.set_ylim(2e-3, 3.5e-3)
        else:
            ax1.set_ylim(3e-3, 6e-3)
        '''
        ### random / cat.(ra, dec) stacking case
        if kk == 0:
            ax1.set_ylim(4e-4, 5e-3)
        elif kk == 1:
            ax1.set_ylim(4e-4, 4e-3)
        else:
            ax1.set_ylim(9e-4, 6e-3)
        ax1.set_yscale('log')
        '''
        ax1.set_ylabel('$SB[nanomaggies / arcsec^2]$')
        ax1.set_xlim(10, 3e3)
        ax1.set_xlabel('$ R[kpc] $')
        ax1.set_xscale('log')

        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_center-stack_SB.png' % band[kk], dpi = 300)
            #plt.savefig(load + 'random_cat/stack/sample_test/%s_band_rand-pont_center-stack_SB.png' % band[kk], dpi = 300)
        if img_id == 1:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_rand-stack_SB.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/%s_band_rand-pont_stack_SB.png' % band[kk], dpi = 300)
            #plt.savefig(load + 'random_cat/stack/sample_test/%s_band_rand-pont_stack_SB.png' % band[kk], dpi = 300) 
        plt.close()

        raise
        ##### SB profile comparison
        if img_id == 0:
            with h5py.File(load + 'random_cat/stack/%s_band_center-stack_cluster_imgs.h5' % band[kk], 'r') as f:
                clust_img_0 = np.array(f['a'])
            with h5py.File(load + 'random_cat/stack/sample_test/%s_band_center-stack_imgs.h5' % band[kk], 'r') as f:
                clust_img_1 = np.array(f['a'])
        if img_id == 2:
            with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_imgs.h5' % band[kk], 'r') as f:
                clust_img_0 = np.array(f['a'])
            with h5py.File(load + 'random_cat/stack/sample_test/%s_band_stack_imgs.h5' % band[kk], 'r') as f:
                clust_img_1 = np.array(f['a'])

        Rt_0, SBt_0, t_err00, t_err01, Intns_0, Intns_r_0, Intns_err_0 = SB_pro(clust_img_0, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

        Rt_1, SBt_1, t_err10, t_err11, Intns_1, Intns_r_1, Intns_err_1 = SB_pro(clust_img_1, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2

        plt.figure()
        gs = gridspec.GridSpec(2,1, height_ratios = [3, 2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        if img_id == 0:
            ax0.set_title('%s band SB [centered on image center]' % band[kk] )
        if img_id == 2:
            ax0.set_title('%s band SB [centered on catalog]' % band[kk] )

        ax0.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5, label = 'total imgs')
        ax0.errorbar(Intns_r_1, Intns_1, yerr = Intns_err_1, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'g', elinewidth = 1, alpha = 0.5, label = 'sub-sample imgs')
        ax0.axvline(x = 1e3, color = 'b', linestyle = '--', label = '0.5 img width')
        ax0.axvline(x = 1.3e3, color = 'b', linestyle = '-', label = '0.5 img length')
        '''
        ### centered on img center
        if kk == 0:
            ax0.set_ylim(3e-3, 5e-3)
        elif kk == 1:
            ax0.set_ylim(2e-3, 4e-3)
        else:
            ax0.set_ylim(3e-3, 6e-3)
        '''
        ### random / cat.(ra, dec) stacking case
        if kk == 0:
            ax0.set_ylim(4e-4, 5e-3)
        elif kk == 1:
            ax0.set_ylim(4e-4, 4e-3)
        else:
            ax0.set_ylim(9e-4, 6e-3)
        ax0.set_yscale('log')

        ax0.set_ylabel('$SB[nanomaggies / arcsec^2]$')
        ax0.set_xlim(10, 3e3)
        ax0.set_xlabel('$ R[kpc] $')
        ax0.set_xscale('log')
        ax0.grid(which = 'both', axis = 'both')
        #ax0.legend(loc = 'upper center', frameon = False)
        ax0.legend(loc = 'lower center', frameon = False)
        ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

        ax1.errorbar(Intns_r_0, Intns_0 - Intns_1, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5,)
        ax1.errorbar(Intns_r_1, Intns_1 - Intns_1, yerr = Intns_err_1, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'g', elinewidth = 1, alpha = 0.5,)
        ax1.axvline(x = 1e3, color = 'b', linestyle = '--', label = '0.5 img width')
        ax1.axvline(x = 1.3e3, color = 'b', linestyle = '-', label = '0.5 img length')

        ax1.set_xlim(ax0.get_xlim())
        if kk == 1:
            ax1.set_ylim(-0.5e-3, 0.5e-3)
        else: 
            ax1.set_ylim(-1e-3, 1e-3)

        ax1.set_xscale('log')
        ax1.set_xlabel('$ R[kpc] $')
        ax1.set_ylabel('$ SB - SB_{sub-sample} $')
        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        ax0.set_xticklabels([])

        plt.subplots_adjust(hspace = 0.05)
        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = None)
        #plt.savefig(load + 'random_cat/stack/%s_band_center-stack_SB_compare.png' % band[kk], dpi = 300)
        plt.savefig(load + 'random_cat/stack/%s_band_stack_SB_compare.png' % band[kk], dpi = 300)
        plt.close()

        return

def SB_pro_select_img():

    img_id = 0 ## 0 : center-stack, 2 : centered on catalog

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL

    dnoise = 30

    for kk in range(rank, rank + 1):

        ### centered on img center
        if kk == 0:
            f_lel = np.arange(2.6, 3.2, 0.1) * 1e-3
        elif kk == 1:
            f_lel = np.arange(2.1, 2.6, 0.1) * 1e-3
        else:
            f_lel = np.arange(2.5, 5.0, 0.2) * 1e-3
        '''
        if kk == 0:
            f_lel = np.logspace(np.log10(5e-4), np.log10(4e-3), 11)
        elif kk == 1:
            f_lel = np.logspace(np.log10(4e-4), np.log10(3e-3), 11)
        else:
            f_lel = np.logspace(np.log10(1e-3), np.log10(1e-2), 11)
        '''
        if img_id == 0:
            with h5py.File(load + 'random_cat/stack/sample_test/%s_band_center-stack_pixSB-select_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])
        if img_id == 2:
            with h5py.File(load + 'random_cat/stack/sample_test/%s_band_stack_pixSB-select_imgs.h5' % band[kk], 'r') as f:
                clust_img = np.array(f['a'])

        ## SB measurement
        Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro(clust_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
        Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

        plt.figure()
        bx0 = plt.subplot(111)
        if img_id == 0:
            bx0.set_title('%s band stacking img [centered on image center]' % band[kk])
        if img_id == 2:
            bx0.set_title('%s band stacking img [centered on catalog]' % band[kk])

        clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'b', ls = '-', alpha = 0.5, label = '1 Mpc')
        clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'b', ls = '--', alpha = 0.5, label = '2 Mpc')
        tf = bx0.imshow(clust_img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 2e-3, vmax = 5e-3, ) #norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = bx0, fraction = 0.040, pad = 0.01, label = '$ flux[nmaggy / arcsec^2] $')

        kernl_img = ndimage.gaussian_filter(clust_img / pixel**2, sigma = dnoise,  mode = 'nearest')
        tf = bx0.contour(kernl_img, origin = 'lower', cmap = 'rainbow', levels = f_lel, )
        plt.clabel(tf, inline = False, fontsize = 6.5, colors = 'r', fmt = '%.4f')

        bx0.add_patch(clust00)
        bx0.add_patch(clust01)
        bx0.legend(loc = 1, frameon = False)
        bx0.axis('equal')

        ## centered on cat.
        #bx0.set_xlim(x0 - np.int(3 * Rpp), x0 + np.int(3 * Rpp))
        #bx0.set_ylim(y0 - np.int(3 * Rpp), y0 + np.int(3 * Rpp))

        ## centered on img center
        bx0.set_xlim(x0 - np.int(1.3 * Rpp), x0 + np.int(1.3 * Rpp))
        bx0.set_ylim(y0 - np.int(Rpp), y0 + np.int(Rpp))

        bx0.set_xticks([])
        bx0.set_yticks([])
        bx0.set_aspect('equal', 'box')
        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/sample_test/%s_band_center-stack_pixSB-select_2D.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/sample_test/%s_band_stack_pixSB-select_2D.png' % band[kk], dpi = 300)
        plt.close()

        ## SB pros.
        plt.figure()
        ax1 = plt.subplot(111)
        if img_id == 0:
            ax1.set_title('%s band SB [centered on image center]' % band[kk] )
        if img_id == 2:
            ax1.set_title('%s band SB [centered on catalog]' % band[kk] )

        ax1.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
            ecolor = 'r', elinewidth = 1, alpha = 0.5)

        ax1.axvline(x = 1e3, color = 'b', linestyle = '--', label = '0.5 img width')
        ax1.axvline(x = 1.3e3, color = 'b', linestyle = '-', label = '0.5 img length') ## physical radius case
        '''
        ### centered on img center
        if kk == 0:
            ax1.set_ylim(2.0e-3, 3.5e-3)
        elif kk == 1:
            ax1.set_ylim(2.0e-3, 3.0e-3)
        else:
            ax1.set_ylim(1.0e-3, 5.0e-3)

        if kk == 0:
            ax1.set_ylim(5.0e-4, 4.0e-3)
        elif kk == 1:
            ax1.set_ylim(4.0e-4, 3.0e-3)
        else:
            ax1.set_ylim(1.0e-3, 1.0e-2)
        ax1.set_yscale('log')
        '''
        ax1.set_ylabel('$SB[nanomaggies / arcsec^2]$')
        ax1.set_xlim(10, 3e3)
        ax1.set_xlabel('$ R[kpc] $')
        ax1.set_xscale('log')

        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = None)
        if img_id == 0:
            plt.savefig(load + 'random_cat/stack/sample_test/%s_band_center-stack_pixSB-select_SB.png' % band[kk], dpi = 300)
        if img_id == 2:
            plt.savefig(load + 'random_cat/stack/sample_test/%s_band_stack_pixSB-select_SB.png' % band[kk], dpi = 300)
        plt.close()

    return

def main():
    #origin_img_case()
    #appli_resampling_img()
    SB_pro_select_img()

if __name__ == "__main__" :
    main()
