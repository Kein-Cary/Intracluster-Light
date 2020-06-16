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
from scipy import interpolate as interp
from astropy import cosmology as apcy
from light_measure import light_measure, light_measure_rn, light_measure_Z0
from Mass_rich_radius import rich2R_Simet

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

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
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

def sers_pro(r, mu_e, r_e, n):
    belta_n = 2 * n - 0.324
    fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
    mu_r = mu_e + fn
    return mu_r

def main():
    out_load = load + 'rich_sample/add_random_sample/'

    rich_a0, rich_a1, rich_a2 = 20, 30, 50
    ## sersic pro of Zibetti 05
    mu_e = np.array([23.87, 25.22, 23.4])
    r_e = np.array([19.29, 19.40, 20])

    x0, y0 = 2427, 1765 ## stacking img center pix
    bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
    bin_1Mpc = 75
    ## star point to measure background
    bl_set = 0.75

    dnoise = 30
    SB_lel = np.arange(27, 31, 1) ## cotour c-label

    N_sum = np.array([3262, 3263, 3252]) ## total sky-select imgs
    N_bin = np.array([ [1855, 1066, 341], [1856, 1069, 338], [1851, 1065, 336] ])

    for kk in range(rank, rank + 1):

        SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
        R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
        R_obs = R_obs**4
        ## sersic part
        Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
        SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

        ## R200 calculate
        with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
            set_array = np.array(f['a'])
        set_z, set_rich = set_array[2,:], set_array[3,:]

        ## record the SB profile of each rich bin
        sub_SB = []
        sub_R = []
        sub_err0 = []
        sub_err1 = []
        R200 = np.zeros(3, dtype = np.float)

        plt.figure()
        gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax0.set_title('$ %s \; band \; binned \; with \; \\lambda $' % band[kk])

        for lamda_k in range(3):

            if lamda_k == 0:
                idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
            elif lamda_k == 1:
                idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
            else:
                idx = (set_rich >= rich_a2)

            lis_z = set_z[idx]
            lis_rich = set_rich[idx]

            M_vir, R_vir = rich2R_Simet(lis_z, lis_rich)
            R200[lamda_k] = np.nanmedian(R_vir)

            bins_0 = np.int( np.ceil(bin_1Mpc * bl_set * R200[lamda_k] / 1e3) )
            R_min_0, R_max_0 = 1, bl_set * R200[lamda_k] # kpc
            R_min_1, R_max_1 = bl_set * R200[lamda_k] + 100., R_max # kpc

            if R_min_1 < R_max:
                x_quen = np.logspace(0, np.log10(R_max_0), bins_0)
                d_step = np.log10(x_quen[-1]) - np.log10(x_quen[-2])
                bins_1 = len( np.arange(np.log10(R_min_1), np.log10(R_max), d_step) )
            else:
                bins_1 = 0
            r_a0, r_a1 = R_max_0, R_min_1

            with h5py.File(load + 'rich_sample/stack_A_%d_in_%s_band_%drich.h5' % (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                clust_img = np.array(f['a'])

            Rt_0, SBt_0, t_err0_0, t_err1_0, Intns_0_0, Intns_r_0_0, Intns_err_0_0 = SB_pro(
                clust_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
            Rt_1, SBt_1, t_err0_1, t_err1_1, Intns_0_1, Intns_r_0_1, Intns_err_0_1 = SB_pro(
                clust_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
            betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(clust_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

            Rt = np.r_[Rt_0, betwn_r, Rt_1]
            SBt = np.r_[SBt_0, betwn_lit, SBt_1]
            t_err0 = np.r_[t_err0_0, btn_err0, t_err0_1]
            t_err1 = np.r_[t_err1_0, btn_err1, t_err1_1]
            Intns_0 = np.r_[Intns_0_0, betwn_Intns, Intns_0_1]
            Intns_r_0 = np.r_[Intns_r_0_0, betwn_r, Intns_r_0_1]
            Intns_err_0 = np.r_[Intns_err_0_0, betwn_err, Intns_err_0_1]
            Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

            # median difference
            with h5py.File(load + 'rich_sample/stack_sky_median_%d_imgs_%s_band_%drich.h5' % 
                (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                BCG_sky = np.array(f['a'])

            with h5py.File(load + 'rich_sample/M_sky_rndm_median_%d_imgs_%s_band_%drich.h5' % 
                (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                rand_sky = np.array(f['a'])
            differ_img = BCG_sky - rand_sky

            R_sky, sky_ICL, sky_err0, sky_err1, Intns, Intns_r, Intns_err = SB_pro(differ_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
            Intns, Intns_err = Intns / pixel**2, Intns_err / pixel**2

            ## add the sky difference image
            add_img = clust_img + differ_img
            R_add_0, SB_add_0, add_err0_0, add_err1_0, Intns_1_0, Intns_r_1_0, Intns_err_1_0 = SB_pro(
                add_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
            R_add_1, SB_add_1, add_err0_1, add_err1_1, Intns_1_1, Intns_r_1_1, Intns_err_1_1 = SB_pro(
                add_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
            betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(add_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

            R_add = np.r_[R_add_0, betwn_r, R_add_1]
            SB_add = np.r_[SB_add_0, betwn_lit, SB_add_1]
            add_err0 = np.r_[add_err0_0, btn_err0, add_err0_1]
            add_err1 = np.r_[add_err1_0, btn_err1, add_err1_1]
            Intns_1 = np.r_[Intns_1_0, betwn_Intns, Intns_1_1]
            Intns_r_1 = np.r_[Intns_r_1_0, betwn_r, Intns_r_1_1]
            Intns_err_1 = np.r_[Intns_err_1_0, betwn_err, Intns_err_1_1]
            Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2

            #minu_bl_img = add_img - betwn_Intns
            """
                # case 1 : measuring based on img
                cli_R_0, cli_SB_0, cli_err0_0, cli_err1_0, Intns_2_0, Intns_r_2_0, Intns_err_2_0 = SB_pro(
                    minu_bl_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
                cli_R_1, cli_SB_1, cli_err0_1, cli_err1_1, Intns_2_1, Intns_r_2_1, Intns_err_2_1 = SB_pro(
                    minu_bl_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
                betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(minu_bl_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

                cli_R = np.r_[cli_R_0, betwn_r, cli_R_1]
                cli_SB = np.r_[cli_SB_0, betwn_lit, cli_SB_1]
                cli_err0 = np.r_[cli_err0_0, btn_err0, cli_err0_1]
                cli_err1 = np.r_[cli_err1_0, btn_err1, cli_err1_1]
                Intns_2 = np.r_[Intns_2_0, betwn_Intns, Intns_2_1]
                Intns_r_2 = np.r_[Intns_r_2_0, betwn_r, Intns_r_2_1]
                Intns_err_2 = np.r_[Intns_err_2_0, betwn_err, Intns_err_2_1]
                Intns_2, Intns_err_2 = Intns_2 / pixel**2, Intns_err_2 / pixel**2

                # case 2 : result from SB profile deviation
                Resi_bl = betwn_Intns / pixel**2
                Resi_std = betwn_err / pixel**2
                Resi_sky = betwn_lit

                cli_R = Intns_r_1 * 1.
                Intns_2 = Intns_1 - Resi_bl
                Intns_r_2 = Intns_r_1 * 1.
                #Intns_err_2 = Intns_err_1 * 1.
                Intns_err_2 = np.sqrt(Intns_err_1**2 + Resi_std**2)

                cli_SB = 22.5 - 2.5 * np.log10(Intns_2) + mag_add[kk]
                cli_dSB0 = 22.5 - 2.5 * np.log10(Intns_2 + Intns_err_2) + mag_add[kk]
                cli_dSB1 = 22.5 - 2.5 * np.log10(Intns_2 - Intns_err_2) + mag_add[kk]
                err0 = cli_SB - cli_dSB0
                err1 = cli_dSB1 - cli_SB
                id_nan = np.isnan(cli_SB)
                cli_SB, cli_R, cli_err0, cli_err1 = cli_SB[id_nan == False], cli_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
                cli_dSB0, cli_dSB1 = cli_dSB0[id_nan == False], cli_dSB1[id_nan == False]
                idx_nan = np.isnan(cli_dSB1)
                cli_err1[idx_nan] = 100.

                # case 3 : subtract random point SB + residual background
                with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_imgs.h5' % band[kk], 'r') as f:
                    rand_field = np.array(f['a'])

                #pre_minu_img = add_img - rand_field
                #betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(pre_minu_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)
                #minu_bl_img = pre_minu_img - betwn_Intns
                minu_bl_img = add_img - rand_field
            """
            # case 4 : subtracted random point (pixel) SB pdf mean / median
            #with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_median_pdf.h5' % band[kk], 'r') as f:
            with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_mean_pdf.h5' % band[kk], 'r') as f:
                dmp_array = np.array(f['a'])
            flux_mean, num_mean = dmp_array[0], dmp_array[1]
            dx = flux_mean[1:] - flux_mean[:-1]
            dx = np.r_[dx[0], dx]
            M_flux_mean = np.sum(flux_mean * num_mean * dx)
            minu_bl_img = add_img - M_flux_mean

            cli_R_0, cli_SB_0, cli_err0_0, cli_err1_0, Intns_2_0, Intns_r_2_0, Intns_err_2_0 = SB_pro(
                minu_bl_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
            cli_R_1, cli_SB_1, cli_err0_1, cli_err1_1, Intns_2_1, Intns_r_2_1, Intns_err_2_1 = SB_pro(
                minu_bl_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
            betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(minu_bl_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

            cli_R = np.r_[cli_R_0, betwn_r, cli_R_1]
            cli_SB = np.r_[cli_SB_0, betwn_lit, cli_SB_1]
            cli_err0 = np.r_[cli_err0_0, btn_err0, cli_err0_1]
            cli_err1 = np.r_[cli_err1_0, btn_err1, cli_err1_1]
            Intns_2 = np.r_[Intns_2_0, betwn_Intns, Intns_2_1]
            Intns_r_2 = np.r_[Intns_r_2_0, betwn_r, Intns_r_2_1]
            Intns_err_2 = np.r_[Intns_err_2_0, betwn_err, Intns_err_2_1]
            Intns_2, Intns_err_2 = Intns_2 / pixel**2, Intns_err_2 / pixel**2

            ## re-calculate SB profile
            sub_pros = 22.5 - 2.5 * np.log10(Intns_2) + mag_add[kk]
            dSB0 = 22.5 - 2.5 * np.log10(Intns_2 + Intns_err_2) + mag_add[kk]
            dSB1 = 22.5 - 2.5 * np.log10(Intns_2 - Intns_err_2) + mag_add[kk]
            err0 = sub_pros - dSB0
            err1 = dSB1 - sub_pros
            idx_nan = np.isnan(dSB1)
            err1[idx_nan] = 100.
            sub_SB.append(sub_pros)
            sub_R.append(Intns_r_2)
            sub_err0.append(err0)
            sub_err1.append(err1)

            ## fig the result
            plt.figure(figsize = (16, 12))
            bx0 = plt.subplot(221)
            bx1 = plt.subplot(222)
            bx2 = plt.subplot(223)
            bx3 = plt.subplot(224)
            if lamda_k == 0:
                bx0.set_title('$ %s \; band \; stack \; img [20 \\leqslant \\lambda \\leqslant 30] $' % band[kk])
            elif lamda_k == 1:
                bx0.set_title('$ %s \; band \; stack \; img [30 \\leqslant \\lambda \\leqslant 50] $' % band[kk])
            else:
                bx0.set_title('$ %s \; band \; stack \; img [ \\lambda \\geq 50 ] $' % band[kk])
            clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust01 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
            tf = bx0.imshow(clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = bx0, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
            ## add contour
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

            bx1.set_title('%s band difference img' % band[kk] )
            clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-',)
            clust11 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--',)
            tf = bx1.imshow(differ_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4)
            plt.colorbar(tf, ax = bx1, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
            bx1.add_patch(clust10)
            bx1.add_patch(clust11)
            bx1.axis('equal')
            bx1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
            bx1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
            bx1.set_xticks([])
            bx1.set_yticks([])

            bx2.set_title('%s band difference + stack img' % band[kk] )
            clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
            tf = bx2.imshow(add_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = bx2, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
            ## add contour
            kernl_img = ndimage.gaussian_filter(add_img, sigma = dnoise,  mode = 'nearest')
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

            bx3.set_title('%s band difference + stack - RBL' % band[kk] )
            clust30 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust31 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)

            tf = bx3.imshow(minu_bl_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = bx3, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
            ## add contour
            kernl_img = ndimage.gaussian_filter(minu_bl_img, sigma = dnoise,  mode = 'nearest')
            SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
            tg = bx3.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
            plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

            bx3.add_patch(clust30)
            bx3.add_patch(clust31)
            bx3.axis('equal')
            bx3.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
            bx3.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
            bx3.set_xticks([])
            bx3.set_yticks([])

            plt.tight_layout()
            if lamda_k == 0:
                plt.savefig(out_load + 'low_rich_%s_band_process.png' % band[kk], dpi = 300)
            elif lamda_k == 1:
                plt.savefig(out_load + 'median_rich_%s_band_process.png' % band[kk], dpi = 300)
            else:
                plt.savefig(out_load + 'high_rich_%s_band_process.png' % band[kk], dpi = 300) 
            plt.close()

            plt.figure()
            cx0 = plt.subplot(111)
            if lamda_k == 0:
                cx0.set_title('$ %s \, band \, SB \, profile \, [20 \\leqslant \\lambda \\leqslant 30] $' % band[kk])
            elif lamda_k == 1:
                cx0.set_title('$ %s \, band \, SB \, profile \, [30 \\leqslant \\lambda \\leqslant 50] $' % band[kk])
            else:
                cx0.set_title('$ %s \, band \, SB \, profile \, [ \\lambda \\geq 50 ] $' % band[kk])

            cx0.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
                ecolor = 'r', elinewidth = 1, label = 'img ICL + background', alpha = 0.5)
            cx0.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
                ecolor = 'g', elinewidth = 1, label = 'img + ICL + background', alpha = 0.5)
            cx0.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1, 
                ecolor = 'b', elinewidth = 1, label = 'img ICL + sky ICL', alpha = 0.5)
            #cx0.errorbar(R_sky, sky_ICL, yerr = [sky_err0, sky_err1], xerr = None, color = 'm', marker = 'None', ls = '--', linewidth = 1, 
            #    ecolor = 'm', elinewidth = 1, label = '$ sky \, ICL $', alpha = 0.5)
            cx0.plot(R_sky, sky_ICL, color = 'm', ls = '-', alpha = 0.5, label = '$ sky \, ICL $',)

            cx0.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
            cx0.plot(R_obs, SB_Z05, 'k:', label = 'Sersic', alpha = 0.5)
            cx0.set_xlabel('$R[kpc]$')
            cx0.set_ylabel('$SB[mag / arcsec^2]$')
            cx0.set_xscale('log')
            cx0.set_ylim(20, 35)
            cx0.set_xlim(1, 2e3)
            cx0.legend(loc = 1)
            cx0.invert_yaxis()
            cx0.grid(which = 'both', axis = 'both')
            cx0.tick_params(axis = 'both', which = 'both', direction = 'in')

            if lamda_k == 0:
                plt.savefig(out_load + '%s_band_low_rich_SB.png' % band[kk], dpi = 300)
            elif lamda_k == 1:
                plt.savefig(out_load + '%s_band_median_rich_SB.png' % band[kk], dpi = 300)
            else:
                plt.savefig(out_load + '%s_band_high_rich_SB.png' % band[kk], dpi = 300) 
            plt.close()

            if lamda_k == 0:
                #ax0.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1, 
                #    ecolor = 'b', elinewidth = 1, label = '$ 20 \\leqslant \\lambda \\leqslant 30 $', alpha = 0.5)
                ax0.plot(cli_R, cli_SB, color = 'b', alpha = 0.5, ls = '-',)
                ax0.fill_between(cli_R, y1 = cli_SB - cli_err0, y2 = cli_SB + cli_err1, color = 'b', alpha = 0.30, 
                    label = '$ 20 \\leq \\lambda \\leq 30 $')
            elif lamda_k == 1:
                #ax0.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
                #    ecolor = 'g', elinewidth = 1, label = '$ 30 \\leqslant \\lambda \\leqslant 50 $', alpha = 0.5)
                ax0.plot(cli_R, cli_SB, color = 'g', alpha = 0.5, ls = '-',)
                ax0.fill_between(cli_R, y1 = cli_SB - cli_err0, y2 = cli_SB + cli_err1, color = 'g', alpha = 0.30, 
                    label = '$ 30 \\leq \\lambda \\leq 50 $')
            else:
                #ax0.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
                #    ecolor = 'r', elinewidth = 1, label = '$ \\lambda \\geq 50 $', alpha = 0.5)
                ax0.plot(cli_R, cli_SB, color = 'r', alpha = 0.5, ls = '-',)
                ax0.fill_between(cli_R, y1 = cli_SB - cli_err0, y2 = cli_SB + cli_err1, color = 'r', alpha = 0.30, 
                    label = '$ \\lambda \\geq 50 $')

        ax0.set_xlabel('$R[kpc]$')
        ax0.set_ylabel('$SB[mag / arcsec^2]$')
        ax0.set_xscale('log')
        ax0.set_ylim(19, 34)
        ax0.set_xlim(1, 2e3)
        ax0.legend(loc = 1)
        ax0.invert_yaxis()
        ax0.grid(which = 'both', axis = 'both')
        ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

        ## deviation comparison
        id_nan = np.isnan(sub_SB[1])
        id_inf = np.isinf(sub_SB[1])
        idu = id_nan | id_inf
        inter_r = sub_R[1][idu == False]
        inter_sb = sub_SB[1][idu == False]
        f_SB = interp.interp1d(inter_r, inter_sb, kind = 'cubic')

        id_nan = np.isnan(sub_SB[0])
        id_inf = np.isinf(sub_SB[0])
        idu = id_nan | id_inf
        id_R0 = sub_R[0][idu == False]
        id_SB0 = sub_SB[0][idu == False]
        id_err0, id_err1 = sub_err0[0][idu == False], sub_err1[0][idu == False]
        idx = (id_R0 > np.min(inter_r)) & (id_R0 < np.max(inter_r))
        dev_R0 = id_R0[idx]
        dev_SB0 = id_SB0[idx] - f_SB(dev_R0)
        dev_err0_0, dev_err0_1 = id_err0[idx], id_err1[idx]

        id_nan = np.isnan(sub_SB[2])
        id_inf = np.isinf(sub_SB[2])
        idu = id_nan | id_inf
        id_R2 = sub_R[2][idu == False]
        id_SB2 = sub_SB[2][idu == False]
        id_err0, id_err1 = sub_err0[2][idu == False], sub_err1[2][idu == False]
        idx = (id_R2 > np.min(inter_r)) & (id_R2 < np.max(inter_r))
        dev_R2 = id_R2[idx]
        dev_SB2 = id_SB2[idx] - f_SB(dev_R2)
        dev_err2_0, dev_err2_1 = id_err0[idx], id_err1[idx]

        ax1.plot(dev_R2, dev_SB2, color = 'r', alpha = 0.5, ls = '-')
        ax1.fill_between(dev_R2, y1 = dev_SB2 - dev_err2_0, y2 = dev_SB2 + dev_err2_1, color = 'r', alpha = 0.30,)
        ax1.plot(sub_R[1], sub_SB[1] - sub_SB[1], color = 'g', alpha = 0.5, ls = '-')
        ax1.fill_between(sub_R[1], y1 = 0 - sub_err0[1], y2 = 0 + sub_err1[1], color = 'g', alpha = 0.30,)
        ax1.plot(dev_R0, dev_SB0, color = 'b', alpha = 0.5, ls = '-')
        ax1.fill_between(dev_R0, y1 = dev_SB0 - dev_err0_0, y2 = dev_SB0 + dev_err0_1, color = 'b', alpha = 0.30,)

        ax1.set_xlim(ax0.get_xlim())
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xscale('log')
        ax1.set_xlabel('$ R[kpc] $')
        ax1.set_ylabel('$ SB - SB_{30 \\leq \\lambda \\leq 50} $')
        ax1.grid(which = 'both', axis = 'both')
        ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
        ax0.set_xticklabels([])

        plt.subplots_adjust(hspace = 0.05)
        plt.savefig(out_load + '%s_band_SB_rich_binned.png' % band[kk], dpi = 300)
        plt.close()

        ## scaled with R200
        plt.figure()
        ax = plt.subplot(111)
        ax.set_title('%s band scaled SB profile' % band[kk],)
        ax.plot(sub_R[2] / R200[2], sub_SB[2], color = 'r', alpha = 0.5, ls = '-', label = '$ \\lambda \\geq 50 $')
        ax.fill_between(sub_R[2] / R200[2], y1 = sub_SB[2] - sub_err0[2], y2 = sub_SB[2] + sub_err1[2], color = 'r', alpha = 0.30,)
        ax.plot(sub_R[1] / R200[1], sub_SB[1], color = 'g', alpha = 0.5, ls = '-', label = '$ 30 \\leq \\lambda \\leq 50 $')
        ax.fill_between(sub_R[1] / R200[1], y1 = sub_SB[1] - sub_err0[1], y2 = sub_SB[1] + sub_err1[1], color = 'g', alpha = 0.30,)
        ax.plot(sub_R[0] / R200[0], sub_SB[0], color = 'b', alpha = 0.5, ls = '-', label = '$ 20 \\leq \\lambda \\leq 30 $')
        ax.fill_between(sub_R[0] / R200[0], y1 = sub_SB[0] - sub_err0[0], y2 = sub_SB[0] + sub_err1[0], color = 'b', alpha = 0.30,)  

        ax.set_xlabel('$ R / R_{200}$')
        ax.set_ylabel('$SB[mag / arcsec^2]$')
        ax.set_xscale('log')
        ax.set_ylim(19, 34)
        ax.set_xlim(1e-3, 2e0)
        ax.legend(loc = 1, frameon = False)
        ax.invert_yaxis()
        ax.grid(which = 'both', axis = 'both')
        ax.tick_params(axis = 'both', which = 'both', direction = 'in')

        plt.savefig(out_load + '%s_band_R200_scaled_SB_rich_bin.png' % band[kk], dpi = 300)
        plt.close()
    '''
    ## stacking centered on image center
    for kk in range(rank, rank + 1):
        for lamda_k in range(3):

            with h5py.File(tmp + 'test/%d_rich_img-center-stack_%s_band.h5' % (lamda_k, band[kk]), 'r') as f:
                stack_img = np.array(f['a'])

            ## measure SB
            bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
            Rt, SBt, t_err0, t_err1, Intns_0, Intns_r_0, Intns_err_0 = SB_pro_0z(stack_img, pixel, 1, 3 * Rpp, x0, y0, bins, kk)
            Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

            plt.figure()
            bx0 = plt.subplot(111)
            bx0.set_title('%s band %d rich sample [centered on image center]' % (band[kk], lamda_k) )

            clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust01 = Circle(xy = (x0, y0), radius = 2 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
            tf = bx0.imshow(stack_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e-1, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = bx0, fraction = 0.050, pad = 0.01, label = 'flux[nmaggy]')
            bx0.add_patch(clust00)
            bx0.add_patch(clust01)
            bx0.axis('equal')
            bx0.set_xlim(x0 - np.ceil(1.3 * Rpp), x0 + np.ceil(1.3 * Rpp))
            bx0.set_ylim(y0 - np.ceil(Rpp), y0 + np.ceil(Rpp))
            bx0.set_xticks([])
            bx0.set_yticks([])
            plt.savefig(tmp + 'test/%s_band_%d_sub-center-stack_2D.png' % (band[kk], lamda_k), dpi = 300)
            plt.close()

            plt.figure(figsize = (12, 6))
            ax0 = plt.subplot(121)
            ax1 = plt.subplot(122)
            ax0.set_title('%s band %d rich sample SB [centered on image center]' % (band[kk], lamda_k) )

            ax0.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
                ecolor = 'r', elinewidth = 1, label = 'stacking random point imgs', alpha = 0.5)
            ax0.set_ylim(27, 29)
            ax0.set_ylabel('$SB[mag / arcsec^2]$')
            ax0.set_xlim(1, 1e3)
            ax0.set_xlabel('$ R[arcsec] $')
            ax0.set_xscale('log')
            ax0.legend(loc = 1, frameon = False)
            ax0.invert_yaxis()
            ax0.grid(which = 'both', axis = 'both')
            ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

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

            plt.tight_layout()
            plt.savefig(tmp + 'test/%s_band_%d_rich-sample_SB.png' % (band[kk], lamda_k), dpi = 300) 
            plt.close()
    '''
if __name__ == "__main__":
    main()
