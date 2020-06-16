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
from light_measure import light_measure, flux_recal
from scipy.stats import binned_statistic as binned

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
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']

def rand_pont(band_id, sub_z, sub_ra, sub_dec):

    stack_N = len(sub_z)
    kk = np.int(band_id)

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
    p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    id_nm = 0.
    for jj in range(stack_N):

        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]

        data_A = fits.open(load + 'random_cat/mask_no_dust/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g)) ## just masking

        img_A = data_A[0].data
        head = data_A[0].header
        wcs_lis = awc.WCS(head)
        xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

        # centered on cat.(ra, dec)
        la0 = np.int(y0 - yn)
        la1 = np.int(y0 - yn + img_A.shape[0])
        lb0 = np.int(x0 - xn)
        lb1 = np.int(x0 - xn + img_A.shape[1])
        '''
        #rnx, rny = np.random.choice(img_A.shape[1], 1, replace = False), np.random.choice(img_A.shape[0], 1, replace = False) ## random center
        rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image center
        la0 = np.int(y0 - rny)
        la1 = np.int(y0 - rny + img_A.shape[0])
        lb0 = np.int(x0 - rnx)
        lb1 = np.int(x0 - rnx + img_A.shape[1])
        '''
        idx = np.isnan(img_A)
        idv = np.where(idx == False)

        sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
        count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
        id_nan = np.isnan(count_array_A)
        id_fals = np.where(id_nan == False)
        p_count_A[id_fals] = p_count_A[id_fals] + 1.
        count_array_A[la0: la1, lb0: lb1][idv] = np.nan
        id_nm += 1.

    p_count_A[0, 0] = id_nm
    with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(sum_array_A)
    with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(p_count_A)

    return

def sky_stack(band_id, sub_z, sub_ra, sub_dec):
    stack_N = len(sub_z)
    kk = np.int(band_id)

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
    p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

    id_nm = 0
    for jj in range(stack_N):
        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]
        '''
        #### after pix-resampling sky imgs
        data = fits.open(load + 'random_cat/sky_edge-cut_img/rand_Edg_cut-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
        img = data[0].data
        cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
        '''
        data = fits.open(load + 'random_cat/sky_img/rand_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) )
        img = data[0].data
        head = data[0].header
        wcs_lis = awc.WCS(head)
        cx, cy = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

        ## catalog (ra, dec)
        la0 = np.int(y0 - cy)
        la1 = np.int(y0 - cy + img.shape[0])
        lb0 = np.int(x0 - cx)
        lb1 = np.int(x0 - cx + img.shape[1])
        '''
        ## image frame center / random center
        #rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False)
        rnx, rny = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
        la0 = np.int(y0 - rny)
        la1 = np.int(y0 - rny + img.shape[0])
        lb0 = np.int(x0 - rnx)
        lb1 = np.int(x0 - rnx + img.shape[1])
        '''
        idx = np.isnan(img)
        idv = np.where(idx == False)

        sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
        count_array[la0: la1, lb0: lb1][idv] = img[idv]
        id_nan = np.isnan(count_array)
        id_fals = np.where(id_nan == False)
        p_count[id_fals] = p_count[id_fals] + 1
        count_array[la0: la1, lb0: lb1][idv] = np.nan
        id_nm += 1.

    p_count[0, 0] = id_nm
    with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(sum_array)
    with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(p_count)
    return

def binned_img_flux():
    N_bin = 30
    bin_side = np.linspace(-1e-1, 1e-1, N_bin + 1)

    for kk in range(rank, rank + 1):

        with h5py.File(load + 'random_cat/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
            tmp_array = np.array(f['a'])
        ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
        zN = len(z)

        bin_flux_mean = np.zeros((zN, N_bin), dtype = np.float)
        bin_flux_median = np.zeros((zN, N_bin), dtype = np.float)
        bin_num = np.zeros((zN, N_bin), dtype = np.float)

        for jj in range(zN):
            ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]
            data_A = fits.open(load + 'random_cat/edge_cut_img/rand_pont_Edg_cut-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
            img_A = data_A[0].data

            id_nan = np.isnan(img_A)
            flux_array = img_A[id_nan == False]

            value_mean = binned(flux_array, flux_array, statistic='mean', bins = bin_side)[0]
            value_median = binned(flux_array, flux_array, statistic='median', bins = bin_side)[0]
            value_number = binned(flux_array, flux_array, statistic='count', bins = bin_side)[0]
            bin_flux_mean[jj,:] = value_mean
            bin_flux_median[jj,:] = value_median
            bin_num[jj,:] = value_number / np.nansum(value_number)

        flux_mean = np.nanmean(bin_flux_mean, axis = 0) / pixel**2
        flux_median = np.nanmean(bin_flux_median, axis = 0) / pixel**2
        num_mean = np.nanmean(bin_num, axis = 0)

        dmp_array = np.array([flux_mean, num_mean])
        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_mean_pdf.h5' % band[kk], 'w') as f:
            f['a'] = np.array(dmp_array)
        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_mean_pdf.h5' % band[kk], ) as f:
            for ll in range(len(dmp_array)):
                f['a'][ll,:] = dmp_array[ll,:]

        dmp_array = np.array([flux_median, num_mean])
        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_median_pdf.h5' % band[kk], 'w') as f:
            f['a'] = np.array(dmp_array)
        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_median_pdf.h5' % band[kk], ) as f:
            for ll in range(len(dmp_array)):
                f['a'][ll,:] = dmp_array[ll,:]

        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_mean_pdf.h5' % band[kk], 'r') as f:
            dmp_array = np.array(f['a'])
        flux_mean, num_mean = dmp_array[0], dmp_array[1]
        dx = flux_mean[1:] - flux_mean[:-1]
        dx = np.r_[dx[0], dx]
        M_flux_mean = np.sum(flux_mean * num_mean * dx)

        with h5py.File(load + 'random_cat/stack/%s_band_random_field_pix_SB_median_pdf.h5' % band[kk], 'r') as f:
            dmp_array = np.array(f['a'])
        flux_median, num_mean = dmp_array[0], dmp_array[1]
        dx = flux_median[1:] - flux_median[:-1]
        dx = np.r_[dx[0], dx]
        M_flux_median = np.sum(flux_median * num_mean * dx)

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_title('%s band random filed pixel SB pdf' % band[kk])
        ax.plot(flux_mean, num_mean, c = 'r', ls = '-', label = 'Mean pixel SB')
        ax.axvline(x = M_flux_mean, color = 'r', linestyle = '-', alpha = 0.5)
        ax.plot(flux_median, num_mean, c = 'g', ls = '--', label = 'Median pixel SB')
        ax.axvline(x = M_flux_median, color = 'g', linestyle = '--', alpha = 0.5)
        ax.set_xlabel('$ pixel \; SB \; [nanomaggies / arcsec^2]$')
        ax.set_ylabel('pdf')
        ax.legend(loc = 1, frameon = False)
        ax.tick_params(axis = 'both', which = 'both', direction = 'in')

        subax = fig.add_axes([0.2, 0.35, 0.25, 0.4])
        subax.plot(flux_mean, num_mean, c = 'r', ls = '-', label = 'Mean pixel SB')
        subax.axvline(x = M_flux_mean, color = 'r', linestyle = '-', alpha = 0.5)        
        subax.set_xlim(-5e-4, 5e-4)
        xtick = subax.get_xticks()
        re_xtick = xtick * 1e4
        subax.set_xticks(xtick)
        subax.set_xticklabels(['%.1f' % ll for ll in re_xtick])
        subax.set_xlabel('1e4 * pixel SB')
        subax.tick_params(axis = 'both', which = 'both', direction = 'in')

        plt.savefig(load + 'random_cat/stack/%s_band_pix_SB_pdf.png' % band[kk], dpi = 300)
        plt.close()

    return

def main():

    #binned_img_flux()
    N_bin = 30
    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    ## stack cluster
    for kk in range( 1 ):

        with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
            tmp_array = np.array(f['a'])
        ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
        zN = len(z)

        set_z, set_ra, set_dec = z, ra, dec

        DN = len(set_z)
        m, n = divmod(DN, cpus)
        N_sub0, N_sub1 = m * rank, (rank + 1) * m
        if rank == cpus - 1:
            N_sub1 += n

        rand_pont(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
        commd.Barrier()
        if rank == 0:

            tot_N = 0.
            mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
            p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

            for pp in range(cpus):

                with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
                    p_count = np.array(f['a'])

                with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
                    sum_img = np.array(f['a'])

                id_zero = p_count == 0
                ivx = id_zero == False
                mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
                p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]
                tot_N += p_count[0, 0]

            ## save the stack image
            tot_N = np.int(tot_N)
            id_zero = p_add_count == 0
            mean_img[id_zero] = np.nan
            p_add_count[id_zero] = np.nan
            stack_img = mean_img / p_add_count
            where_are_inf = np.isinf(stack_img)
            stack_img[where_are_inf] = np.nan

            #### centered on the (ra, dec) in catalog (_stack_)
            #### centered on the image center (_center-stack_)

            ## after resampling imgs
            #with h5py.File(load + 'random_cat/stack/%s_band_stack_cluster_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)
            #with h5py.File(load + 'random_cat/stack/%s_band_center-stack_cluster_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)

            ## test smaple bias
            #with h5py.File(load + 'random_cat/stack/sample_test/%s_band_center-stack_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)
            #with h5py.File(load + 'random_cat/stack/sample_test/%s_band_stack_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)

            ## Extinction-corrected + masking imgs
            #with h5py.File(load + 'random_cat/angle_stack/%s_band_stack_cluster_Extinction-corrected-mask.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)
            #with h5py.File(load + 'random_cat/angle_stack/%s_band_center-stack_Extinction-corrected-mask.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)

            ## imgs without Extinction correction
            with h5py.File(load + 'random_cat/angle_stack/%s_band_stack_mask-only_imgs.h5' % band[kk], 'w') as f:
                f['a'] = np.array(stack_img)
            #with h5py.File(load + 'random_cat/angle_stack/%s_band_center-stack_mask-only_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)

        commd.Barrier()
    raise
    ## stack sky
    for kk in range( 3 ):

        with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
            tmp_array = np.array(f['a'])
        ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
        zN = len(z)
        da0, da1 = 0, zN

        set_z, set_ra, set_dec = z[da0:da1], ra[da0:da1], dec[da0:da1]
        DN = len(set_z)
        m, n = divmod(DN, cpus)
        N_sub0, N_sub1 = m * rank, (rank + 1) * m
        if rank == cpus - 1:
            N_sub1 += n
        sky_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
        commd.Barrier()

        ## combine all of the sub-stack imgs
        if rank == 0:

            bcg_stack = np.zeros((len(Ny), len(Nx)), dtype = np.float)
            bcg_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

            for pp in range(cpus):

                with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
                    p_count = np.array(f['a'])
                with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
                    sum_img = np.array(f['a'])

                id_zero = p_count == 0
                ivx = id_zero == False
                bcg_stack[ivx] = bcg_stack[ivx] + sum_img[ivx]
                bcg_count[ivx] = bcg_count[ivx] + p_count[ivx]

            ## centered on BCG
            id_zero = bcg_count == 0
            bcg_stack[id_zero] = np.nan
            bcg_count[id_zero] = np.nan
            stack_img = bcg_stack / bcg_count
            id_inf = np.isinf(stack_img)
            stack_img[id_inf] = np.nan

            ### sky img stacking
            #with h5py.File(load + 'random_cat/angle_stack/%s_band_center-stack_sky_imgs.h5' % band[kk], 'w') as f:
            #    f['a'] = np.array(stack_img)
            with h5py.File(load + 'random_cat/angle_stack/%s_band_stack_sky_imgs.h5' % band[kk], 'w') as f:
                f['a'] = np.array(stack_img)

        commd.Barrier()

if __name__ == "__main__" :
    main()
