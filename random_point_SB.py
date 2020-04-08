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
from Mass_rich_radius import rich2R

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
M_dot = 4.83 # the absolute magnitude of SUN

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def rand_pont(band_id, sub_z, sub_ra, sub_dec):

    stack_N = len(sub_z)
    ii = np.int(band_id)

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

        try:
            data_A = fits.open(load + 'random_cat/resample_img/rand-resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g) )
            img_A = data_A[0].data
            xn = data_A[0].header['CENTER_X']
            yn = data_A[0].header['CENTER_Y']

            la0 = np.int(y0 - yn)
            la1 = np.int(y0 - yn + img_A.shape[0])
            lb0 = np.int(x0 - xn)
            lb1 = np.int(x0 - xn + img_A.shape[1])

            idx = np.isnan(img_A)
            idv = np.where(idx == False)

            sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
            count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
            id_nan = np.isnan(count_array_A)
            id_fals = np.where(id_nan == False)
            p_count_A[id_fals] = p_count_A[id_fals] + 1.
            count_array_A[la0: la1, lb0: lb1][idx] = np.nan
            id_nm += 1.

        except FileNotFoundError:
            continue

    p_count_A[0, 0] = id_nm
    with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
        f['a'] = np.array(sum_array_A)
    with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
        f['a'] = np.array(p_count_A)

    return

def main():
    ## random catalogue
    with h5py.File(load + 'mpi_h5/redMapper_rand_cat.h5', 'r') as f:
        tmp_array = np.array(f['a'])
    ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
    zN = len(z)

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    R_cut, bins = 1280, 80
    R_smal, R_max = 1, 1.7e3 # kpc

    for kk in range(3):

        m, n = divmod(zN, cpus)
        N_sub0, N_sub1 = m * rank, (rank + 1) * m
        if rank == cpus - 1:
            N_sub1 += n

        rand_pont(kk, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
        commd.Barrier()
        if rank == 0:

            tot_N = 0
            mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
            p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

            for pp in range(cpus):

                with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
                    p_count = np.array(f['a'])
                with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
                    sum_img = np.array(f['a'])

                tot_N += p_count[0, 0]
                id_zero = p_count == 0
                ivx = id_zero == False
                mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
                p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

                sub_mean = sum_img / p_count
                id_zeros = sub_mean == 0.
                sub_mean[id_zeros] = np.nan
                id_inf = np.isinf(sub_mean)
                sub_mean[id_inf] = np.nan

                plt.figure()
                ax = plt.subplot(111)
                ax.set_title('%s band %d cpus img' % (band[kk], pp) )
                clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
                clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
                tf = bx2.imshow(sub_mean, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
                plt.colorbar(tf, ax = ax, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')

                ax.add_patch(clust20)
                ax.add_patch(clust21)
                ax.axis('equal')
                ax.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
                ax.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
                ax.set_xticks([])
                ax.set_yticks([])

                plt.savefig(load + 'random_cat/stack/%s_band_%d_cpus_img.png' % (band[kk], pp), dpi = 300)
                plt.close()

            ## save the stack image
            id_zero = p_add_count == 0
            mean_img[id_zero] = np.nan
            p_add_count[id_zero] = np.nan
            tot_N = np.int(tot_N)
            stack_img = mean_img / p_add_count
            where_are_inf = np.isinf(stack_img)
            stack_img[where_are_inf] = np.nan

            with h5py.File(load + 'random_cat/stack/stack_A_%d_in_%s_band_%drich.h5' % (tot_N, band[kk], lamda_k), 'w') as f:
                f['a'] = np.array(stack_img)
            plt.figure()
            ax = plt.subplot(111)
            ax.set_title('%s band stack %d img' % (band[kk], tot_N) )
            clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
            clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
            tf = bx2.imshow(stack_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
            plt.colorbar(tf, ax = ax, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')

            ax.add_patch(clust20)
            ax.add_patch(clust21)
            ax.axis('equal')
            ax.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
            ax.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.savefig(load + 'random_cat/stack/%s_band_stack_%d_img.png' % (band[kk], tot_N), dpi = 300)
            plt.close()

        commd.Barrier()

if __name__ == "__main__" :
    main()
