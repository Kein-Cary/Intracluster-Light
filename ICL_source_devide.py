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
import astropy.io.ascii as asc
import astropy.io.fits as fits

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
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
                    4.63,  7.43,  11.42,  18.20,  28.20, 
                    44.21, 69.00, 107.81, 168.20, 263.00])

def sers_pro(r, mu_e, r_e, n):
    belta_n = 2 * n - 0.324
    fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
    mu_r = mu_e + fn
    return mu_r

def rich_divid(band_id, sub_z, sub_ra, sub_dec):

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
        Da_g = Test_model.angular_diameter_distance(z_g).value
        '''
        ## A mask imgs with edge pixels
        # 1.5sigma
        data_A = fits.getdata(load + 
            'resample/1_5sigma_larger_R/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
        '''
        ## A mask imgs without edge pixels
        data_A = fits.getdata(load + 
            'sky_select_img/imgs/cut_edge-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)

        img_A = data_A[0]
        xn = data_A[1]['CENTER_X']
        yn = data_A[1]['CENTER_Y']

        la0 = np.int(y0 - yn)
        la1 = np.int(y0 - yn + img_A.shape[0])
        lb0 = np.int(x0 - xn)
        lb1 = np.int(x0 - xn + img_A.shape[1])

        idx = np.isnan(img_A)
        idv = np.where(idx == False)
        BL = 0.
        sub_BL_img = img_A - BL

        sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + sub_BL_img[idv]
        count_array_A[la0: la1, lb0: lb1][idv] = sub_BL_img[idv]
        id_nan = np.isnan(count_array_A)
        id_fals = np.where(id_nan == False)
        p_count_A[id_fals] = p_count_A[id_fals] + 1.
        count_array_A[la0: la1, lb0: lb1][idv] = np.nan
        id_nm += 1.

    p_count_A[0, 0] = id_nm
    with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
        f['a'] = np.array(sum_array_A)
    with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
        f['a'] = np.array(p_count_A)

    return

def main():
    rich_a0, rich_a1, rich_a2 = 20, 30, 50
    ## sersic pro of Zibetti 05
    mu_e = np.array([23.87, 25.22, 23.4])
    r_e = np.array([19.29, 19.40, 20])

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    R_cut, bins = 1280, 80
    R_smal, R_max = 1, 1.7e3 # kpc
    '''
    for kk in range(3):

        for lamda_k in range(3):
            with h5py.File(load + 'sky_select_img/%s_band_sky_0.80Mpc_select.h5' % band[kk], 'r') as f:
                set_array = np.array(f['a'])
            set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[4,:]

            if lamda_k == 0:
                idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
            elif lamda_k == 1:
                idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
            else:
                idx = (set_rich >= rich_a2)

            lis_z = set_z[idx]
            lis_ra = set_ra[idx]
            lis_dec = set_dec[idx]
            lis_rich = set_rich[idx]

            zN = len(lis_z)
            m, n = divmod(zN, cpus)
            N_sub0, N_sub1 = m * rank, (rank + 1) * m
            if rank == cpus - 1:
                N_sub1 += n

            rich_divid(kk, lis_z[N_sub0 :N_sub1], lis_ra[N_sub0 :N_sub1], lis_dec[N_sub0 :N_sub1])
            commd.Barrier()
            if rank == 0:

                tot_N = 0
                mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
                p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

                for pp in range(cpus):

                    with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
                        p_count = np.array(f['a'])
                    with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
                        sum_img = np.array(f['a'])

                    tot_N += p_count[0, 0]
                    id_zero = p_count == 0
                    ivx = id_zero == False
                    mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
                    p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

                ## save the stack image
                id_zero = p_add_count == 0
                mean_img[id_zero] = np.nan
                p_add_count[id_zero] = np.nan
                tot_N = np.int(tot_N)
                stack_img = mean_img / p_add_count
                where_are_inf = np.isinf(stack_img)
                stack_img[where_are_inf] = np.nan

                ## %drich : 0rich: low richness; 1rich: mid richness; 2rich: high richness
                with h5py.File(load + 'rich_sample/stack_A_%d_in_%s_band_%drich.h5' % (tot_N, band[kk], lamda_k), 'w') as f:
                    f['a'] = np.array(stack_img)

            commd.Barrier()
    '''
    #N_sum = np.array([2013, 2008, 2002, 2008, 2009]) ## sky-select sample(1Mpc)
    #N_bin = np.array([ [1137, 657, 212], [1133, 658, 210], [1131, 653, 211] ])

    N_sum = np.array([1291, 1286, 1283, 1294, 1287]) ## 0.8Mpc
    N_bin = np.array([ [711, 434, 141], [710, 430, 141], [707, 429, 142] ])

    r_a0, r_a1 = 1.0, 1.1
    ## R200 calculate parameter
    M0, lamd0, z0 = 14.37, 30, 0.5
    F_lamda, G_z = 1.12, 0.18
    V_num = 200

    if rank == 0:
        for kk in range(3):

            SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
            R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
            R_obs = R_obs**4
            ## sersic part
            Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
            SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

            ## R200 calculate
            with h5py.File(load + 'sky_select_img/%s_band_sky_0.80Mpc_select.h5' % band[kk], 'r') as f:
                set_array = np.array(f['a'])
            set_z, set_rich = set_array[2,:], set_array[4,:]

            ## record the SB profile of each rich bin
            SB_arr = []
            flux_arr = []
            R_arr = []
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

                R_vir = rich2R(lis_z, lis_rich, M0, lamd0, z0, F_lamda, G_z, V_num)
                R200[lamda_k] = np.nanmean(R_vir)

                with h5py.File(load + 'rich_sample/stack_A_%d_in_%s_band_%drich.h5' % (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                    stack_img = np.array(f['a'])
                ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

                Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
                SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
                id_nan = np.isnan(SB)
                SBt, Rt = SB[id_nan == False], Intns_r[id_nan == False]
                '''
                ## read difference img
                # mean difference
                with h5py.File(load + 'rich_sample/stack_sky_mean_%d_imgs_%s_band_%drich.h5' % 
                    (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                    BCG_sky = np.array(f['a'])

                with h5py.File(load + 'rich_sample/M_sky_rndm_mean_%d_imgs_%s_band_%drich.h5' % 
                    (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                    rand_sky = np.array(f['a'])                
                differ_img = BCG_sky - rand_sky
                '''
                # median difference
                with h5py.File(load + 'rich_sample/stack_sky_median_%d_imgs_%s_band_%drich.h5' % 
                    (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                    BCG_sky = np.array(f['a'])

                with h5py.File(load + 'rich_sample/M_sky_rndm_median_%d_imgs_%s_band_%drich.h5' % 
                    (N_bin[kk, lamda_k], band[kk], lamda_k), 'r') as f:
                    rand_sky = np.array(f['a'])
                differ_img = BCG_sky - rand_sky

                resi_add = differ_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

                add_img = ss_img + resi_add
                Intns, Intns_r, Intns_err, Npix = light_measure(add_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
                SB_add = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
                R_add = Intns_r * 1

                cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
                BL_img = add_img * 1
                grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
                grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
                grd = np.array( np.meshgrid(grd_x, grd_y) )
                ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
                idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
                Resi_bl = np.nanmean( BL_img[idu] )

                # minus the RBL
                sub_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
                flux0 = Intns + Intns_err - Resi_bl
                flux1 = Intns - Intns_err - Resi_bl
                dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
                dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
                err0 = sub_SB - dSB0
                err1 = dSB1 - sub_SB

                SB_arr.append(sub_SB)
                flux_arr.append(Intns - Resi_bl) ## ratio of SB need to use flux
                R_arr.append(R_add)

                id_nan = np.isnan(sub_SB)
                cli_SB, cli_R, cli_err0, cli_err1 = sub_SB[id_nan == False], R_add[id_nan == False], err0[id_nan == False], err1[id_nan == False]
                dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
                idx_nan = np.isnan(dSB1)
                cli_err1[idx_nan] = 100.

                if lamda_k == 0:
                    ax0.plot(cli_R, cli_SB, 'b-', alpha = 0.5, label = '$ 20 \\leqslant \\lambda \\leqslant 30 $')
                elif lamda_k == 1:
                    ax0.plot(cli_R, cli_SB, 'g-', alpha = 0.5, label = '$ 30 \\leqslant \\lambda \\leqslant 50 $')
                else:
                    ax0.plot(cli_R, cli_SB, 'r-', alpha = 0.5, label = '$ 50  \\leqslant \\lambda $')

            ax0.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
            ax0.plot(R_obs, SB_Z05, 'k:', label = 'Sersic', alpha = 0.5)
            ax0.set_xlabel('$R[kpc]$')
            ax0.set_ylabel('$SB[mag / arcsec^2]$')
            ax0.set_xscale('log')
            ax0.set_ylim(20, 32)
            ax0.set_xlim(1, 1.5e3)
            ax0.legend(loc = 1)
            ax0.invert_yaxis()
            ax0.grid(which = 'both', axis = 'both')
            ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

            SB_arr = np.array(SB_arr)
            Rarr = np.array(R_arr)
            flux_arr = np.array(flux_arr)
            Rarr = np.nanmean(Rarr, axis = 0)

            ax1.plot(Rarr, flux_arr[0,:] / flux_arr[2,:], 'b-', alpha = 0.5, label = '$ low \; \\lambda / high \; \\lambda $')
            ax1.plot(Rarr, flux_arr[1,:] / flux_arr[2,:], 'g-', alpha = 0.5, label = '$ mid \; \\lambda / high \; \\lambda $')
            ax1.set_xlim(ax0.get_xlim())
            ax1.set_xscale('log')
            ax1.set_xlabel('$R[kpc]$')
            ax1.set_ylabel('$ SB / SB_{50 \\leqslant \\lambda} $')
            ax1.legend(loc = 2)
            ax1.grid(which = 'both', axis = 'both')
            ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
            ax0.set_xticks([])

            plt.subplots_adjust(hspace = 0.01)
            #plt.savefig(load + 'rich_sample/%s_band_SB_rich_binned.png' % band[kk], dpi = 300)
            plt.savefig(load + 'rich_sample/%s_band_SB_rich_binned_median.png' % band[kk], dpi = 300)
            plt.close()

            ## scaled with R200
            plt.figure()
            ax = plt.subplot(111)
            ax.set_title('%s band scaled SB profile' % band[kk],)
            ax.plot(R_arr[0] / R200[0], SB_arr[0], 'b-', alpha = 0.5, label = '$ 20 \\leqslant \\lambda \\leqslant 30 $')
            ax.plot(R_arr[1] / R200[1], SB_arr[1], 'g-', alpha = 0.5, label = '$ 30 \\leqslant \\lambda \\leqslant 50 $')
            ax.plot(R_arr[2] / R200[2], SB_arr[2], 'r-', alpha = 0.5, label = '$ 50 \\leqslant \\lambda $')
            ax.set_xlabel('$ R / R_{200}$')
            ax.set_ylabel('$SB[mag / arcsec^2]$')
            ax.set_xscale('log')
            ax.set_ylim(20, 32)
            #ax.set_xlim(1, 1.5e3)
            ax.legend(loc = 1,)
            ax.invert_yaxis()
            ax.grid(which = 'both', axis = 'both')
            ax.tick_params(axis = 'both', which = 'both', direction = 'in')

            #plt.savefig(load + 'rich_sample/%s_band_R200_scaled_SB_rich_bin.png' % band[kk], dpi = 300)
            plt.savefig(load + 'rich_sample/%s_band_R200_scaled_SB_rich_bin_median.png' % band[kk], dpi = 300)
            plt.close()

    commd.Barrier()
    raise

if __name__ == "__main__":
    main()
