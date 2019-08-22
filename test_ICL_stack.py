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
from numba import vectorize
from extinction_redden import A_wave
from scipy.optimize import curve_fit
from light_measure import light_measure, flux_recal
from light_measure import sigmamc
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
band = ['r', 'g', 'i', 'z', 'u']
mag_add = np.array([0, 0, 0, 0.02, -0.04])

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

def SB_fit(r, m0, Mc, c, M2L):
    bl = m0
    surf_mass = sigmamc(r, Mc, c)
    surf_lit = surf_mass / M2L

    Lz = surf_lit / ((1 + z_ref)**4 * np.pi * 4 * rad2asec**2)
    Lob = Lz * Lsun / kpc2cm**2
    fob = Lob/(10**(-9)*f0)
    mock_SB = 22.5 - 2.5 * np.log10(fob)

    mock_L = mock_SB + bl

    return mock_L

def stack_light(band_number, stack_number, subz, subra, subdec):
    stack_N = np.int(stack_number)
    ii = np.int(band_number)
    sub_z = subz
    sub_ra = subra
    sub_dec = subdec

    x0 = 2427
    y0 = 1765
    bins = 90
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    sum_grid = np.array(np.meshgrid(Nx, Ny))

    sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
    p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

    sum_array_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    count_array_B = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
    p_count_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)

    for jj in range(stack_N):
        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]

        Da_g = Test_model.angular_diameter_distance(z_g).value
        data = fits.getdata(load + 'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
        img = data[0]
        wcs = awc.WCS(data[1])
        cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

        data_B = fits.getdata(load + 'mask_data/B_plane/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
        img_B = data_B[0]

        Angur = (R0*rad2asec/Da_g)
        Rp = Angur/pixel
        L_ref = Da_ref*pixel/rad2asec
        L_z0 = Da_g*pixel/rad2asec
        b = L_ref/L_z0
        Rref = (R0*rad2asec/Da_ref)/pixel

        ox = np.linspace(0, img.shape[1]-1, img.shape[1])
        oy = np.linspace(0, img.shape[0]-1, img.shape[0])
        oo_grd = np.array(np.meshgrid(ox, oy))
        cdr = np.sqrt((oo_grd[0,:] - cx)**2 + (oo_grd[1,:] - cy)**2)
        idd = (cdr > Rp) & (cdr < 1.1 * Rp)
        cut_region = img[idd]
        id_nan = np.isnan(cut_region)
        idx = np.where(id_nan == False)
        bl_array = cut_region[idx]
        back_lel = np.mean(bl_array)
        cc_img = img - back_lel

        f_goal = flux_recal(cc_img, z_g, z_ref)
        xn, yn, resam = gen(f_goal, 1, b, cx, cy)
        xn = np.int(xn)
        yn = np.int(yn)
        if b > 1:
            resam_A = resam[1:, 1:]
        elif b == 1:
            resam_A = resam[1:-1, 1:-1]
        else:
            resam_A = resam

        la0 = np.int(y0 - yn)
        la1 = np.int(y0 - yn + resam_A.shape[0])
        lb0 = np.int(x0 - xn)
        lb1 = np.int(x0 - xn + resam_A.shape[1])

        idx = np.isnan(resam_A)
        idv = np.where(idx == False)
        sum_array[la0:la1, lb0:lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + resam_A[idv]
        count_array[la0: la1, lb0: lb1][idv] = resam_A[idv]
        id_nan = np.isnan(count_array)
        id_fals = np.where(id_nan == False)
        p_count[id_fals] = p_count[id_fals] + 1
        count_array[la0: la1, lb0: lb1][idv] = np.nan

        # stack img_B
        cc_img_B = img_B - back_lel
        f_B = flux_recal(cc_img_B, z_g, z_ref)
        xn, yn, resam = gen(f_B, 1, b, cx, cy)
        xn = np.int(xn)
        yn = np.int(yn)
        if b > 1:
            resam_B = resam[1:, 1:]
        elif b == 1:
            resam_B = resam[1:-1, 1:-1]
        else:
            resam_B = resam

        la0 = np.int(y0 - yn)
        la1 = np.int(y0 - yn + resam_B.shape[0])
        lb0 = np.int(x0 - xn)
        lb1 = np.int(x0 - xn + resam_B.shape[1])

        idx = np.isnan(resam_B)
        idv = np.where(idx == False)
        sum_array_B[la0:la1, lb0:lb1][idv] = sum_array_B[la0:la1, lb0:lb1][idv] + resam_B[idv]
        count_array_B[la0: la1, lb0: lb1][idv] = resam_B[idv]
        id_nan = np.isnan(count_array_B)
        id_fals = np.where(id_nan == False)
        p_count_B[id_fals] = p_count_B[id_fals] + 1
        count_array_B[la0: la1, lb0: lb1][idv] = np.nan

    mean_array_B = sum_array_B / p_count_B
    where_are_inf = np.isinf(mean_array_B)
    mean_array_B[where_are_inf] = np.nan
    id_zeros = np.where(p_count_B == 0)
    mean_array_B[id_zeros] = np.nan

    SB, R, Ar, error = light_measure(mean_array_B, bins, 1, Rpp, x0, y0, pixel, z_ref)
    SB_tot = SB[1:] + mag_add[ii]
    R_tot = R[1:]
    Ar_tot = Ar[1:]
    err_tot = error[1:]

    mean_array = sum_array / p_count
    where_are_inf = np.isinf(mean_array)
    mean_array[where_are_inf] = np.nan
    id_zeros = np.where(p_count == 0)
    mean_array[id_zeros] = np.nan

    SB, R, Ar, error = light_measure(mean_array, bins, 1, Rpp, x0, y0, pixel, z_ref)
    SB_diff = SB[1:] + mag_add[ii]
    R_diff = R[1:]
    Ar_diff = Ar[1:]
    err_diff = error[1:]

    ix = R_diff >= 100
    iy = R_diff <= 900
    iz = ix & iy
    r_fit = R_diff[iz]
    sb_fit = SB_diff[iz]

    m0 = 30.5
    mc = 14.5
    cc = 5
    m2l = 237
    po = np.array([m0, mc, cc, m2l])
    popt, pcov = curve_fit(SB_fit, r_fit, sb_fit, p0 = po, bounds = ([30, 13.5, 1, 200], [37, 15, 6, 270]), method = 'trf')

    M0 = popt[0]
    Mc = popt[1]
    Cc = popt[2]
    M2L = popt[3]
    fit_line = SB_fit(r_fit, M0, Mc, Cc, M2L)

    plt.figure(figsize = (16, 9))
    ax = plt.subplot(111)
    ax.set_title('$SB \; profile \; with \; BL \; subtracted \; in \; %s \; band$' % band[ii])
    ax.errorbar(R_diff, SB_diff, yerr = err_diff, xerr = None, ls = '', fmt = 'ro', label = '$BCG + ICL$')
    ax.errorbar(R_tot, SB_tot, yerr = err_tot, xerr = None, ls = '', fmt = 'bs', label = '$Total$')
    #ax.axhline(y = M0, linestyle = '--', linewidth = 1, color = 'g', label = '$Background$')

    ax.set_xlabel('$R[kpc]$')
    ax.set_xscale('log')
    ax.set_ylabel('$SB[mag/arcsec^2]$')
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    ax.invert_yaxis()
    ax.legend(loc = 1, fontsize = 12)

    ax1 = ax.twiny()
    xtik = ax.get_xticks(minor = True)
    xR = xtik * 10**(-3) * rad2asec / Da_ref
    ax1.set_xticks(xtik)
    ax1.set_xticklabels(["%.2f" % uu for uu in xR])
    ax1.set_xlim(ax.get_xlim())
    ax1.set_xlabel('$R[arcsec]$')
    ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

    plt.savefig(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_in_%sband_profile_compare.png' % (stack_N, band[ii]), dpi = 300)
    plt.close()

    fig = plt.figure(figsize = (16, 9))
    bx = plt.subplot(111)
    cx = fig.add_axes([0.15, 0.25, 0.175, 0.175])

    bx.set_title('$fit \; for \; background \; estimate \; in \; %s \; band$' % band[ii])
    bx.errorbar(R_diff[iz], SB_diff[iz], yerr = err_diff[iz], xerr = None, ls = '', fmt = 'ro', label = '$BCG + ICL$')
    bx.plot(r_fit, fit_line, 'b-', label = '$NFW+C$')

    bx.set_xlabel('$R[kpc]$')
    bx.set_xscale('log')
    bx.set_ylabel('$SB[mag/arcsec^2]$')
    bx.tick_params(axis = 'both', which = 'both', direction = 'in')
    bx.invert_yaxis()
    bx.legend(loc = 1, fontsize = 15)

    bx1 = bx.twiny()
    xtik = bx.get_xticks(minor = True)
    xR = xtik * 10**(-3) * rad2asec / Da_ref
    bx1.set_xticks(xtik)
    bx1.set_xticklabels(["%.2f" % uu for uu in xR])
    bx1.set_xlim(bx.get_xlim())
    bx1.set_xlabel('$R[arcsec]$')
    bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

    cx.text(0, 0, s = 'BL = %.2f' % M0 + '\n' + '$Mc = %.2fM_\odot $' % Mc + '\n' + 'C = %.2f' % Cc + '\n' + 'M2L = %.2f' % M2L, fontsize = 15)
    cx.axis('off')
    cx.set_xticks([])
    cx.set_yticks([])

    plt.savefig(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_in_%sband_with_NFW_fit.png' % (stack_N, band[ii]), dpi = 300)
    plt.close()

    return SB_diff, R_diff, Ar_diff, err_diff

def main():
    import random
    import matplotlib.gridspec as gridspec
    stackN = np.int(20)

    ix = (red_rich >= 25) & (red_rich <= 27.5)
    RichN = red_rich[ix]
    zN = red_z[ix]
    raN = red_ra[ix]
    decN = red_dec[ix]

    tt0 = [random.randint(0, 422) for _ in range(stackN)]
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/random_index.h5', 'w') as f:
        f['a'] = np.array(tt0)

    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/random_index.h5') as f:
        tt0 = np.array(f['a'])

    richa = RichN[tt0]
    za = zN[tt0]
    ra_a = raN[tt0]
    dec_a = decN[tt0]
    
    for ii in range(len(band)):
        id_band = ii
        SB, R, Ar, err = stack_light(id_band, stackN, za, ra_a, dec_a)

        SB0 = SB *1
        R0 = R *1
        Ar0 = Ar *1
        err0 = err *1

        tmp_SB = np.array([SB0, R0, Ar0, err0])        
        with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_in_%s_band_%.2f_sigma.h5' % (stackN, band[ii], 1.5), 'w') as f:
            f['a'] = tmp_SB
        with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_in_%s_band_%.2f_sigma.h5' % (stackN, band[ii], 1.5)) as f:
            for kk in range(len(tmp_SB)):
                f['a'][kk,:] = tmp_SB[kk,:]
    print('tmp_saved!!')
    raise
    Mean_rich = np.mean(richa)
    Medi_rich = np.median(richa)
    Mstd_rich = np.std(richa)
    stackn = np.array([25, 50, 75])
    eps = 0.5
    for ii in range(len(band)):

        id_band = ii
        with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_in_%s_band_%.2f_sigma.h5' % (stackN, band[ii], 1.5)) as f:
            comp = np.array(f['a'])
        SB0 = comp[0,:]
        R0 = comp[1,:]
        Ar0 = comp[2,:]
        err0 = comp[3,:]

        for tt in range(len(stackn)):
            tes1 = 0
            tes2 = 0
            while (np.abs(Mean_rich - tes1) > eps) & (np.abs(Medi_rich - tes2) > eps):
                # random select sample
                tt1 = [random.randint(0, 99) for _ in range(stackn[tt])]
                rich1 = richa[tt1]
                tes1 = np.mean(rich1)
                tes2 = np.median(rich1)
            print('Now stack N is %d' % stackn[tt])
            richb = richa[tt1]
            m_rich = np.mean(richb)
            zb = za[tt1]
            ra_b = ra_a[tt1]
            dec_b = dec_a[tt1]
            SB, R, Ar, err = stack_light(id_band, stackn[tt], zb, ra_b, dec_b)

            fig = plt.figure(figsize = (16,12))
            gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
            bx = plt.subplot(gs[0])
            cx = plt.subplot(gs[1])
            bx.set_title('$stack \; light \; test \; in \; %s \; band$' % band[ii])
            bx.errorbar(Ar, SB, yerr = err, xerr = None, color = mpl.cm.rainbow(tt/len(stackn)), marker = 'o', linewidth = 1, markersize = 10, 
                ecolor = mpl.cm.rainbow(tt/len(stackn)), elinewidth = 1, alpha = 0.5, label = '$SB_{stack%.0f}$' % stackn[tt])
            bx.errorbar(Ar0, SB0, yerr = err0, xerr = None, color = 'green', marker = 'o', linewidth = 1, markersize = 10, 
                ecolor = 'green', elinewidth = 1, alpha = 0.5, label = '$SB_{stack%.0f}$' % stackN)
            bx.set_xscale('log')
            bx.set_xlabel('$R[arcsec]$')
            bx.set_ylabel('$SB[mag/arcsec^2]$')
            bx.tick_params(axis = 'both', which = 'both', direction = 'in')
            #handles, labels = plt.gca().get_legend_handles_labels()
            bx1 = bx.twiny()
            bx1.plot(R, SB, ls = '-', color = mpl.cm.rainbow(tt/len(stackn)), alpha = 0.5)
            bx1.set_xscale('log')
            bx1.set_xlabel('$R[kpc]$')
            bx1.tick_params(axis = 'x', which = 'both', direction = 'in')
            bx.invert_yaxis()
            bx.legend( loc = 1)

            subax = fig.add_axes([0.55, 0.5, 0.25, 0.25])
            subax.hist(richb, bins = 10, histtype = 'step', color = 'blue', label = '$\lambda_{%d}$' % stackn[tt], alpha = 0.5)
            subax.axvline(m_rich, color = 'b', alpha = 0.5, label = '$\bar{\lambda_{%d}}$' % stackn[tt])
            subax.hist(richa, bins = 10, histtype = 'step', color = 'red', label = '$\lambda_{%d}$' % stackN, alpha = 0.5)
            subax.axvline(Mean_rich, color = 'r', alpha = 0.5, label = '$\bar{\lambda_{%d}}$' % stackN)

            subax.set_title('$\lambda \; distributtion$')
            subax.tick_params(axis = 'x', which = 'both', direction = 'in')
            subax.set_xlabel('$\lambda$')
            subax.set_ylabel('$N$')

            cx.plot(Ar0, SB - SB0, 'r-', label = '$\Delta_{SB}$')
            cx.set_xlabel('$R[arcsec]$')
            cx.set_xscale('log')
            cx.set_ylabel('$\Delta_{SB}$')
            cx.legend(loc = 1)
            subax.tick_params(axis = 'x', which = 'both', direction = 'in')

            plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_test_%s_band.png' % (stackn[tt], band[ii]), dpi = 600)
            plt.close()

if __name__ == "__main__":
    main()
