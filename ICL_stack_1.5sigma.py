import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from scipy.interpolate import interp1d as interp
from scipy.optimize import curve_fit

from resamp import gen
from extinction_redden import A_wave
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
    bins = 45
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)
    sum_grid = np.array(np.meshgrid(Nx, Ny))

    sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # sum the flux value for each time
    count_array_0 = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan # sum array but use for pixel count for each time
    p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # how many times of each pixel get value

    for jj in range(stack_N):

        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]

        Da_g = Test_model.angular_diameter_distance(z_g).value
        data = fits.getdata(load + 
            'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
        img = data[0]
        xn = data_tt[1]['CENTER_X']
        yn = data_tt[1]['CENTER_Y']

        la0 = np.int(y0 - yn)
        la1 = np.int(y0 - yn + img.shape[0])
        lb0 = np.int(x0 - xn)
        lb1 = np.int(x0 - xn + img.shape[1])

        idx = np.isnan(img)
        idv = np.where(idx == False)
        sum_array_0[la0:la1, lb0:lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + img[idv]
        count_array_0[la0: la1, lb0: lb1][idv] = img[idv]
        id_nan = np.isnan(count_array_0)
        id_fals = np.where(id_nan == False)
        p_count_0[id_fals] = p_count_0[id_fals] + 1
        count_array_0[la0: la1, lb0: lb1][idv] = np.nan

    mean_array_0 = sum_array_0 / p_count_0
    where_are_inf = np.isinf(mean_array_0)
    mean_array_0[where_are_inf] = np.nan
    id_zeros = np.where(p_count_0 == 0)
    mean_array_0[id_zeros] = np.nan

    SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)
    SB_0 = SB[1:] + mag_add[ii]
    R_0 = R[1:]
    Ar_0 = Ar[1:]
    err_0 = error[1:]

    ix = R_0 >= 100
    iy = R_0 <= 900
    iz = ix & iy
    r_fit = R_0[iz]
    sb_fit = SB_0[iz]

    m0 = 30.5
    mc = 14.5
    cc = 5
    m2l = 237
    po = np.array([m0, mc, cc, m2l])
    popt, pcov = curve_fit(SB_fit, r_fit, sb_fit, p0 = po, method = 'trf')

    M0 = popt[0]
    Mc = popt[1]
    Cc = popt[2]
    M2L = popt[3]
    fit_l = SB_fit(r_fit, M0, Mc, Cc, M2L)
    f_SB = interp(Ar_0[iz], fit_l)

    plt.figure(figsize = (16,9))
    bx = plt.subplot(111)
    bx.set_title('$stack \; light \; test \; in \; %s \; band$' % band[ii])
    bx.errorbar(Ar_0, SB_0, yerr = err_0, xerr = None, color = 'r', marker = 'o', ls = '', linewidth = 1, markersize = 5, 
        ecolor = 'r', elinewidth = 1, alpha = 0.5, label = '$stack%.0f$' % stack_N)
    bx.set_xscale('log')
    bx.set_xlabel('$R[arcsec]$')
    bx.set_ylabel('$SB[mag/arcsec^2]$')
    bx.tick_params(axis = 'both', which = 'both', direction = 'in')

    bx1 = bx.twiny()
    bx1.plot(R_0, SB_0, ls = '-', color = 'r', alpha = 0.5)
    bx1.set_xscale('log')
    bx1.set_xlabel('$R[kpc]$')
    bx1.tick_params(axis = 'x', which = 'both', direction = 'in')

    bx.invert_yaxis()
    bx.legend( loc = 1, fontsize = 12)
    plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/light_errorbar_%s_band.png' % band[ii], dpi = 600)
    plt.close()

    plt.figure(figsize = (16, 9))
    ax = plt.subplot(111)
    ax.set_title('stack %d image with background subtraction' % stack_N)
    ax.plot(Ar_0[iz], SB_0[iz], yerr = err_0, xerr = None, color = 'b', marker = 'o', ls = '', linewidth = 1, markersize = 5, 
        ecolor = 'b', elinewidth = 1, alpha = 0.5, label = '$SB_{stack%.0f}$' % stack_N)
    ax.plot(Ar_0[iz], fit_l, 'r-', label = '$NFW + C$', alpha = 0.5)

    ax.set_xscale('log')
    ax.set_xlabel('$R[arcsec]$')
    ax.set_ylabel('$SB[mag/arcsec^2]$')
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    ax.invert_yaxis()
    ax.legend(loc = 1, fontsize = 12)
    ax.set_title('stacked SB with resample correct')

    ax1 = ax.twiny()
    xtik = ax.get_xticks(minor = True)
    xR = xtik * 10**(-3) * rad2asec / Da_ref
    ax1.set_xticks(xtik)
    ax1.set_xticklabels(["%.2f" % uu for uu in xR])
    ax1.set_xlim(ax.get_xlim())
    ax1.set_xlabel('$R[arcsec]$')
    ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

    ax.text(1.5e2, 29, s = 'BL = %.2f' % M0 + '\n' + '$Mc = %.2fM_\odot $' % Mc + '\n' 
        + 'C = %.2f' % Cc + '\n' + 'M2L = %.2f' % M2L, fontsize = 15)
    plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/light_test_%s_band.png' % band[ii], dpi = 600)
    plt.close()

    return 

def main():

    ix = (red_rich >= 25) & (red_rich <= 29)
    RichN = red_rich[ix]
    zN = red_z[ix]
    raN = red_ra[ix]
    decN = red_dec[ix]
    #stackn = len(zN)
    stackn = 200

    for ii in range(len(band)):
        id_band = ii

        stack_light(id_band, stackn, zN, raN, decN)
        print(ii)

if __name__ == "__main__":
    main()
