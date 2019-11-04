import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc
import matplotlib.gridspec as gridspec

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy.optimize import curve_fit, minimize
from ICL_surface_mass_density import sigma_m_c
from light_measure_tmp import light_measure, flux_recal
from resample_modelu import down_samp, sum_samp

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time

## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Angu_ref = rad2asec / Da_ref
Rpp = Angu_ref / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2
Lstar = 2e10 # in unit L_sun ("copy from Galaxies in the Universe", use for Luminosity function calculation)

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'

band = ['r', 'g', 'i', 'u', 'z']
sky_SB = [21.04, 22.01, 20.36, 22.30, 19.18]
gain = np.array([ [4.71, 4.6, 4.72, 4.76, 4.725, 4.895], 
				[3.32, 3.855, 3.845, 3.995, 4.05, 4.035], 
				[5.165, 6.565, 4.86, 4.885, 4.64, 4.76], 
				[1.62, 1.71, 1.59, 1.6, 1.47, 2.17], 
				[4.745, 5.155, 4.885, 4.775, 3.48, 4.69] ])

def SB_fit(r, mu_e0, mu_e1, mu_e2, r_e0, r_e1, r_e2, ndex0, ndex1, ndex2):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	mock_SB0 = mu_e0 + 1.086 * (2 * ndex0 - 0.324) * ( (r / r_e0)**(1 / ndex0) - 1) # in unit mag/arcsec^2
	mock_SB1 = mu_e1 + 1.086 * (2 * ndex1 - 0.324) * ( (r / r_e1)**(1 / ndex1) - 1)
	mock_SB2 = mu_e2 + 1.086 * (2 * ndex2 - 0.324) * ( (r / r_e2)**(1 / ndex2) - 1)
	f_SB0 = 10**( (22.5 - mock_SB0 + 2.5 * np.log10(pixel**2)) / 2.5 ) # in unit nmaggy
	f_SB1 = 10**( (22.5 - mock_SB1 + 2.5 * np.log10(pixel**2)) / 2.5 )
	f_SB2 = 10**( (22.5 - mock_SB2 + 2.5 * np.log10(pixel**2)) / 2.5 )
	mock_SB = 22.5 - 2.5 * np.log10(f_SB0 + f_SB1 + f_SB2) + 2.5 * np.log10(pixel**2)
	return mock_SB

def SB_dit(r, mu_e0, mu_e1, mu_e2, r_e0, r_e1, r_e2, ndex0, ndex1, ndex2):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	mock_SB0 = mu_e0 + 1.086 * (2 * ndex0 - 0.324) * ( (r / r_e0)**(1 / ndex0) - 1) # in unit mag/arcsec^2
	mock_SB1 = mu_e1 + 1.086 * (2 * ndex1 - 0.324) * ( (r / r_e1)**(1 / ndex1) - 1)
	mock_SB2 = mu_e2 + 1.086 * (2 * ndex2 - 0.324) * ( (r / r_e2)**(1 / ndex2) - 1)
	f_SB0 = 10**( (22.5 - mock_SB0 + 2.5 * np.log10(pixel**2)) / 2.5 ) # in unit nmaggy
	f_SB1 = 10**( (22.5 - mock_SB1 + 2.5 * np.log10(pixel**2)) / 2.5 )
	f_SB2 = 10**( (22.5 - mock_SB2 + 2.5 * np.log10(pixel**2)) / 2.5 )
	mock_SB = 22.5 - 2.5 * np.log10(f_SB0 + f_SB1 + f_SB2) + 2.5 * np.log10(pixel**2)
	return mock_SB0, mock_SB1, mock_SB2, mock_SB

def SB_pro():
	for kk in range(1):
		## read the profile
		SB0 = pds.read_csv(load + 'Zibetti_SB/%s_band_1.csv' % band[kk], skiprows = 1)
		R_r = SB0['Mpc']
		SB_r0 = SB0['mag/arcsec^2']
		SB1 = pds.read_csv(load + 'Zibetti_SB/%s_band_2.csv' % band[kk], skiprows = 1)
		SB_r1 = SB1['mag/arcsec^2']
		SB2 = pds.read_csv(load + 'Zibetti_SB/%s_band_3.csv' % band[kk], skiprows = 1)
		SB_r2 = SB2['mag/arcsec^2']
		## fit the profile
		mu_e0, mu_e1, mu_e2 = 23.87, 30, 20 # mag/arcsec^2
		Re_0, Re_1, Re_2 = 19.29, 120, 10 # kpc
		ndex0, ndex1, ndex2 = 4., 4., 4.

		r_fit = R_r * 10**3
		po = np.array([mu_e0, mu_e1, mu_e2, Re_0, Re_1, Re_2, ndex0, ndex1, ndex2])
		popt, pcov = curve_fit(SB_fit, r_fit, SB_r0, p0 = po, 
				bounds = ([21, 27, 18, 18, 100, 9, 1., 1., 1.], [24, 32, 21, 22, 500, 18, 6., 12., 4.]), method = 'trf')
		mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2 = popt
		r = np.logspace(0, 3.08, 1000)
		r_sc = r / 10**3
		r_max = np.max(r_sc)
		r_min = np.min(r_sc)
		SB_r = SB_fit(r, mu_fit0, mu_fit1, mu_fit2, re_fit0, re_fit1, re_fit2, ndex_fit0, ndex_fit1, ndex_fit2) # profile at z = 0.25

		key0 = ['%.3f' % z_ref, 'r']
		value0 = [SB_r, r]
		fill0 = dict(zip(key0, value0))
		data = pds.DataFrame(fill0)
		data.to_csv(load + 'mock_ccd/mock_intrinsic_SB_%s_band.csv' % band[kk])

def mock_ccd(band_id, z_set, ra_set, dec_set):
	exp_time = 54 # exposure time, in unit second
	kk = np.int(band_id)
	Nz = len(z_set)
	SB_bl = sky_SB[kk]

	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	Nx0, Ny0 = len(x0), len(y0)
	pxl = np.meshgrid(x0, y0)

	ins_SB = pds.read_csv(load + 'mock_ccd/mock_intrinsic_SB_%s_band.csv' % band[kk])
	r, INS_SB = ins_SB['r'], ins_SB['0.250']
	r_sc = r / 10**3
	r_max = np.max(r_sc)
	r_min = np.min(r_sc)
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		Angu_r = rad2asec / Da_g
		R_pixel = Angu_r / pixel

		## read the mask img for given redshift
		data = fits.getdata(load + 
			'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[kk], ra_g, dec_g, z_g), header = True)
		img = data[0]
		NMGY = data[1]['NMGY']
		cx0 = data[1]['CRPIX1']
		cy0 = data[1]['CRPIX2']
		RA0 = data[1]['CRVAL1']
		DEC0 = data[1]['CRVAL2']
		wcs = awc.WCS(data[1])
		xc, yc = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		xc, yc = np.int(xc), np.int(yc)

		id_nan = np.isnan(img) # the masked pix
		N_sky = 10**( (22.5 - SB_bl + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY
		dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
		dr_sc = dr / R_pixel

		## mock CCD
		ttx = INS_SB - 10 * np.log10( (1 + z_ref) / (1 + z_g) )
		sub_SB = ttx * 1
		sub_DN = 10**( (22.5 - ttx + 2.5*np.log10(pixel**2) ) / 2.5 ) / NMGY
		DN_min = np.min(sub_DN)
		mock_DN = interp.interp1d(r_sc, sub_DN, kind = 'cubic')

		## save the SB profile
		key0 = ['%.3f' % z_g, 'r']
		value0 = [sub_SB, r]
		fill0 = dict(zip(key0, value0))
		array = pds.DataFrame(fill0)
		array.to_csv(load + 'mock_ccd/mock_SB/SB_%s_ra%.3f_dec%.3f_Z%.3f.csv' % (band[kk], ra_g, dec_g, z_g) )

		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]
		Nois = np.zeros((Ny0, Nx0), dtype = np.float)

		ref_data = fits.open(d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g) )
		CAMCOL = ref_data[3].data['CAMCOL'][0]
		camcol = np.int(CAMCOL - 1)
		Gain = gain[kk, camcol]

		for pp in range(Ny0):
			for qq in range(Nx0):
				if (dr_sc[pp, qq] >= r_max ) | ( dr_sc[pp, qq] <= r_min ):

					lam_x = DN_min * Gain / 10
					N_e = lam_x + N_sky * Gain
					rand_x = np.random.poisson( N_e )
					Nois[pp, qq] += rand_x # electrons number

				else:

					lam_x = mock_DN( dr_sc[pp, qq] ) * Gain
					N_e = lam_x + N_sky * Gain
					rand_x = np.random.poisson( N_e )
					Nois[pp, qq] += rand_x # electrons number

		N_sub = Nois / Gain - N_sky
		N_sub[id_nan] = np.nan

		## save the mock img
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, Nx0, Ny0, cx0, cy0, xc, yc, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict( zip(keys, value) )
		fil = fits.Header(ff)
		fits.writeto(load + 
			'mock_ccd/mock_frame/mock-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g), N_sub, header = fil, overwrite=True)

def mock_resamp(band_id, sub_z, sub_ra, sub_dec):
	kk = np.int(band_id)
	zn = len(sub_z)
	#for ii in range(len(band)):
	print('Now band is %s' % band[kk])
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data = fits.open( load + 'mock_ccd/mock_frame/mock-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx0, cy0 = data[0].header['CRPIX1'], data[0].header['CRPIX2']
		RA0, DEC0 = data[0].header['CRVAL1'], data[0].header['CRVAL2']
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		ref_d = fits.open( d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g) )
		NMGY = ref_d[0].header['NMGY']
		img = img * NMGY # so the resample image have change DN to nmaggy

		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		b = L_ref / L_z0

		f_goal = flux_recal(img, z_g, z_ref) # scale all mock to z_ref
		ix0 = np.int(cx0 / b)
		iy0 = np.int(cy0 / b)
		if b > 1:
			resam, xn, yn = sum_samp(b, b, f_goal, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, f_goal, cx, cy)
		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 
			'mock_ccd/mock_resamp/mock_resam-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)

def mock_stack(band_id, sub_z, sub_ra, sub_dec):
	stack_N = len(sub_z)
	kk = np.int(band_id)
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		data = fits.open(load + 'mock_ccd/mock_resamp/mock_resam-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g))
		img = data[0].data
		xn = data[0].header['CENTER_X']
		yn = data[0].header['CENTER_Y']
		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array[la0:la1, lb0:lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1.
		count_array[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(load + 'test_h5/mock_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)

	with h5py.File(load + 'test_h5/mock_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

def main():
	Nz = len(z)
	#SB_pro()
	'''
	## mock ccd
	for tt in range(1):
		m, n = divmod(Nz, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		mock_ccd(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		print('%s band finished !' % band[tt])
	print('finished !!!')

	## resample mock frame
	for tt in range(1):
		m, n = divmod(Nz, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		mock_resamp(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		print('%s band finished !' % band[tt])
	print('finished !!!')
	'''
	N_tt = np.array([50, 100, 200, 500, 1000, 3000])
	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	popu = np.linspace(0, len(z) - 1, len(z))
	popu = popu.astype( int )
	popu = set(popu)

	for aa in range( len(N_tt) ):
		tt0 = random.sample(popu, N_tt[aa])
		set_z = z[tt0]
		set_ra = ra[tt0]
		set_dec = dec[tt0]

		## stack mock frame
		for tt in range(1):
			m, n = divmod(N_tt[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			mock_stack(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

		# stack eht sub-sum array
		mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		if rank == 0:
			for qq in range(1):
				## SB_ref
				ins_SB = pds.read_csv(load + 'mock_ccd/mock_intrinsic_SB_%s_band.csv' % band[qq])
				r, INS_SB = ins_SB['r'], ins_SB['0.250']
				f_SB = interp.interp1d(r, INS_SB, kind = 'cubic')

				tot_N = 0
				for pp in range(cpus):

					with h5py.File(load + 'test_h5/mock_sum_%d_in_%s_band.h5' % (pp, band[qq]), 'r')as f:
						sum_img = np.array(f['a'])
					with h5py.File(load + 'test_h5/mock_pcount_%d_in_%s_band.h5' % (pp, band[qq]), 'r') as f:
						p_count = np.array(f['a'])

					sub_Num = np.nanmax(p_count)
					tot_N += sub_Num
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				tot_N = np.int(tot_N)

				stack_img = mean_img / p_add_count
				R_cut = 800
				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, 10, Rpp, R_cut, R_cut, pixel, z_ref)
				flux0 = Intns + Intns_err
				flux1 = Intns - Intns_err
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
				SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
				err0 = SB - SB0
				err1 = SB1 - SB
				id_nan = np.isnan(SB)
				SB, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
				pR, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
				id_nan = np.isnan(SB1)
				err1[id_nan] = 100. # set a large value for show the break out errorbar

				iux = ( pR > np.min(r) ) & ( pR < np.max(r) )
				ddsb = SB[iux] - f_SB( pR[iux] )
				ddsr = pR[iux]
				std = np.nanstd(ddsb)
				aver = np.nanmean(ddsb)

				plt.figure()
				ax = plt.subplot(111)
				ax.set_title('stack mock [%d img %s band]' % (tot_N, band[qq]) )		
				tf = ax.imshow(stack_img, origin = 'lower', vmin = 1e-2, vmax = 1e1, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = ax, fraction = 0.035, pad =  0.01, label = '$ flux[nmaggy] $')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-',)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--',)	
				plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
				plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
				plt.savefig(load + 'mock_ccd/mock_stack_%d_%s_band.png' % (tot_N, band[qq]), dpi = 300)
				plt.close()

				fig = plt.figure()
				fig.suptitle('stack img SB [%d img %s band]' % (tot_N, band[qq]))
				gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
				ax = plt.subplot(gs[0])
				bx = plt.subplot(gs[1])

				#ax.set_title('stack img SB [%d img %s band]' % (tot_N, band[qq]))
				ax.errorbar(pR, SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'ro', label = 'Mock [noise + mask]', alpha = 0.5)					
				ax.plot(r, INS_SB, 'b-', label = ' intrinsic SB ', alpha = 0.5)
				ax.set_xscale('log')
				ax.set_xlabel('R [kpc]')
				ax.set_xlim(9, 1010)
				ax.set_ylim(19, 34)
				ax.invert_yaxis()
				ax.set_ylabel('$ SB[mag / arcsec^2] $')
				ax.legend(loc = 1)
				ax.tick_params(axis = 'both', which = 'both', direction = 'in')

				bx1 = ax.twiny()
				xtik = ax.get_xticks()
				xtik = np.array(xtik)
				xR = xtik * 10**(-3) * rad2asec / Da_ref
				bx1.set_xscale('log')
				bx1.set_xticks(xtik)
				bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
				bx1.set_xlim(ax.get_xlim())
				bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
				ax.set_xticks([])

				bx.axhline(y = 0, linestyle = '-.', color = 'k', alpha = 0.5)
				bx.plot(ddsr, ddsb, 'b--', alpha = 0.5)
				bx.axhline(y = aver, linestyle = ':', color = 'b', alpha = 0.5)
				bx.axhline(y = aver + std, linestyle = '--', color = 'b', alpha = 0.5)
				bx.axhline(y = aver - std, linestyle = '--', color = 'b', alpha = 0.5)

				bx.set_xscale('log')
				bx.set_xlim(9, 1010)
				bx.set_ylim(aver - 1.2 * std, aver + 1.2 * std)
				bx.set_xlabel('$R[kpc]$')
				bx.set_ylabel('$ SB_{stacking} - SB_{reference} $')
				bx.tick_params(axis = 'both', which = 'both', direction = 'in')

				plt.subplots_adjust(hspace = 0)
				plt.savefig(load + 'mock_ccd/mock_stack_SB_%d_%s_band.png' % (tot_N, band[qq]), dpi = 300)
				plt.close()

		commd.Barrier()

	raise

if __name__ == "__main__":
	main()
