import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from groups import groups_find_func

def cat_combine( cat_lis, ra, dec, z, alt_G_size = None,):

	Ns = len( cat_lis )

	tot_cx, tot_cy = np.array([]), np.array([])
	tot_a, tot_b = np.array([]), np.array([])
	tot_theta = np.array([])
	tot_Numb = 0

	for ll in range( Ns ):

		ext_cat = cat_lis[ll] % ( ra, dec, z )
		try:
			source = asc.read(ext_cat)
			Numb = np.array(source['NUMBER'][-1])
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1

			if alt_G_size is not None:
				Kron = alt_G_size + 0.
			else:
				Kron = 16

			a = Kron * A
			b = Kron * B

			tot_cx = np.r_[tot_cx, cx]
			tot_cy = np.r_[tot_cy, cy]
			tot_a = np.r_[tot_a, a]
			tot_b = np.r_[tot_b, b]
			tot_theta = np.r_[tot_theta, theta]
			tot_Numb = tot_Numb + Numb

		except:
			continue

	return tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta

def adjust_mask_func(d_file, cat_file, z_set, ra_set, dec_set, band, gal_file, out_file, bcg_mask, extra_cat = None, alter_fac = None, 
	alt_bright_R = None, alt_G_size = None, stack_info = None, pixel = 0.396,):
	"""
	after img masking, use this function to detection "light" region, which
	mainly due to nearby brightstars, for SDSS case: taking the brightness of
	img center region as a normal brightness (mu, with scatter sigma), and rule out all the sub-patches
	whose mean pixel flux is larger than mu + 3.5 * sigma
	------------
	d_file : path where image data saved (include file-name structure:
	'/xxx/xxx/xxx.xxx')
	cat_file : path where photometric data saved, the same structure as d_file
	gal_file : the source catalog based on SExTractor calculation
	set_ra, set_dec, set_z : ra, dec, z of will be masked imgs
	band: band of image data, 'str' type
	out_file : save the masking data
	bcg_mask : 0 : keep BCGs; 1 : BCGs will be masked
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	stack_info : path to save the information of stacking (ra, dec, z, img_x, img_y)
	including file-name: '/xxx/xxx/xxx.xxx'

	extra_cat : extral galaxy catalog for masking adjust, (list type, .cat files)
	alter_fac : size adjust for normal stars
	alt_bright_R : size adjust for bright stars (also for saturated sources)
	alt_G_size : size adjust for galaxy-like sources
	"""
	Nz = len(z_set)
	bcg_x, bcg_y = [], []

	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head = data[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		bcg_x.append(xn)
		bcg_y.append(yn)

		source = asc.read(gal_file % (band, ra_g, dec_g, z_g), )
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1
		p_type = np.array(source['CLASS_STAR'])

		if alt_G_size is not None:
			Kron = alt_G_size + 0.
		else:
			Kron = 16
		a = Kron * A
		b = Kron * B

		## extral catalog load
		if extra_cat is not None:

			Ecat_num, Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = cat_combine( extra_cat, ra_g, dec_g, z_g, alt_G_size = None,)

		else:
			Ecat_num = 0
			Ecat_x, Ecat_y, Ecat_a, Ecat_b, Ecat_chi = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

		## stars
		mask = cat_file % (z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		flags = [str(qq) for qq in xt]

		x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)

		set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
		set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
		set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

		lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
		lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
		sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

		# normal stars
		iq = lln >= 2 ## at lest observed in 2 band
		ig = OBJ == 6
		ie = (set_mag <= 20)

		ic = (ie & ig & iq)
		sub_x0 = x[ic]
		sub_y0 = y[ic]
		if alter_fac is not None:
			sub_A0 = lr_iso[ic] * alter_fac
			sub_B0 = sr_iso[ic] * alter_fac
		else:
			sub_A0 = lr_iso[ic] * 30
			sub_B0 = sr_iso[ic] * 30
		sub_chi0 = set_chi[ic]

		# saturated source(may not stars)
		xa = ['SATURATED' in qq for qq in flags]
		xv = np.array(xa)
		idx = xv == True
		ipx = (idx)

		sub_x2 = x[ipx]
		sub_y2 = y[ipx]

		if alt_bright_R is not None:
			sub_A2 = lr_iso[ipx] * alt_bright_R
			sub_B2 = sr_iso[ipx] * alt_bright_R			
		else:
			sub_A2 = lr_iso[ipx] * 75
			sub_B2 = sr_iso[ipx] * 75
		sub_chi2 = set_chi[ipx]

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0] ]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0] ]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0] ]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0] ]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0] ]

		tot_cx = np.r_[cx, comx, Ecat_x]
		tot_cy = np.r_[cy, comy, Ecat_y]
		tot_a = np.r_[a, Lr, Ecat_a]
		tot_b = np.r_[b, Sr, Ecat_b]
		tot_theta = np.r_[theta, phi, Ecat_chi]
		tot_Numb = Numb + len(comx) + Ecat_num

		mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
		ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
		oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
		basic_coord = np.array(np.meshgrid(ox, oy))
		major = tot_a / 2
		minor = tot_b / 2
		senior = np.sqrt(major**2 - minor**2)

		for k in range(tot_Numb):
			xc = tot_cx[k]
			yc = tot_cy[k]

			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = tot_theta[k] * np.pi/180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

		mask_img = mask_path * img

		### add BCG region back
		if bcg_mask == 0:

			copy_mask = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
			tdr = np.sqrt((xn - cx)**2 + (yn - cy)**2)
			idx = tdr == np.min(tdr)
			id_bcg = np.where(idx)[0][0]

			for k in range( Numb ):
				xc = cx[k]
				yc = cy[k]

				lr = A[k] * 5 #3
				sr = B[k] * 5 #3

				chi = theta[k] * np.pi / 180

				if k == id_bcg:
					continue
				else:
					set_r = np.int(np.ceil(1.0 * lr))
					la0 = np.max( [np.int(xc - set_r), 0])
					la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
					lb0 = np.max( [np.int(yc - set_r), 0] ) 
					lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

					df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
					df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
					fr = df1**2 / lr**2 + df2**2 / sr**2
					jx = fr <= 1

					iu = np.where(jx == True)
					iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
					iv[iu] = np.nan
					copy_mask[lb0: lb1, la0: la1] = copy_mask[lb0: lb1, la0: la1] * iv

			copy_imgs = copy_mask * img

			tdr = np.sqrt((xn - cx)**2 + (yn - cy)**2)
			idx = tdr == np.min(tdr)
			lr = A[idx] * 8
			sr = B[idx] * 8

			targ_x = cx[idx]
			targ_y = cy[idx]
			targ_chi = theta[idx] * np.pi / 180

			set_r = np.int(np.ceil(1.0 * lr))
			la0 = np.max( [np.int(targ_x - set_r), 0])
			la1 = np.min( [np.int(targ_x + set_r +1), img.shape[1] ] )
			lb0 = np.max( [np.int(targ_y - set_r), 0] )
			lb1 = np.min( [np.int(targ_y + set_r +1), img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - targ_x)* np.cos(targ_chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - targ_y)* np.sin(targ_chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - targ_y)* np.cos(targ_chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - targ_x)* np.sin(targ_chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == False)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan

			dpt_img = copy_imgs[lb0: lb1, la0: la1]
			dpt_img = dpt_img * iv

			#mask_img[lb0: lb1, la0: la1] = dpt_img
			mask_img[lb0: lb1, la0: la1] = copy_imgs[lb0: lb1, la0: la1]

		hdu = fits.PrimaryHDU()
		hdu.data = mask_img
		hdu.header = head
		hdu.writeto(out_file % (band, ra_g, dec_g, z_g), overwrite = True)

	bcg_x = np.array(bcg_x)
	bcg_y = np.array(bcg_y)

	if stack_info != None:
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ra_set, dec_set, z_set, bcg_x, bcg_y]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(stack_info)

	return

def main():

	from mpi4py import MPI
	commd = MPI.COMM_WORLD
	rank = commd.Get_rank()
	cpus = commd.Get_size()

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/data/SDSS/'

	### test-1000 sample (r band)
	dat = pds.read_csv('/home/xkchen/fig_tmp/test_1000_no_select.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	cat_file = home + 'corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'
	gal_file = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	extra_cat = home + 'source_detect_cat/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv'

	bcg_mask = 1
	band = 'r'
	'''
	### masking test for normal stars.
	size_arr = np.array([10, 20])

	for mm in range( 2 ):

		out_file = '/home/xkchen/fig_tmp/norm_mask/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

		adjust_mask_func(d_file, cat_file, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], 
			band, gal_file, out_file, bcg_mask, extra_cat, size_arr[mm],)
	'''
	### masking test for bright stars
	size_arr = np.array([25, 50])

	for mm in range( 2 ):

		out_file = '/home/xkchen/fig_tmp/bright_mask/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2_bri-star.fits' % (size_arr[mm])

		adjust_mask_func(d_file, cat_file, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], 
			band, gal_file, out_file, bcg_mask, extra_cat, alter_fac = None, alt_bright_R = size_arr[mm],)

	raise

if __name__ == "__main__":
	main()

