import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as fits
import scipy.stats as sts

import h5py
import pandas as pds

def match_func(set_ra, set_dec, set_z, cat_file, out_file):

	goal_data = fits.getdata(cat_file)
	RA = np.array(goal_data.RA)
	DEC = np.array(goal_data.DEC)
	ID = np.array(goal_data.ID)
	redshift = np.array(goal_data.Z_SPEC)
	Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
	Mag_err = np.array(goal_data.MODEL_MAGERR_R)
	lamda = np.array(goal_data.LAMBDA)

	r_Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
	r_Mag_err = np.array(goal_data.MODEL_MAGERR_R)
	g_Mag_bcgs = np.array(goal_data.MODEL_MAG_G)
	g_Mag_err = np.array(goal_data.MODEL_MAGERR_G)
	i_Mag_bcgs = np.array(goal_data.MODEL_MAG_I)
	i_Mag_err = np.array(goal_data.MODEL_MAGERR_I)
	u_Mag_bcgs = np.array(goal_data.MODEL_MAG_U)
	u_Mag_err = np.array(goal_data.MODEL_MAGERR_U)
	z_Mag_bcgs = np.array(goal_data.MODEL_MAG_Z)
	z_Mag_err = np.array(goal_data.MODEL_MAGERR_Z)

	com_z = redshift[(redshift >= 0.2) & (redshift <= 0.3)]
	com_ra = RA[(redshift >= 0.2) & (redshift <= 0.3)]
	com_dec = DEC[(redshift >= 0.2) & (redshift <= 0.3)]
	com_rich = lamda[(redshift >= 0.2) & (redshift <= 0.3)]

	com_r_Mag = r_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
	com_r_Mag_err = r_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

	com_g_Mag = g_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
	com_g_Mag_err = g_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

	com_i_Mag = i_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
	com_i_Mag_err = i_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

	com_u_Mag = u_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
	com_u_Mag_err = u_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

	com_z_Mag = z_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
	com_z_Mag_err = z_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]
	com_ID = ID[(redshift >= 0.2) & (redshift <= 0.3)]

	##### list of the target cat
	targ_ra = ['%.3f' % ll for ll in set_ra]
	targ_dec = ['%.3f' % ll for ll in set_dec]
	targ_z = ['%.3f' % ll for ll in set_z]

	#### initial the list
	sub_z = []
	sub_ra = []
	sub_dec = []
	sub_rich = []
	sub_ID = []
	sub_r_mag, sub_r_Merr = [], []
	sub_g_mag, sub_g_Merr = [], []
	sub_i_mag, sub_i_Merr = [], []
	sub_u_mag, sub_u_Merr = [], []
	sub_z_mag, sub_z_Merr = [], []

	zN = len(com_z)

	for jj in range(zN):
		ra_g = com_ra[jj]
		dec_g = com_dec[jj]
		z_g = com_z[jj]
		rich_g = com_rich[jj]
		ID_g = com_ID[jj]

		r_mag, r_err = com_r_Mag[jj], com_r_Mag_err[jj]
		g_mag, g_err = com_g_Mag[jj], com_g_Mag_err[jj]
		i_mag, i_err = com_i_Mag[jj], com_i_Mag_err[jj]
		u_mag, u_err = com_u_Mag[jj], com_u_Mag_err[jj]
		z_mag, z_err = com_z_Mag[jj], com_z_Mag_err[jj]

		identi = ('%.3f' % ra_g in targ_ra) & ('%.3f' % dec_g in targ_dec) # & ('%.3f' % z_g in targ_z)

		if  identi == True: 
			sub_z.append(z_g)
			sub_ra.append(ra_g)
			sub_dec.append(dec_g)
			sub_rich.append(rich_g)
			sub_ID.append(ID_g)

			sub_r_mag.append(r_mag)
			sub_g_mag.append(g_mag)
			sub_i_mag.append(i_mag)
			sub_u_mag.append(u_mag)
			sub_z_mag.append(z_mag)

			sub_r_Merr.append(r_err)
			sub_g_Merr.append(g_err)
			sub_i_Merr.append(i_err)
			sub_u_Merr.append(u_err)
			sub_z_Merr.append(z_err)
		else:
			continue

	sub_z = np.array(sub_z)
	sub_ra = np.array(sub_ra)
	sub_dec = np.array(sub_dec)
	sub_rich = np.array(sub_rich)
	sub_ID = np.array(sub_ID)

	sub_r_mag = np.array(sub_r_mag)
	sub_g_mag = np.array(sub_g_mag)
	sub_i_mag = np.array(sub_i_mag)
	sub_u_mag = np.array(sub_u_mag)
	sub_z_mag = np.array(sub_z_mag)

	sub_r_Merr = np.array(sub_r_Merr)
	sub_g_Merr = np.array(sub_g_Merr)
	sub_i_Merr = np.array(sub_i_Merr)
	sub_u_Merr = np.array(sub_u_Merr)
	sub_z_Merr = np.array(sub_z_Merr)

	## save the csv file
	keys = ['ra', 'dec', 'z', 'rich', 'r_Mag', 'g_Mag', 'i_Mag', 'u_Mag', 'z_Mag', 'r_Mag_err', 'g_Mag_err', 
	'i_Mag_err', 'u_Mag_err', 'z_Mag_err', 'CAT_ID']
	values = [sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
	sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr, sub_ID]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(out_file)

	return

def random_match_func(set_ra, set_dec, set_z, cat_file, out_file):

	rand_data = fits.getdata(cat_file)
	RA = rand_data.RA
	DEC = rand_data.DEC
	Z = rand_data.Z
	LAMBDA = rand_data.LAMBDA

	idx = (Z >= 0.2) & (Z <= 0.3)
	# select the nearly universe
	z_eff = Z[idx]
	ra_eff = RA[idx]
	dec_eff = DEC[idx]
	lamda_eff = LAMBDA[idx]

	##### list of the target cat
	targ_ra = ['%.5f' % ll for ll in set_ra]
	targ_dec = ['%.5f' % ll for ll in set_dec]
	targ_z = ['%.5f' % ll for ll in set_z]

	zN = len(z_eff)

	sub_z = []
	sub_ra = []
	sub_dec = []
	sub_rich = []

	for jj in range(zN):

		ra_g = ra_eff[jj]
		dec_g = dec_eff[jj]
		z_g = z_eff[jj]
		rich_g = lamda_eff[jj]

		identi = ('%.5f' % ra_g in targ_ra) & ('%.5f' % dec_g in targ_dec) & ('%.5f' % z_g in targ_z)

		if  identi == True:
			sub_z.append(z_g)
			sub_ra.append(ra_g)
			sub_dec.append(dec_g)
			sub_rich.append(rich_g)
		else:
			continue
	sub_z = np.array(sub_z)
	sub_ra = np.array(sub_ra)
	sub_dec = np.array(sub_dec)
	sub_rich = np.array(sub_rich)

	## save the csv file
	keys = ['ra', 'dec', 'z', 'rich']
	values = [sub_ra, sub_dec, sub_z, sub_rich]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(out_file)

	return

def main():
	"""
	### random img
	cat_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_randoms.fits'
	out_file = '/home/xkchen/Downloads/test_imgs/tot_random_norm_sample_cat-match.csv'
	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/tot_random_norm_sample.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	random_match_func(ra, dec, z, cat_file, out_file)

	### cluster img
	cat_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
	out_file = '/home/xkchen/Downloads/test_imgs/tot_cluster_norm_sample_cat-match.csv'
	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/tot_cluster_norm_sample.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	match_func(ra, dec, z, cat_file, out_file)
	"""
	rnd_file = '/home/xkchen/Downloads/test_imgs/tot_random_norm_sample_cat-match.csv'
	clu_file = '/home/xkchen/Downloads/test_imgs/tot_cluster_norm_sample_cat-match.csv'

	dat_clus = pds.read_csv(clu_file)
	z, ra, dec, rich = np.array(dat_clus.z), np.array(dat_clus.ra), np.array(dat_clus.dec), np.array(dat_clus.rich)
	set_z, set_ra, set_dec, set_rich = z[2613: 2814], ra[2613: 2814], dec[2613: 2814], rich[2613: 2814]

	dat_rnd = pds.read_csv(rnd_file)
	rnd_z, rnd_ra, rnd_dec = np.array(dat_rnd.z), np.array(dat_rnd.ra), np.array(dat_rnd.dec)
	rnd_rich = np.array(dat_rnd.rich)

	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import scipy.stats as sts

	clus_cont, edg_rich, edg_z = sts.binned_statistic_2d(set_rich, set_z, set_z, statistic = 'count', bins = [5, 6],)[:3]
	clus_cont = clus_cont.astype(int)

	targ_z, targ_ra, targ_dec, targ_rich = np.array([0]), np.array([0]), np.array([0]), np.array([0])

	for kk in range( len(edg_rich) - 1 ):
		for ll in range( len(edg_z) - 1 ):
			if clus_cont[kk, ll] == 0:
				continue
			else:
				idy = (rnd_rich >= edg_rich[kk]) & (rnd_rich <= edg_rich[kk+1] )
				idx = (rnd_z >= edg_z[ll]) & (rnd_z <= edg_z[ll+1] )
				idv = idx & idy
				sub_z, sub_ra, sub_dec, sub_rich = rnd_z[idv], rnd_ra[idv], rnd_dec[idv], rnd_rich[idv]

				np.random.seed(1)
				tt0 = np.random.choice( len(sub_z), size = clus_cont[kk,ll], replace = False)

				targ_z = np.r_[targ_z, sub_z[tt0] ]
				targ_ra = np.r_[targ_ra, sub_ra[tt0] ]
				targ_dec = np.r_[targ_dec, sub_dec[tt0] ]
				targ_rich = np.r_[targ_rich, sub_rich[tt0] ]

	targ_z = targ_z[1:]
	targ_ra = targ_ra[1:]
	targ_dec = targ_dec[1:]
	targ_rich = targ_rich[1:]

	keys = ['ra', 'dec', 'z', 'rich']
	values = [targ_ra, targ_dec, targ_z, targ_rich]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv('random_sub-13-match_cat.csv')

	plt.figure()
	plt.plot(set_z, set_rich, 'bs', alpha = 0.5, label = 'cluster')
	plt.plot(targ_z, targ_rich, 'ro', alpha = 0.5, label = 'random')
	plt.xlabel('$ z $')
	plt.ylabel('$ \\lambda $')
	plt.legend(loc = 2, frameon = False)
	plt.savefig('compare.png', dpi = 300)
	plt.close()

	raise

if __name__ == "__main__":
	main()
