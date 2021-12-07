import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pds
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

###
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396

url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

def BCG_mag_sql(cat_z, cat_ra, cat_dec, cat_ID, out_file):

	Nz = len(cat_z)

	for kk in range(Nz):
		z_g = cat_z[kk]
		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			G.objID, 
			G.deVAB_u, G.deVAB_g, G.deVAB_r, G.deVAB_i, G.deVAB_z,
			G.deVPhi_u, G.deVPhi_g, G.deVPhi_r, G.deVPhi_i, G.deVPhi_z,
			G.deVRad_u, G.deVRad_g, G.deVRad_r, G.deVRad_i, G.deVRad_z,
			G.expAB_u, G.expAB_g, G.expAB_r, G.expAB_i, G.expAB_z,
			G.expPhi_u, G.expPhi_g, G.expPhi_r, G.expPhi_i, G.expPhi_z,
			G.expRad_u, G.expRad_g, G.expRad_r, G.expRad_i,G.expRad_z

		FROM Galaxy AS G
		WHERE
			G.objID = %d
		""" % cat_ID[kk]

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()

		## PS : effective radius, in unit of arcsec
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( out_file % (z_g, ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

	return

def BCG_pro_sql(cat_z, cat_ra, cat_dec, cat_ID, out_file):
	Nz = len(cat_z)
	for kk in range(Nz):
		z_g = cat_z[kk]
		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			pro.objID, pro.bin, pro.band, pro.profMean, pro.profErr
		FROM PhotoProfile AS pro
		WHERE
			pro.objID = %d
			AND pro.bin BETWEEN 0 AND 15
		""" % cat_ID[kk]

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()
		#print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')

		doc = open( out_file % (z_g, ra_g, dec_g),)
		print(s, file = doc)
		doc.close()

	return

### star sql
r_select = 0.42 ## 1.5 * diagonal line length

def sdss_sql_star(z_set, ra_set, dec_set, ref_ra, ref_dec, out_file,):

	Nz = len(z_set)

	for q in range(Nz):

		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		cen_ra = ref_ra[q]
		cen_dec = ref_dec[q]

		c_ra0 = cen_ra - r_select
		c_dec0 = cen_dec - r_select
		c_ra1 = cen_ra + r_select
		c_dec1 = cen_dec + r_select

		if cen_ra + r_select > 360:

			l_ra_0 = cen_ra
			l_ra_1 = 0

			r_ra_0 = 360
			r_ra_1 = r_select - (360 - cen_ra)

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
			ORDER by p.r
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

		elif cen_ra - r_select < 0:

			l_ra_0 = 0
			l_ra_1 = 360 + (cen_ra - r_select)

			r_ra_0 = cen_ra
			r_ra_1 = 360

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
			ORDER by p.r
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

		else:

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				p.ra BETWEEN %.5f AND %.5f
				AND p.dec BETWEEN %.5f AND %.5f
				AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
			ORDER by p.r
			""" % (c_ra0, c_ra1, c_dec0, c_dec1)

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()

		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8',)
		doc = open( out_file % (z_g, ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

	return

# query with sqlcl.py
def put_sql(z_set, ra_set, dec_set, ref_ra, ref_dec, out_file,):

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	import sqlcl
	from sql_ask import sdss_sql

	fmt = 'csv'

	Nz = len(z_set)

	for q in range(Nz):

		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		cen_ra = ref_ra[q]
		cen_dec = ref_dec[q]

		c_ra0 = cen_ra - r_select
		c_dec0 = cen_dec - r_select
		c_ra1 = cen_ra + r_select
		c_dec1 = cen_dec + r_select

		if cen_ra + r_select > 360:

			l_ra_0 = cen_ra
			l_ra_1 = 0

			r_ra_0 = 360
			r_ra_1 = r_select - (360 - cen_ra)

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
			ORDER by p.r
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

		elif cen_ra - r_select < 0:

			l_ra_0 = 0
			l_ra_1 = 360 + (cen_ra - r_select)

			r_ra_0 = cen_ra
			r_ra_1 = 360

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
			ORDER by p.r
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

		else:

			data_set = """
			SELECT ALL
				p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
				p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
				p.flags, dbo.fPhotoFlagsN(p.flags)
			FROM PhotoObjAll AS p
			WHERE
				p.ra BETWEEN %.5f AND %.5f
				AND p.dec BETWEEN %.5f AND %.5f
				AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
			ORDER by p.r
			""" % (c_ra0, c_ra1, c_dec0, c_dec1)

		print_file = out_file % (z_g, ra_g, dec_g)

		sdss_sql(url, data_set, fmt, print_file, id_print = False,)

	return

load = '/home/xkchen/data/SDSS/photo_files/'
home = '/home/xkchen/data/SDSS/'

"""
dat = pds.read_csv(home + 'selection/redMapper_z-photo_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
ID = np.array(dat['objID'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

out_file_0 = load + 'BCG_photometry/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'
BCG_mag_sql( z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], ID[N_sub0 : N_sub1], out_file_0)

print('%d rank finished part-1' % rank)

out_file_1 = load + 'BCG_profile/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'
BCG_pro_sql( z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1], ID[N_sub0 : N_sub1], )

print('%d rank finished part-2' % rank)
"""

dat = pds.read_csv(home + 'selection/photo-z_sample_sql_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
ref_ra, ref_dec = np.array(dat['ref_ra']), np.array(dat['ref_dec'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

out_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

sdss_sql_star(z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], ref_ra[N_sub0:N_sub1], ref_dec[N_sub0:N_sub1], out_file,)
#put_sql(z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], ref_ra[N_sub0:N_sub1], ref_dec[N_sub0:N_sub1], out_file,)

print('%d rank finished part-3' % rank)

