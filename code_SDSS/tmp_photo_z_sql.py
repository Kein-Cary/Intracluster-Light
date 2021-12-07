import h5py
import numpy as np
import astropy.io.fits as fits

#import mechanize
import pandas as pds
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import sqlcl
from sql_ask import sdss_sql

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

## based on mechanize package
#url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

## based on sqlcl.py case
#url = "http://skyserver.sdss.org/dr12/en/tools/search/x_sql.aspx"

### star sql
r_select = 0.42 ## 1.5 * diagonal line length

# query with sqlcl.py
def put_sql(z_set, ra_set, dec_set, ref_ra, ref_dec, out_file,):

	#url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'
	url = "http://skyserver.sdss.org/dr12/en/tools/search/x_sql.aspx"

	fmt = 'csv'

	Nz = len(z_set)

	doc = open('/home/xkchen/project/tmp/%d_rank_record.txt' % rank, 'w')

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

		log_s = '%d rank, order = %d' % (rank, q)
		print( log_s, file = doc)

	doc.close()

	return

load = '/home/xkchen/data/SDSS/photo_files/'
home = '/home/xkchen/data/SDSS/'

dat = pds.read_csv(home + 'selection/photo-z_sample_sql_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
ref_ra, ref_dec = np.array(dat['ref_ra']), np.array(dat['ref_dec'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

out_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'

put_sql(z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], ref_ra[N_sub0:N_sub1], ref_dec[N_sub0:N_sub1], out_file,)

print('%d rank finished part-3' % rank)

