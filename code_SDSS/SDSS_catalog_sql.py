import time
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

### constant
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396

###
url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

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
				p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z,
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
				p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z,
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
				p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z,
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

def sdss_sql_galaxy(set_ra, set_dec, set_z, ref_ra, ref_dec, out_file,):

	Nz = len(set_z)
	for q in range(Nz):

		z_g = set_z[q]
		ra_g = set_ra[q]
		dec_g = set_dec[q]

		c_ra0 = ref_ra[q] - r_select
		c_dec0 = ref_dec[q] - r_select
		c_ra1 = ref_ra[q] + r_select
		c_dec1 = ref_dec[q] + r_select

		cen_ra = ref_ra[q]
		cen_dec = ref_dec[q]

		if cen_ra + r_select > 360:

			l_ra_0 = cen_ra
			l_ra_1 = 0

			r_ra_0 = 360
			r_ra_1 = r_select - (360 - cen_ra)

			data_set = """
			SELECT ALL
				p.ra,p.dec,p.g,p.r,p.i,p.type,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z

			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 3 )

				AND ( p.mode = 1 )
				AND ( p.clean = 1 )
				AND ( p.flags & cast(8800388251650 as bigint) = 0 )
				AND ( p.cModelMag_i - p.extinction_i BETWEEN 0 and 21.0 )
				AND ( p.cModelMagErr_i BETWEEN 0 and 0.1 )

				--AND ((p.flags & 0x10000000) != 0)
				--AND ((flags & 0x800a0) = 0)
				--AND (((flags & 0x400000000000) = 0) OR (psfmagerr_r <= 0.2))
				--AND (((flags & 0x100000000000) = 0) OR (flags & 0x1000) = 0)
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)		

		elif cen_ra - r_select < 0:

			l_ra_0 = 0
			l_ra_1 = 360 + (cen_ra - r_select)

			r_ra_0 = cen_ra
			r_ra_1 = 360

			data_set = """
			SELECT ALL
				p.ra,p.dec,p.g,p.r,p.i,p.type,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z

			FROM PhotoObjAll AS p
			WHERE
				( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 3 )

				AND ( p.mode = 1 )
				AND ( p.clean = 1 )
				AND ( p.flags & cast(8800388251650 as bigint) = 0 )
				AND ( p.cModelMag_i - p.extinction_i BETWEEN 0 AND 21.0 )
				AND ( p.cModelMagErr_i BETWEEN 0 AND 0.1 )

				--AND ((p.flags & 0x10000000) != 0)
				--AND ((flags & 0x800a0) = 0)
				--AND (((flags & 0x400000000000) = 0) OR (psfmagerr_r <= 0.2))
				--AND (((flags & 0x100000000000) = 0) OR (flags & 0x1000) = 0)
			""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

		else:

			data_set = """
			SELECT ALL
				p.ra,p.dec,p.g,p.r,p.i,p.type,

				p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
				p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
				p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

				p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
				p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
				p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z

			FROM PhotoObjAll AS p
			WHERE
					( p.ra BETWEEN %.5f AND %.5f )
				AND ( p.dec BETWEEN %.5f AND %.5f )
				AND ( p.type = 3 )

				AND ( p.mode = 1 )
				AND ( p.clean = 1 )
				AND ( p.flags & cast(8800388251650 as bigint) = 0 )
				AND ( p.cModelMag_i - p.extinction_i BETWEEN 0 AND 21.0 )
				AND ( p.cModelMagErr_i BETWEEN 0 AND 0.1 )

				--AND ((p.flags & 0x10000000) != 0)
				--AND ((flags & 0x800a0) = 0)
				--AND (((flags & 0x400000000000) = 0) OR (psfmagerr_r <= 0.2))
				--AND (((flags & 0x100000000000) = 0) OR (flags & 0x1000) = 0)
			""" % (c_ra0, c_ra1, c_dec0, c_dec1)

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()

		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open(out_file % (z_g, ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

	return

home = '/home/xkchen/data/SDSS/'

# random cat
dat = pds.read_csv(home + 'selection/SDSS_random-sample_sql_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
ref_ra, ref_dec = np.array(dat['ref_ra']), np.array(dat['ref_dec'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

out_file = home + 'new_star_sql/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

sdss_sql_star(z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1],
	ref_ra[N_sub0:N_sub1], ref_dec[N_sub0:N_sub1], out_file,)

print('random finished !')

'''
dat = pds.read_csv(home + 'selection/SDSS_spec-sample_sql_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
ref_ra, ref_dec = np.array(dat['ref_ra']), np.array(dat['ref_dec'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

out_file = home + 'new_star_sql/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

sdss_sql_star(z[N_sub0:N_sub1], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1],
	ref_ra[N_sub0:N_sub1], ref_dec[N_sub0:N_sub1], out_file,)

print('cluster finished !')
'''

