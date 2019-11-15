import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pd 
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time
###
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

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
Ra = catalogue[1]
Dec = catalogue[2]
'''
Rp = 1.5 * rad2asec / Test_model.angular_diameter_distance(z).value # select the region within 1.5Mpc
r_select = Rp / 3600
'''
r_select = 0.167 # centered at BCG, radius = 10 arcmin (1515.15 pixel)

def sdss_sql(z_set, ra_set, dec_set):
	Nz = len(z_set)
	for q in range(Nz):
		time = z_set[q]
		ra = ra_set[q]
		dec = dec_set[q]
		set_r = r_select
		c_ra0 = str(ra - set_r)
		c_dec0 = str(dec - set_r)
		c_ra1 = str(ra + set_r)
		c_dec1 = str(dec + set_r)

		data_set = """
		SELECT ALL
			p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,p.type,  
			p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
			p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,

			p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
			p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
			p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

			p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
			p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
			p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z,
			p.flags, dbo.fPhotoFlagsN(p.flags)
		FROM PhotoObj AS p
		WHERE
			p.ra BETWEEN %s AND %s
			AND p.dec BETWEEN %s AND %s
			AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
		ORDER by p.r
		""" % (c_ra0, c_ra1, c_dec0, c_dec1)
		
		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()
		print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( 
			'/mnt/ddnfs/data_users/cxkttwl/ICL/data/bright_star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(time, ra, dec), 'w')
		print(s, file = doc)
		doc.close()

	return

def main():
	t0 = time.time()
	Ntot = len(z)
	commd.Barrier()

	m, n = divmod(Ntot, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n
	sdss_sql(z[N_sub0 :N_sub1], Ra[N_sub0 :N_sub1], Dec[N_sub0 :N_sub1])
	commd.Barrier()
	t1 = time.time() - t0
	print('t = ', t1)

if __name__ == "__main__":
	main()
