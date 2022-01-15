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

url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/mpi_h5/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

## in unit of degree, centered at BCG center, about 70 arcsec
r_select = 0.02

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
goal_data = fits.getdata(load + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits')
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
Mag_err = np.array(goal_data.MODEL_MAGERR_R)
GID = np.array(goal_data.OBJID)
com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag_err = Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]
com_ID = GID[(redshift >= 0.2) & (redshift <= 0.3)]

def BCG_mag_sql(cat_z, cat_ra, cat_dec, cat_ID):
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

		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( load + 'BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

	return

def main():
	Ntot = len(z)
	commd.Barrier()

	m, n = divmod(Ntot, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	BCG_mag_sql(z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], com_ID[N_sub0 :N_sub1])
	commd.Barrier()

if __name__ == "__main__":
	main()
