"""
use for field galaxy profMean query
(those galaxy have applied the same pre-selection as redMapper)
"""
import glob

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


### === 
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)


### ===
def galx_prof_sql(ra_g, dec_g, ID_x, out_files):

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	"""
	out_files : *.txt files
	"""

	data_set = """
	SELECT
		pro.objID, pro.bin, pro.band, pro.profMean, pro.profErr

	FROM PhotoProfile AS pro

	WHERE
		pro.objID = %d
		AND pro.bin BETWEEN 0 AND 15
	""" % ID_x

	br = mechanize.Browser()
	resp = br.open( url )
	resp.info()

	br.select_form( name = "sql" )
	br['cmd'] = data_set
	br['format'] = ['csv']
	response = br.submit()

	s = str(response.get_data(), encoding = 'utf-8')

	doc = open( out_files % (ra_g, dec_g), 'w')
	print(s, file = doc)
	doc.close()

	return


### === ### data load
cat_path = '/home/xkchen/data/SDSS/field_galx_redMap/prof_sql_cat/'
out_path = '/home/xkchen/data/SDSS/field_galx_redMap/prof_Mean_cat/' 

cat_files = glob.glob( cat_path + '*.fit')

N_files = len( cat_files )
print( N_files )


# for dd in range( N_files ):
for dd in range( 1 ):

	# data = fits.open( cat_files[ dd ] )
	data = fits.open( cat_path + 'red_Map_limit_galx_cat_21.0-21.1mag.fit' )

	cat_table = data[1].data

	ra = cat_table['ra']
	dec = cat_table['dec']
	IDs = cat_table['objID']

	N_g = len( ra )

	for nn in range( N_g ):

		ra_g, dec_g = ra[ nn ], dec[ nn ]

		ID_x = IDs[ nn ]

		out_file = out_path + 'ctrl_prof_ra%.5f_dec%.5f.txt'

		galx_prof_sql(ra_g, dec_g, ID_x, out_file)

		print('nn = ', nn)

