import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

from io import StringIO
import wget as wt
import subprocess as subpro

""" 
##.. information match
pat = pds.read_csv('/home/xkchen/redMapper_z-photo_cat.csv')
ra, dec, z = np.array( pat['ra'] ), np.array( pat['dec'] ), np.array( pat['z'] )

img_file = '/home/xkchen/data/SDSS/photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

Ns = len( ra )

keys = ['ra', 'dec', 'z', 'RERUN', 'RUN', 'CAMCOL', 'FIELD']

rerun_arr = np.zeros( Ns, dtype = np.int32 )
run_arr = np.zeros( Ns, dtype = np.int32 )
camcol_arr = np.zeros( Ns, dtype = np.int32 )
field_arr = np.zeros( Ns, dtype = np.int32 )

for kk in range( Ns ):

	ra_g, dec_g, z_g = ra[ kk ], dec[ kk ], z[ kk ]

	img_data = fits.open( img_file % (ra_g, dec_g, z_g),)

	Table = img_data[3].data

	rerun_arr[ kk ] = Table['RERUN'][0]
	run_arr[ kk ] = Table['RUN'][0]
	camcol_arr[ kk ] = Table['CAMCOL'][0]
	field_arr[ kk ] = Table['FIELD'][0]

values = [ ra, dec, z, rerun_arr, run_arr, camcol_arr, field_arr ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/z-photo_cat_SDSS_frame_info.csv')

"""


##.. PSF image download
# 'https://data.sdss.org/sas/dr12/boss/photo/redux/301/3840/objcs/4/psField-003840-4-0269.fit'
# ................................................rerun/run/...../camcol/...-run-camcol-field...

url = 'https://data.sdss.org/sas/dr12/boss/photo/redux/%d/%d/objcs/%d/psField-%s-%s-%s.fit'

pat = pds.read_csv('/home/xkchen/z-photo_cat_SDSS_frame_info.csv')
ra, dec, z = np.array( pat['ra'] ), np.array( pat['dec'] ), np.array( pat['z'] )
rerun, run, camcol, field = np.array( pat['RERUN'] ), np.array( pat['RUN'] ), np.array( pat['CAMCOL'] ), np.array( pat['FIELD'] )

Ns = len( ra )

err_ID = np.array( [] )
err_ra, err_dec, err_z = np.array( [] ), np.array( [] ), np.array( [] )

for kk in range( Ns ):

	ra_g, dec_g, z_g = ra[ kk ], dec[ kk ], z[ kk ]

	t_rerun, t_run = rerun[ kk ], run[ kk ]
	t_camcol, t_field = camcol[ kk ], field[ kk ]


	fill_s_0 = str( t_run ).zfill( 6 )
	fill_s_1 = str( t_field ).zfill( 4 )

	links = url % ( t_rerun, t_run, t_camcol, fill_s_0, t_camcol, fill_s_1 )
	out_file = '/home/xkchen/figs/psfField_ra%.3f_dec%.3f_z%.3f.fit' % (ra_g, dec_g, z_g)

	try:
		wt.download( links, out_file )

	except:
		err_ID = np.r_[ err_ID, tt ]
		err_ra = np.r_[ err_ra, ra_g ]
		err_dec = np.r_[ err_dec, dec_g ]
		err_z = np.r_[ err_z, z_g ]


#..
keys = ['ordex', 'ra', 'dec', 'z']
values = [err_ID, err_ra, err_dec, err_z]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/z-photo_psfField_sql-err_cat.csv')



# ##. extract psf image
# cmd = 'read_PSF /home/xkchen/psfField_ra250.083_dec46.712_z0.233.fit 3 500 600 /home/xkchen/test_psf.fit'
# A = subpro.Popen( cmd, shell = True )
# A.wait()

