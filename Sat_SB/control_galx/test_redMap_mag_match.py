import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

from scipy import spatial
from sklearn.neighbors import KDTree
from sklearn import metrics

from scipy import interpolate as interp
from scipy import integrate as integ

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()



###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396


### === distance function test

lim_data = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/' + 
                    'sdss_redMap_member-mag_of_clus_z0.2to0.3.fits')

lim_table = lim_data[1].data

lim_z = lim_table['z']

lim_cmag_u = lim_table['cModelMag_u']
lim_cmag_g = lim_table['cModelMag_g']
lim_cmag_r = lim_table['cModelMag_r']
lim_cmag_i = lim_table['cModelMag_i']
lim_cmag_z = lim_table['cModelMag_z']

lim_mag_u = lim_table['modelMag_u']
lim_mag_g = lim_table['modelMag_g']
lim_mag_r = lim_table['modelMag_r']
lim_mag_i = lim_table['modelMag_i']
lim_mag_z = lim_table['modelMag_z']

lim_ug = lim_mag_u - lim_mag_g
lim_gr = lim_mag_g - lim_mag_r
lim_ri = lim_mag_r - lim_mag_i
lim_iz = lim_mag_i - lim_mag_z
lim_gi = lim_mag_g - lim_mag_i


##.
all_cat = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/' + 
			'sdss_galaxy_i-cmag_to_21mag.fits' )

all_arr = all_cat[1].data

all_z, all_z_err = np.array( all_arr['z'] ), np.array( all_arr['zErr'] )

all_dered_u = np.array( all_arr['dered_u'] )
all_dered_g = np.array( all_arr['dered_g'] )
all_dered_r = np.array( all_arr['dered_r'] )
all_dered_i = np.array( all_arr['dered_i'] )
all_dered_z = np.array( all_arr['dered_z'] )

all_cmag_u = np.array( all_arr['cModelMag_u'] )
all_cmag_g = np.array( all_arr['cModelMag_g'] )
all_cmag_r = np.array( all_arr['cModelMag_r'] )
all_cmag_i = np.array( all_arr['cModelMag_i'] )
all_cmag_z = np.array( all_arr['cModelMag_z'] )

all_gr = all_dered_g - all_dered_r
all_ri = all_dered_r - all_dered_i
all_gi = all_dered_g - all_dered_i
all_ug = all_dered_u - all_dered_g
all_iz = all_dered_i - all_dered_z


##. use partial for test
N_all = len( all_cmag_r )
tt0 = np.int( N_all / 5 )

t_dex = np.random.choice( N_all, tt0, replace = False,)

cc_cmag_u = all_cmag_u[ t_dex ]
cc_cmag_g = all_cmag_g[ t_dex ]
cc_cmag_r = all_cmag_r[ t_dex ]
cc_cmag_i = all_cmag_i[ t_dex ]
cc_cmag_z = all_cmag_z[ t_dex ]

cc_gr = all_gr[ t_dex ]
cc_ri = all_ri[ t_dex ]
cc_gi = all_gi[ t_dex ]
cc_ug = all_ug[ t_dex ]
cc_iz = all_iz[ t_dex ]


##.
# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z ] ).T
# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_gi ] ).T
# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_gi, lim_ug ] ).T
lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_ug ] ).T
pre_Tree = spatial.KDTree( lim_arr )

# sql_arr = np.array( [ cc_cmag_u, cc_cmag_g, cc_cmag_r, cc_cmag_i, cc_cmag_z ] ).T
# sql_arr = np.array( [ cc_cmag_u, cc_cmag_g, cc_cmag_r, cc_cmag_i, cc_cmag_z, cc_gr, cc_gi ] ).T
# sql_arr = np.array( [ cc_cmag_u, cc_cmag_g, cc_cmag_r, cc_cmag_i, cc_cmag_z, cc_gr, cc_gi, cc_ug ] ).T
sql_arr = np.array( [ cc_cmag_u, cc_cmag_g, cc_cmag_r, cc_cmag_i, cc_cmag_z, cc_gr, cc_ri, cc_ug ] ).T
sql_Tree = spatial.KDTree( sql_arr )
map_Tree, map_idex = sql_Tree.query( lim_arr, k = 3,)


# map_cmag_u = all_cmag_u[ map_idex ].flatten()
# map_cmag_g = all_cmag_g[ map_idex ].flatten()
# map_cmag_r = all_cmag_r[ map_idex ].flatten()
# map_cmag_i = all_cmag_i[ map_idex ].flatten()
# map_cmag_z = all_cmag_z[ map_idex ].flatten()

# map_ug = all_ug[ map_idex ].flatten()
# map_gr = all_gr[ map_idex ].flatten()
# map_gi = all_gi[ map_idex ].flatten()
# map_ri = all_ri[ map_idex ].flatten()
# map_iz = all_iz[ map_idex ].flatten()


map_cmag_u = cc_cmag_u[ map_idex ].flatten()
map_cmag_g = cc_cmag_g[ map_idex ].flatten()
map_cmag_r = cc_cmag_r[ map_idex ].flatten()
map_cmag_i = cc_cmag_i[ map_idex ].flatten()
map_cmag_z = cc_cmag_z[ map_idex ].flatten()

map_ug = cc_ug[ map_idex ].flatten()
map_gr = cc_gr[ map_idex ].flatten()
map_gi = cc_gi[ map_idex ].flatten()
map_ri = cc_ri[ map_idex ].flatten()
map_iz = cc_iz[ map_idex ].flatten()


bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 65)
bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)
bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 65)

bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 65)
bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)
bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 65)


##.
fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '-', alpha = 0.75, label = 'u-band, redMapper')
gax.hist( all_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'u-band, all galaxy')
gax.hist( cc_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'random selected')
gax.set_ylim( 0, 0.8 )
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '-', alpha = 0.75, label = 'g-band, redMapper')
gax.hist( all_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'g-band, all galaxy')
gax.hist( cc_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'random, selected')
gax.set_ylim( 0, 0.8 )
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '-', alpha = 0.75, label = 'r-band, redMapper')
gax.hist( all_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'r-band, all galaxy')
gax.hist( cc_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'random, selected')
gax.set_ylim( 0, 0.8 )
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '-', alpha = 0.75, label = 'i-band, redMapper')
gax.hist( all_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'i-band, all galaxy')
gax.hist( cc_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'random, selected')
gax.set_ylim( 0, 0.8 )
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '-', alpha = 0.75, label = 'z-band, redMapper')
gax.hist( all_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'z-band, all galaxy')
gax.hist( cc_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'random, selected')
gax.set_ylim( 0, 0.8 )
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/sql_mag_list.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'u-band, redMapper')
gax.hist( map_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'g-band, redMapper')
gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'r-band, redMapper')
gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'i-band, redMapper')
gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'z-band, redMapper')
gax.hist( map_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/mag_map_test.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_ug, bins = bins_ug, density = True, color = 'b', alpha = 0.75, label = 'u-g, redMapper')
gax.hist( map_ug, bins = bins_ug, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_gr, bins = bins_gr, density = True, color = 'g', alpha = 0.75, label = 'g-r, redMapper')
gax.hist( map_gr, bins = bins_gr, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_gi, bins = bins_gi, density = True, color = 'r', alpha = 0.75, label = 'g-i, redMapper')
gax.hist( map_gi, bins = bins_gi, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_ri, bins = bins_ri, density = True, color = 'm', alpha = 0.75, label = 'r-i, redMapper')
gax.hist( map_ri, bins = bins_ri, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_iz, bins = bins_iz, density = True, color = 'c', alpha = 0.75, label = 'i-z, redMapper')
gax.hist( map_iz, bins = bins_iz, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'control')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/color_map_test.png', dpi = 300)
plt.close()


raise


### === total histogram compare
lim_data = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/' + 
                    'sdss_redMap_member-mag_of_clus_z0.2to0.3.fits')

lim_table = lim_data[1].data

lim_z = lim_table['z']

lim_cmag_u = lim_table['cModelMag_u']
lim_cmag_g = lim_table['cModelMag_g']
lim_cmag_r = lim_table['cModelMag_r']
lim_cmag_i = lim_table['cModelMag_i']
lim_cmag_z = lim_table['cModelMag_z']

lim_mag_r = lim_table['modelMag_u']
lim_mag_g = lim_table['modelMag_g']
lim_mag_i = lim_table['modelMag_r']
lim_mag_u = lim_table['modelMag_i']
lim_mag_z = lim_table['modelMag_z']

lim_ug = lim_mag_u - lim_mag_g
lim_gr = lim_mag_g - lim_mag_r
lim_ri = lim_mag_r - lim_mag_i
lim_iz = lim_mag_i - lim_mag_z
lim_gi = lim_mag_g - lim_mag_i


all_cat = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/' + 
                    'sdss_galaxy_i-cmag_to_21mag.fits' )

all_arr = all_cat[1].data

all_z, all_z_err = np.array( all_arr['z'] ), np.array( all_arr['zErr'] )

all_dered_u = np.array( all_arr['dered_u'] )
all_dered_g = np.array( all_arr['dered_g'] )
all_dered_r = np.array( all_arr['dered_r'] )
all_dered_i = np.array( all_arr['dered_i'] )
all_dered_z = np.array( all_arr['dered_z'] )

all_cmag_u = np.array( all_arr['cModelMag_u'] )
all_cmag_g = np.array( all_arr['cModelMag_g'] )
all_cmag_r = np.array( all_arr['cModelMag_r'] )
all_cmag_i = np.array( all_arr['cModelMag_i'] )
all_cmag_z = np.array( all_arr['cModelMag_z'] )

all_gr = all_dered_g - all_dered_r
all_ri = all_dered_r - all_dered_i
all_gi = all_dered_g - all_dered_i
all_ug = all_dered_u - all_dered_g
all_iz = all_dered_i - all_dered_z


bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 55)
bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 55)
bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 55)
bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 55)
bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 55)

bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 55)
bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 55)
bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 55)
bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 55)
bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 55)

bins_z = np.linspace( np.median( lim_z ) - 5 * np.std( lim_z ), np.median( lim_z ) + 5 * np.std( lim_z ), 55 )


print('redMap, g', np.sum(lim_cmag_g < 21.) )
print('all, g', np.sum(all_cmag_g < 21.) )

print('redMap, r', np.sum(lim_cmag_r < 20.) )
print('all, r', np.sum(all_cmag_r < 20.) )

print('redMap, i', np.sum(lim_cmag_i < 20.) )
print('all, i', np.sum(all_cmag_i < 20.) )


##. figs
plt.figure()
plt.hist( lim_z, bins = bins_z, density = False, color = 'r', alpha = 0.75, label = 'redMaPPer member')
plt.hist( all_z, bins = bins_z, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
plt.legend( loc = 2, frameon = False)
plt.savefig('/home/xkchen/redMap_lim_galaxy_z-photo.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_cmag_u, bins = bins_mag_u, density = False, color = 'b', alpha = 0.75, label = 'u-band, redMaPPer')
gax.hist( all_cmag_u, bins = bins_mag_u, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_cmag_g, bins = bins_mag_g, density = False, color = 'g', alpha = 0.75, label = 'g-band, redMaPPer')
gax.hist( all_cmag_g, bins = bins_mag_g, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_cmag_r, bins = bins_mag_r, density = False, color = 'r', alpha = 0.75, label = 'r-band, redMaPPer')
gax.hist( all_cmag_r, bins = bins_mag_r, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_cmag_i, bins = bins_mag_i, density = False, color = 'm', alpha = 0.75, label = 'i-band, redMaPPer')
gax.hist( all_cmag_i, bins = bins_mag_i, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_cmag_z, bins = bins_mag_z, density = False, color = 'c', alpha = 0.75, label = 'z-band,m redMaPPer')
gax.hist( all_cmag_z, bins = bins_mag_z, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/redMap_lim_galaxy_mag.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_ug, bins = bins_ug, density = False, color = 'b', alpha = 0.75, label = 'u-g, redMaPPer')
gax.hist( all_ug, bins = bins_ug, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_gr, bins = bins_gr, density = False, color = 'g', alpha = 0.75, label = 'g-r, redMaPPer')
gax.hist( all_gr, bins = bins_gr, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_gi, bins = bins_gi, density = False, color = 'r', alpha = 0.75, label = 'g-i, redMaPPer')
gax.hist( all_gi, bins = bins_gi, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_ri, bins = bins_ri, density = False, color = 'm', alpha = 0.75, label = 'r-i, redMaPPer')
gax.hist( all_ri, bins = bins_ri, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_iz, bins = bins_iz, density = False, color = 'c', alpha = 0.75, label = 'i-z, redMaPPer')
gax.hist( all_iz, bins = bins_iz, density = False, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'all galaxy')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/redMap_lim_galaxy_color.png', dpi = 300)
plt.close()

