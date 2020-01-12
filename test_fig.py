import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import matplotlib.image as mpimg

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse
from scipy import ndimage
from PIL import Image
# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

## fits data
# read the member galaxy of SDSS dr8 Redmapper
BCG_cat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
member_cat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

# find the member of each BGC -cluster, by find the repeat ID
repeat = sts.find_repeats(member_cat.ID)
rept_ID = np.int0(repeat)

ID_array = np.int0(member_cat.ID)
sub_redshift = np.array(member_cat.Z_SPEC)
center_distance = member_cat.R
member_pos = np.array([member_cat.RA, member_cat.DEC])

# read the center galaxy position
RA = np.array(BCG_cat.RA)
DEC = np.array(BCG_cat.DEC)
redshift = np.array(BCG_cat.Z_SPEC)
richness = np.array(BCG_cat.LAMBDA)
host_ID = np.array(BCG_cat.ID)

ra_g, dec_g = 20.832, 12.319
Da_g = Test_model.angular_diameter_distance(0.216).value
Rp = (rad2asec / Da_g) / pixel / 0.7

ds = np.sqrt((RA - ra_g)**2 + (DEC - dec_g)**2)
idx = ds == np.min(ds)
indx = np.where(idx == True)[0][0]
satellite_N = repeat[1][indx]
sum_before = np.sum(repeat[1][:indx])
satellite_pos_ra = member_pos[0][sum_before : sum_before + satellite_N]
satellite_pos_dec = member_pos[1][sum_before : sum_before + satellite_N]

load = '/home/xkcehn/mywork/ICL/data/tot_data/'
sdss_file = '/home/xkchen/mywork/ICL/data/total_data/frame-r-ra20.832-dec12.319-redshift0.216.fits.bz2'

dat_sdss = fits.open(sdss_file)
sdss_img = dat_sdss[0].data
wcs_sdss = awc.WCS(dat_sdss[0].header)
cx0, cy0 = wcs_sdss.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
sub_x, sub_y = wcs_sdss.all_world2pix(satellite_pos_ra * U.deg, satellite_pos_dec * U.deg, 0)

la0, la1 = np.max([0, cx0 - Rp]), np.min([sdss_img.shape[1], cx0 + Rp])
lb0, lb1 = np.max([0, cy0 - Rp]), np.min([sdss_img.shape[0], cy0 + Rp])
'''
plt.figure()
ax = plt.subplot(111)
ax.set_title('SDSS image')
ax.imshow(sdss_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
Cile0 = Circle(xy = (cx0, cy0), radius = Rp, fill = False, ec = 'b', alpha = 0.5, label = 'Cluster [1Mpc]')
ax.add_patch(Cile0)
for kk in range(satellite_N):
	Cilek = Circle(xy = (sub_x[kk], sub_y[kk]), radius = 10, fill = False, ec = 'r', linewidth = 0.75,)
	ax.add_patch(Cilek)
ax.legend(loc = 1)

ax.set_xlim(la0, la1)
ax.set_ylim(lb0, lb1)

plt.savefig('SDSS_match_test.png', dpi = 300)
plt.show()
'''
## rot image and compare

file_0 = '/home/xkchen/jupyter/Decals_ra20.832_dec12.319.jpg'
file_1 = '/home/xkchen/jupyter/SDSS_ra20.832_dec12.319.jpg'

open_decal = Image.open(file_0)
open_sdss = Image.open(file_1)

Decal_img = open_decal.convert('L')
SDSS_img = open_sdss.convert('L')
'''
Decal_img = mpimg.imread(file_0)
SDSS_img = mpimg.imread(file_1)
'''
Alpha = 90
rot_SDSS = ndimage.rotate(SDSS_img, angle = Alpha,)

plt.figure()
plt.title('Decals image')
plt.imshow(Decal_img, cmap = 'Greys', )#vmin = 1e-5, vmax = 1e3, norm = mpl.colors.LogNorm())
plt.xticks([])
plt.yticks([])
plt.axis('equal')
plt.savefig('Decals_Greys.png', dpi = 600)
plt.show()

plt.figure()
ax = plt.subplot(111)
ax.set_title('SDSS image')
#ax.imshow(rot_SDSS, cmap = 'Greys', )#vmin = 1e-5, vmax = 1e3, norm = mpl.colors.LogNorm())
ax.imshow(SDSS_img, cmap = 'Greys', )#vmin = 1e-5, vmax = 1e3, norm = mpl.colors.LogNorm())
Cile0 = Circle(xy = (cx0, 1489 - cy0), radius = Rp, fill = False, ec = 'b', alpha = 0.25, label = 'Cluster [1Mpc]')
ax.add_patch(Cile0)
for kk in range(satellite_N):
	Cilek = Circle(xy = (sub_x[kk], 1489 - sub_y[kk]), radius = 10, fill = False, ec = 'r', linewidth = 0.5, alpha = 0.25)
	ax.add_patch(Cilek)
ax.legend(loc = 1)

ax.set_xticks([])
ax.set_yticks([])
ax.axis('equal')
plt.savefig('SDSS_Greys.png', dpi = 600)
plt.show()
