# this file use to figure the sample skymap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as aft
import scipy.stats as sts
# astronomy model
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
# astroML model
goal_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
# select the nearly universe
z = z_eff[z_eff <= 0.3]
Ra = ra_eff[z_eff <= 0.3]
Dec = dec_eff[z_eff <= 0.3]
# set the galactic coordinate
Ra -= 180
Ra *= np.pi / 180
Dec *= np.pi / 180
plt.figure()
ax = plt.axes(projection = 'mollweide')
im = plt.scatter(Ra,Dec,s=10,c= z,cmap = 'rainbow', vmin = np.min(z), vmax = np.max(z),
                 norm = mpl.colors.LogNorm(),alpha = 0.5)
cb = plt.colorbar(im, pad = 0.007,orientation='horizontal')
plt.clim(np.min(z), np.max(z))
cb.set_label('redshift')
plt.grid(True)
plt.title('sample position')
plt.savefig('sample position.png', dpi = 300)
plt.close()
## 0.2~z~0.3
zx = z[(z<=0.3) & (z>=0.2)]
ra = Ra[(z<=0.3) & (z>=0.2)]
dec = Dec[(z<=0.3) & (z>=0.2)]
plt.figure()
ax1 = plt.axes(projection = 'mollweide')
im1 = plt.scatter(ra,dec,s=10,c= zx,cmap = 'rainbow', vmin = np.min(zx), vmax = np.max(zx),
                 norm = mpl.colors.LogNorm(),alpha = 0.5)
cb1 = plt.colorbar(im1, pad = 0.007,orientation='horizontal',)
plt.clim(0.2, 0.3)
cb1.set_label('redshift')
plt.grid(True)
plt.title('$sample [0.2 \sim z \sim 0.3]$')
plt.savefig('sub sample position.png', dpi = 300)
plt.close()
