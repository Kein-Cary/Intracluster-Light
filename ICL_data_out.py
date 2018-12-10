# this file use to ouput the data files
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
import find
##### section 1: read the redmapper data and the member data
goal_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
# find the member of each BGC -cluster, by find the repeat ID
repeat = sts.find_repeats(sub_data.ID)
rept_ID = np.int0(repeat)
ID_array = np.int0(sub_data.ID)
sub_redshift = np.array(sub_data.Z_SPEC)

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

size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
#### section 2: cite the data and save fits figure
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astroML.plotting import setup_text_plots
from astropy.table import Table
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit
band = ['u','g','r','i','z']
k = 1
##### section 2: read the observation data, and point out the BCG and member
cir_data = aft.getdata('area_test.fits',header = True)
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
h = cir_data[0].shape[0]
w = cir_data[0].shape[1]
member_pos = np.array([sub_data.RA,sub_data.DEC])
from matplotlib.patches import Circle

use_z = redshift*1
IA = find.find1d(use_z,z[k])
rich_IA = rept_ID[1][IA]
sum_IA = np.sum(rept_ID[1][:IA])

wcs = awc.WCS(cir_data[1])
fig = plt.figure(figsize = (10,10))
ax = fig.add_axes([0.1,0.1,0.8,0.8*h/w],projection = wcs)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
ax.imshow(cir_data[0],cmap = 'viridis',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
#ax.contour(cir_data[0],cmap = 'gist_heat',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
ra = ax.coords[0]
#ra.set_major_formatter('dd:mm:ss')
ra.set_major_formatter('d.ddddd')
ra.grid(color = 'red')
dec = ax.coords[1]
#dec.set_major_formatter('dd:mm:ss')
dec.set_major_formatter('d.ddddd')
dec.grid(color = 'blue')
xx = Ra[k]
yy = Dec[k]
ax.scatter(xx,yy,facecolors = 'r', marker = 'o',edgecolors = 'b',transform=ax.get_transform('world'))
aa = ax.get_xlim()
bb = ax.get_ylim()
ax.scatter(member_pos[0][sum_IA:sum_IA+rept_ID[1][IA]],member_pos[1][sum_IA:sum_IA+rept_ID[1][IA]],facecolors = '', 
           marker = 'o',edgecolors = 'r',transform=ax.get_transform('world'))
r1 = Circle((xx,yy),R_A[k].value/3600,alpha = 0.25,transform=ax.get_transform('world'))
ax.add_patch(r1)
ax.set_xlim(aa[0],aa[1])
ax.set_ylim(bb[0],bb[1])
plt.savefig('cluster_area.png',dpi= 600)
plt.show()
plt.close()
