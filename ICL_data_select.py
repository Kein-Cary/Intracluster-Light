# theis file use to select those cluster:
# total in the fits image file get the data by wget
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
import changds
import find
import h5py 
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from matplotlib.patches import Circle
'''
goal_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
'''
goal_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
# find the member of each BGC -cluster, by find the repeat ID
repeat = sts.find_repeats(sub_data.ID)
rept_ID = np.int0(repeat)
ID_array = np.int0(sub_data.ID)
sub_redshift = np.array(sub_data.Z_SPEC) #use to figure out how big the satellite
center_distance = sub_data.R # select the distance of satellite galaxies
member_pos = np.array([sub_data.RA,sub_data.DEC]) # record the position of satellite

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
use_z = redshift*1
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
# set 'r' band as a test ('r' band is closer to visible range)
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit
S0 = R_A.value**2*np.pi # use to estimate the effective area of each galaxy in the frame area
# select the number which the cluster total fall in the framefile
tot_sub = np.zeros(len(z), dtype = np.int0)
inr_sub = np.zeros(len(z), dtype = np.int0)
sub_ratio = np.zeros(len(z), dtype = np.float)
area_ratio = np.zeros(len(z), dtype = np.float)
for k in range(len(z)):
    cir_data = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',Ra[k],Dec[k],z[k]),header = True)
    wcs = awc.WCS(cir_data[1])
    h = cir_data[0].shape[0]
    w = cir_data[0].shape[1] # read the data and set the coordinate projection
    
    # find the satellite 
    IA = find.find1d(use_z,z[k])
    rich_IA = rept_ID[1][IA]
    tot_sub[k] = rich_IA-1 # except the BCG itself
    sum_IA = np.sum(rept_ID[1][:IA])
    blnr = center_distance[sum_IA:sum_IA+rich_IA]
    subr = blnr[blnr<=size_cluster/2]
    inr_sub[k] = len(subr)-1 # except the BCG itself
    sub_ratio[k] = inr_sub[k]*1/tot_sub[k]*1
    
    # build the pixel--sky coordinate system
    y = np.linspace(0,1488,1489)
    x = np.linspace(0,2047,2048)
    vx, vy = np.meshgrid(x,y)
    cx, cy = wcs.all_pix2world(vx,vy,1)
    ff = np.abs(cx-Ra[k])
    gg = np.abs(cy-Dec[k])
    hh = np.sqrt((cx-Ra[k])**2+(cy-Dec[k])**2)
    goal = find.find2d(hh,np.min(hh))
    xg = goal[1]
    yg = goal[0]
    # select with ra,dec
    Dpixel = np.sqrt((cx-Ra[k])**2+(cy-Dec[k])**2)
    inpixel = Dpixel[Dpixel <= R_A[k].value/3600]
    npixel = len(inpixel)
    # calculate the area ratio
    SK = npixel*0.396**2
    area_ratio[k] = SK/S0[k]

    xx = Ra[k]
    yy = Dec[k]    
    # select those satellite locate in 1Mpc/h, from the cluster center
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_axes([0.1,0.1,0.8,0.8],projection = wcs)
    im = ax.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm())
    ra = ax.coords[0]
    #ra.set_major_formatter('dd:mm:ss')
    ra.set_major_formatter('d.ddddd')
    ra.grid(color = 'red',alpha = 0.5)
    dec = ax.coords[1]
    #dec.set_major_formatter('dd:mm:ss')
    dec.set_major_formatter('d.ddddd')
    dec.grid(color = 'green', alpha = 0.5)
    ax.scatter(xx,yy, facecolors = '', marker = 'P',edgecolors = 'b',transform=ax.get_transform('world'))
    aa = ax.get_xlim()
    bb = ax.get_ylim()
    
    poa = member_pos[0][sum_IA:sum_IA+rept_ID[1][IA]]
    pob = member_pos[1][sum_IA:sum_IA+rept_ID[1][IA]]
    ak = find.find1d(poa,Ra[k])
    posx = poa[poa!=poa[ak]]
    posy = pob[pob!=pob[ak]] # out the BCG, and then mark the satellite 
    ax.scatter(posx,posy,facecolors = '',marker = 'o',edgecolors = 'r',transform=ax.get_transform('world'))
    #r1 = Circle((xx,yy),radius = R_A[k].value/3600,facecolor = 'None',edgecolor = 'r',transform=ax.get_transform('world'))
    #ax.add_patch(r1)
    hsc.circles(xg,yg,s = R_A[k].value/0.396,fc = '',ec = 'r')
    ra.set_ticks(spacing = 0.05*U.deg)
    ra.set_ticklabel(color = 'red')
    dec.set_ticks(spacing = 0.05*U.deg) 
    dec.set_ticklabel(color = 'green')# set the spacing of ra-dec grid, and must in unit 'degree'
    
    scale_L = R_A[k].value/0.396
    A = [[100,100],[100+scale_L,100]]
    ax.plot(A[1],A[0],'b',)
    ax.text(x=112,y=112,s = 'r = 1Mpc/h',color = 'b')
    ax.axis('scaled')
    ax.set_xlabel('RA-(spacing-[0.05deg])')
    ax.set_ylabel('DEC-(spacing-[0.05deg])')
    ax.set_xlim(aa[0],aa[1])
    ax.set_ylim(bb[0],bb[1])
    plt.colorbar(im,fraction = 0.035,pad = 0.03,label = r'$f_{flux}[nanoMaggy]$') # colorbar adjust
    ax.set_title(r'$Cluster-ra%.3f-dec%.3f-z%.3f-inr%.0f-rS%.3f$'%(Ra[k],Dec[k],z[k],inr_sub[k],area_ratio[k]))
    #plt.savefig(
    #       '/mnt/ddnfs/data_users/cxkttwl/ICL/fig/cluster_select_ra%.3f_dec%.3f_z%.3f_rich%.0f.png'%(Ra[k],Dec[k],z[k],inr_sub[k]),dpi= 600)
    plt.savefig(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/cluster_select_ra%.3f_dec%.3f_z%.3f_rS%.3f_rich%.0f.png'%(Ra[k],Dec[k],z[k],area_ratio[k],inr_sub[k]),dpi= 600)
    plt.show()
    plt.close()
    # after calculate, set the element as 0., avoid check the same cluster
    use_z[IA] = 0.
    # calculate the area size of the cluster in the frame range
"""
record_array: record the total richiness, richiness in R = 1Mpc/h, the ratio of the two; and finally, record
the effective of the cluster in the photo
"""
record_array = np.array([tot_sub, inr_sub, sub_ratio, area_ratio])
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5', 'w') as f:
    f['a'] = record_array
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    for q in range(len(record_array)):
        f['a'][q] = record_array[q]
    