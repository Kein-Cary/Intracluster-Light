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
#import find
import h5py
import pandas as pds
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
goal_data = aft.getdata(load + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
Mag_err = np.array(goal_data.MODEL_MAGERR_R)
lamda = np.array(goal_data.LAMBDA)

r_Mag_bcgs = np.array(goal_data.MODEL_MAG_R) ## mag for r band
r_Mag_err = np.array(goal_data.MODEL_MAGERR_R)
g_Mag_bcgs = np.array(goal_data.MODEL_MAG_G)
g_Mag_err = np.array(goal_data.MODEL_MAGERR_G)
i_Mag_bcgs = np.array(goal_data.MODEL_MAG_I)
i_Mag_err = np.array(goal_data.MODEL_MAGERR_I)
u_Mag_bcgs = np.array(goal_data.MODEL_MAG_U)
u_Mag_err = np.array(goal_data.MODEL_MAGERR_U)
z_Mag_bcgs = np.array(goal_data.MODEL_MAG_Z)
z_Mag_err = np.array(goal_data.MODEL_MAGERR_Z)

com_z = redshift[(redshift >= 0.2) & (redshift <= 0.3)]
com_ra = RA[(redshift >= 0.2) & (redshift <= 0.3)]
com_dec = DEC[(redshift >= 0.2) & (redshift <= 0.3)]
com_rich = lamda[(redshift >= 0.2) & (redshift <= 0.3)]

com_r_Mag = r_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_r_Mag_err = r_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

com_g_Mag = g_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_g_Mag_err = g_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

com_i_Mag = i_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_i_Mag_err = i_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

com_u_Mag = u_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_u_Mag_err = u_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

com_z_Mag = z_Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_z_Mag_err = z_Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

band = ['r', 'g', 'i', 'u', 'z']
zN, bN = len(com_z), len(band)
for kk in range(bN):

        with h5py.File(load + 'mpi_h5/Except_%s_sample.h5' % band[kk], 'r') as f:
            except_cat = np.array(f['a'])
        except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
        except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
        #except_z = ['%.3f' % ll for ll in except_cat[2,:] ]

        sub_z = []
        sub_ra = []
        sub_dec = []
        sub_rich = []
        sub_r_mag, sub_r_Merr = [], []
        sub_g_mag, sub_g_Merr = [], []
        sub_i_mag, sub_i_Merr = [], []
        sub_u_mag, sub_u_Merr = [], []
        sub_z_mag, sub_z_Merr = [], []

        for jj in range(zN):
            ra_g = com_ra[jj]
            dec_g = com_dec[jj]
            z_g = com_z[jj]
            rich_g = com_rich[jj]

            r_mag, r_err = com_r_Mag[jj], com_r_Mag_err[jj]
            g_mag, g_err = com_g_Mag[jj], com_g_Mag_err[jj]
            i_mag, i_err = com_i_Mag[jj], com_i_Mag_err[jj]
            u_mag, u_err = com_u_Mag[jj], com_u_Mag_err[jj]
            z_mag, z_err = com_z_Mag[jj], com_z_Mag_err[jj]
            ## rule out bad image (once a cluster image is bad in a band, all the five band image will be ruled out!)
            identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z)
            if  identi == True: 
                continue
            else:
                sub_z.append(z_g)
                sub_ra.append(ra_g)
                sub_dec.append(dec_g)
                sub_rich.append(rich_g)

                sub_r_mag.append(r_mag)
                sub_g_mag.append(g_mag)
                sub_i_mag.append(i_mag)
                sub_u_mag.append(u_mag)
                sub_z_mag.append(z_mag)

                sub_r_Merr.append(r_err)
                sub_g_Merr.append(g_err)
                sub_i_Merr.append(i_err)
                sub_u_Merr.append(u_err)
                sub_z_Merr.append(z_err)

        sub_z = np.array(sub_z)
        sub_ra = np.array(sub_ra)
        sub_dec = np.array(sub_dec)
        sub_rich = np.array(sub_rich)

        sub_r_mag = np.array(sub_r_mag)
        sub_g_mag = np.array(sub_g_mag)
        sub_i_mag = np.array(sub_i_mag)
        sub_u_mag = np.array(sub_u_mag)
        sub_z_mag = np.array(sub_z_mag)

        sub_r_Merr = np.array(sub_r_Merr)
        sub_g_Merr = np.array(sub_g_Merr)
        sub_i_Merr = np.array(sub_i_Merr)
        sub_u_Merr = np.array(sub_u_Merr)
        sub_z_Merr = np.array(sub_z_Merr)

        ## save the csv file
        keys = ['ra', 'dec', 'z', 'rich', 'r_Mag', 'g_Mag', 'i_Mag', 'u_Mag', 'z_Mag', 
                'r_Mag_err', 'g_Mag_err', 'i_Mag_err', 'u_Mag_err', 'z_Mag_err']
        values = [sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
                sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr]
        fill = dict(zip(keys, values))
        data = pds.DataFrame(fill)
        data.to_csv(load + '%s_band_stack_catalog.csv' % band[kk])
        ## save h5py for mpirun
        sub_array = np.array([sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
                sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr])
        with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'w') as f:
            f['a'] = np.array(sub_array)
        with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk]) as f:
            for tt in range( len(sub_array) ):
                f['a'][tt,:] = sub_array[tt,:]


raise
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
richness = np.array(goal_data.LAMBDA)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]
# select the nearly universe
z = z_eff[z_eff <= 0.3]
Ra = ra_eff[z_eff <= 0.3]
Dec = dec_eff[z_eff <= 0.3]
rich = rich_eff[z_eff <= 0.3]
use_z = redshift*1
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
# set 'r' band as a test ('r' band is closer to visible range)
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit
S0 = R_A.value**2*np.pi/(0.396**2) # use to estimate the effective area of each galaxy in the frame area
# select the number which the cluster total fall in the framefile
tot_sub = np.zeros(len(z), dtype = np.int0)
inr_sub = np.zeros(len(z), dtype = np.int0)
sub_ratio = np.zeros(len(z), dtype = np.float)
area_ratio = np.zeros(len(z), dtype = np.float)
reference_ratio = np.zeros(len(z), dtype = np.float)
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
    '''
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
    SK = npixel*1
    '''
    c0 = SkyCoord(Ra[k]*U.deg,Dec[k]*U.deg,frame = 'icrs')
    c1 = SkyCoord(cx*U.deg,cy*U.deg,frame = 'icrs')
    sep = c0.separation(c1)
    ak = R_A[k].value/3600
    ik = sep.value <= ak
    inrg = sep[ik]
    referS = len(inrg)
    reference_ratio[k] = referS/S0[k]
    t1, t2 = wcs.all_world2pix(Ra[k]*U.deg,Dec[k]*U.deg,1)
    dep = np.sqrt((vx-t1)**2+(vy-t2)**2)
    ag = R_A[k].value/0.396
    ig = dep <= ag
    al = dep[ig]
    SK = len(al)
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
    hsc.circles(t1,t2,s = R_A[k].value/0.396,fc = '',ec = 'r')
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
    ax.set_title(r'$Cluster \ ra%.3f \ dec%.3f \ z%.3f \ \lambda%.3f \ rS%.3f$'%(Ra[k],Dec[k],z[k],rich[k],area_ratio[k]))
    #plt.savefig(
    #       '/mnt/ddnfs/data_users/cxkttwl/ICL/fig/cluster_select_ra%.3f_dec%.3f_z%.3f_rS%.3f_rich%.0f.png'%(Ra[k],Dec[k],z[k],area_ratio[k],inr_sub[k]),dpi= 600)
    plt.savefig(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/cluster_select_ra%.3f_dec%.3f_z%.3f_rS%.3f_rich%.3f.png'%(Ra[k],Dec[k],z[k],area_ratio[k],rich[k]),dpi= 600)
    plt.show()
    plt.close()
    # after calculate, set the element as 0., avoid check the same cluster
    use_z[IA] = 0.
    # calculate the area size of the cluster in the frame range
"""
record_array: record the total richiness, richiness in R = 1Mpc/h, the ratio of the two; and finally, record
the effective of the cluster in the photo
"""
record_array = np.array([tot_sub, inr_sub, sub_ratio, area_ratio, reference_ratio, rich])
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5', 'w') as f:
    f['a'] = record_array
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    for q in range(len(record_array)):
        f['a'][q] = record_array[q]
