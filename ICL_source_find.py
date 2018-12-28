#combine with sextractor
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as aft
import scipy.stats as sts
import find
import h5py 
import matplotlib.gridspec as grid
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from astropy.coordinates import SkyCoord
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
# read the center galaxy position
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)
host_ID = np.array(goal_data.ID)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]
ID_eff = host_ID[redshift != -1]
# select the nearly universe
z = z_eff[z_eff <= 0.3]
Ra = ra_eff[z_eff <= 0.3]
Dec = dec_eff[z_eff <= 0.3]
Rich = rich_eff[z_eff <= 0.3]
cg_ID = ID_eff[z_eff <= 0.3]
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
# set 'r' band as a test ('r' band is closer to visible range)
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit
# use the follow three array to trace the member galaixes
use_z = redshift*1
use_rich = richness*1
use_ID = host_ID*1
####### read the S/S0, richness
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
sampl_S_ratio = area_ratio[(z<=0.3) & (z>=0.2)]
sampl_refer = reference_ratio[(z<=0.3) & (z>=0.2)]
sampl_z = z[(z<=0.3) & (z>=0.2)]
sampl_rich = rich[(z<=0.3) & (z>=0.2)]
sampl_R_A = R_A[(z<=0.3) & (z>=0.2)]
sampl_ID = cg_ID[(z<=0.3) & (z>=0.2)]
## divide bins
bins = 5
a0 = np.min(sampl_S_ratio)
b0 = np.max(sampl_S_ratio)
bins_S = np.linspace(a0,b0,bins+1)
## record those goal cluster
# select clusters and figure
for k in range(bins):
    eta0 = bins_S[k]
    eta1 = bins_S[k+1]
    R_A_sub = sampl_R_A[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    lamda_sub = sampl_rich[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    id_sub = sampl_ID[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    z_sub = sampl_z[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)] # the four lines use to get the selection sample 
    a1 = np.min(z_sub)
    b1 = np.max(z_sub)
    bins_z = np.linspace(a1,b1,bins+1)
    f = plt.figure(figsize = (50,50))
    f.suptitle(r'$S/S0_{%.3f \sim %.3f}$'%(eta0,eta1), fontsize = 30) # set the figure title
    spc = grid.GridSpec(ncols = 5,nrows = 5,figure = f)
    for p in range(bins):
        z0 = bins_z[p]
        z1 = bins_z[p+1]
        mean_z = (z0 + z1) /2
        try:
            R_A_array = R_A_sub[(z_sub>=z0) & (z_sub<=z1)]
            lamda_array = lamda_sub[(z_sub>=z0) & (z_sub<=z1)]
            id_array = id_sub[(z_sub>=z0) & (z_sub<=z1)]
            '''
            R_A_array = sampl_R_A[(sampl_z>=z0) & (sampl_z<=z1)]
            lamda_array = sampl_rich[(sampl_z>=z0) & (sampl_z<=z1)]
            id_array = sampl_ID[(sampl_z>=z0) & (sampl_z<=z1)]
            '''
            a2 = np.min(lamda_array)
            b2 = np.max(lamda_array)
            bins_L = np.logspace(np.log10(a2),np.log10(b2),bins+1)
            c1 = (bins_L[0] + bins_L[1]) /2
            c2 = (bins_L[1] + bins_L[2]) /2
            c3 = (bins_L[2] + bins_L[3]) /2
            c4 = (bins_L[3] + bins_L[4]) /2
            c5 = (bins_L[4] + bins_L[5]) /2
            k1 = find.find1d(np.abs(lamda_array -c1),np.min( np.abs(lamda_array -c1)))
            k2 = find.find1d(np.abs(lamda_array -c2),np.min( np.abs(lamda_array -c2)))
            k3 = find.find1d(np.abs(lamda_array -c3),np.min( np.abs(lamda_array -c3)))
            k4 = find.find1d(np.abs(lamda_array -c4),np.min( np.abs(lamda_array -c4)))
            k5 = find.find1d(np.abs(lamda_array -c5),np.min( np.abs(lamda_array -c5)))
            # check the goal clusters
            #### first
            di1 = find.find1d(use_ID,id_array[k1])
            ti1 = 0
            di1 = np.array([di1,di1])
            '''
            try:
                di1 = find.find1ds(use_rich,lamda_array[k1]) # find all the lambda value index
                qi1 = use_z[di1] # get the redshift value
                ti1 = find.find1d(np.abs(qi1 -z1),np.min(np.abs(qi1 -z1)))
            except IndexError:
                di1 = find.find1d(use_rich,lamda_array[k1]) # exactly just find one objects 
                qi1 = use_z[di1]
                ti1 = 0
                di1 = np.array([di1,di1])
            '''
            goal1 = di1[ti1] # find the goal cluster information
            ra1 = RA[goal1]
            dec1 = DEC[goal1]
            r1 = R_A_array[k1].value/0.396
            z_ref1 = use_z[goal1]
            # find the satellite and exclude center galaxy
            sum_IA1 = np.sum(rept_ID[1][:goal1])
            poa1 = member_pos[0][sum_IA1:sum_IA1+rept_ID[1][goal1]]
            pob1 = member_pos[1][sum_IA1:sum_IA1+rept_ID[1][goal1]]
            ak1 = find.find1d(poa1,ra1)
            posx1 = poa1[poa1!=poa1[ak1]]
            posy1 = pob1[pob1!=pob1[ak1]]
            clust1 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra1,dec1,z_ref1),header = True)
            #### second
            di2 = find.find1d(use_ID,id_array[k2])
            ti2 = 0
            di2 = np.array([di2,di2])
            '''
            try:
                di2 = find.find1ds(use_rich,lamda_array[k2]) # find all the lambda value index
                qi2 = use_z[di2] # get the redshift value
                ti2 = find.find1d(np.abs(qi2 -z1),np.min(np.abs(qi2 -z1)))
            except IndexError:
                di2 = find.find1d(use_rich,lamda_array[k2]) # exactly just find one objects 
                qi2 = use_z[di2]
                ti2 = 0
                di2 = np.array([di2,di2])
            '''
            goal2 = di2[ti2] # find the goal cluster
            ra2 = RA[goal2]
            dec2 = DEC[goal2]
            r2 = R_A_array[k2].value/0.396
            z_ref2 = use_z[goal2]
            # find the satellite and exclude center galaxy
            sum_IA2 = np.sum(rept_ID[1][:goal2])
            poa2 = member_pos[0][sum_IA2:sum_IA2+rept_ID[1][goal2]]
            pob2 = member_pos[1][sum_IA2:sum_IA2+rept_ID[1][goal2]]
            ak2 = find.find1d(poa2,ra2)
            posx2 = poa2[poa2!=poa2[ak2]]
            posy2 = pob2[pob2!=pob2[ak2]]
            clust2 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra2,dec2,z_ref2),header = True)
            #### third
            di3 = find.find1d(use_ID,id_array[k3])
            ti3 = 0
            di3 = np.array([di3,di3])
            '''
            try:
                di3 = find.find1ds(use_rich,lamda_array[k3]) # find all the lambda value index
                qi3 = use_z[di3] # get the redshift value
                ti3 = find.find1d(np.abs(qi3 -z1),np.min(np.abs(qi3 -z1)))
            except IndexError:
                di3 = find.find1d(use_rich,lamda_array[k3]) # exactly just find one objects 
                qi3 = use_z[di3]
                ti3 = 0
                di3 = np.array([di3,di3])
            '''
            goal3 = di3[ti3] # find the goal cluster
            ra3 = RA[goal3]
            dec3 = DEC[goal3]
            r3 = R_A_array[k3].value/0.396
            z_ref3 = use_z[goal3]
            # find the satellite and exclude center galaxy
            sum_IA3 = np.sum(rept_ID[1][:goal3])
            poa3 = member_pos[0][sum_IA3:sum_IA3+rept_ID[1][goal3]]
            pob3 = member_pos[1][sum_IA3:sum_IA3+rept_ID[1][goal3]]
            ak3 = find.find1d(poa3,ra3)
            posx3 = poa3[poa3!=poa3[ak3]]
            posy3 = pob3[pob3!=pob3[ak3]]
            clust3 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra3,dec3,z_ref3),header = True)
            #### fourth
            di4 = find.find1d(use_ID,id_array[k4])
            ti4 = 0
            di4 = np.array([di4,di4])
            '''
            try:
                di4 = find.find1ds(use_rich,lamda_array[k4]) # find all the lambda value index
                qi4 = use_z[di4] # get the redshift value
                ti4 = find.find1d(np.abs(qi4 -z1),np.min(np.abs(qi4 -z1)))
            except IndexError:
                di4 = find.find1d(use_rich,lamda_array[k4]) # exactly just find one objects 
                qi4 = use_z[di4]
                ti4 = 0
                di4 = np.array([di4,di4])
            '''
            goal4 = di4[ti4] # find the goal cluster
            ra4 = RA[goal4]
            dec4 = DEC[goal4]
            r4 = R_A_array[k4].value/0.396
            z_ref4 = use_z[goal4]
            # find the satellite and exclude center galaxy
            sum_IA4 = np.sum(rept_ID[1][:goal4])
            poa4 = member_pos[0][sum_IA4:sum_IA4+rept_ID[1][goal4]]
            pob4 = member_pos[1][sum_IA4:sum_IA4+rept_ID[1][goal4]]
            ak4 = find.find1d(poa4,ra4)
            posx4 = poa4[poa4!=poa4[ak4]]
            posy4 = pob4[pob4!=pob4[ak4]]
            clust4 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra4,dec4,z_ref4),header = True)
            #### fifth
            di5 = find.find1d(use_ID,id_array[k5])
            ti5 = 0
            di5 = np.array([di5,di5])
            '''
            try:
                di5 = find.find1ds(use_rich,lamda_array[k5]) # find all the lambda value index
                qi5 = use_z[di5] # get the redshift value
                ti5 = find.find1d(np.abs(qi5 -z1),np.min(np.abs(qi5 -z1)))
            except IndexError:
                di5 = find.find1d(use_rich,lamda_array[k5]) # exactly just find one objects 
                qi5 = use_z[di5]
                ti5 = 0
                di5 = np.array([di5,di5])
            '''
            goal5 = di5[ti5] # find the goal cluster
            ra5 = RA[goal5]
            dec5 = DEC[goal5]
            r5 = R_A_array[k5].value/0.396
            z_ref5 = use_z[goal5]
            # find the satellite and exclude center galaxy
            sum_IA5 = np.sum(rept_ID[1][:goal5])
            poa5 = member_pos[0][sum_IA5:sum_IA5+rept_ID[1][goal5]]
            pob5 = member_pos[1][sum_IA5:sum_IA5+rept_ID[1][goal5]]
            ak5 = find.find1d(poa5,ra5)
            posx5 = poa5[poa5!=poa5[ak5]]
            posy5 = pob5[pob5!=pob5[ak5]]
            clust5 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra5,dec5,z_ref5),header = True)
            #### figure out the select 
            wcs1 = awc.WCS(clust1[1])
            ax1 = f.add_subplot(spc[p,0], projection = wcs1)
            ax01 = ax1.axes.coords[0]
            ay01 = ax1.axes.coords[1]
            ax01.set_ticks(spacing = 0.05*U.deg)
            ax01.set_ticklabel(color = 'red')
            ax01.grid(color = 'red',alpha = 0.5)
            ay01.set_ticks(spacing = 0.05*U.deg) 
            ay01.set_ticklabel(color = 'green')
            ay01.grid(color = 'green',alpha = 0.5)
            ax01.set_major_formatter('d.dd')
            ay01.set_major_formatter('d.dd')
            tx1, ty1 = wcs1.all_world2pix(posx1*U.deg, posy1*U.deg, 1)
            x01, y01 = wcs1.all_world2pix(ra1*U.deg, dec1*U.deg, 1)
            im1 = ax1.imshow(clust1[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',
                       norm=mpl.colors.LogNorm())
            lima1 = ax1.get_xlim()
            limb1 = ax1.get_ylim()
            hsc.circles(x01,y01,s = r1,fc = '',ec = 'r')
            ax1.scatter(x01,y01, facecolors = '', marker = 'P',edgecolors = 'b')
            ax1.scatter(tx1,ty1,facecolors = '',marker = 'o',edgecolors = 'r')
            ax1.set_title(
                    '$Ra_{%.3f} \ Dec_{%.3f} \ \lambda_{%.3f} \ z_{%.3f}$ \n $[%.3f \sim \lambda \sim %.3f][%.3f \sim z \sim%.3f]$'
                          %(ra1,dec1,lamda_array[k1],z_ref1,bins_L[0],bins_L[1],z0,z1),fontsize = 20)
            #ax1.set_ylabel(r'$\lambda_{%.3f \sim %.3f} \ Z_{%.3f \sim %.3f}$'%
            #               (bins_L[0],bins_L[1],z0,z1),fontsize = 20)
            ax1.set_xticklabels(labels = [], fontsize = 10)
            ax1.set_yticklabels(labels = [], fontsize = 10)
            ax1.set_xlim(lima1[0],lima1[1])
            ax1.set_ylim(limb1[0],limb1[1])
            ax1.axes.set_aspect('equal')
            co1 = plt.colorbar(im1, label='flux', fraction = 0.035,pad = 0.001)
            co1.set_ticks(np.logspace(-5,2,6))
            
            wcs2 = awc.WCS(clust2[1])            
            ax2 = f.add_subplot(spc[p,1], projection = wcs2)
            ax02 = ax2.axes.coords[0]
            ay02 = ax2.axes.coords[1]
            ax02.set_ticks(spacing = 0.05*U.deg)
            ax02.set_ticklabel(color = 'red')
            ax02.grid(color = 'red',alpha = 0.5)
            ay02.set_ticks(spacing = 0.05*U.deg) 
            ay02.set_ticklabel(color = 'green')
            ay02.grid(color = 'green',alpha = 0.5)
            ax02.set_major_formatter('d.dd')
            ay02.set_major_formatter('d.dd')
            tx2, ty2 = wcs2.all_world2pix(posx2*U.deg, posy2*U.deg, 1)
            x02, y02 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)
            im2 = ax2.imshow(clust2[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',
                       norm=mpl.colors.LogNorm())
            lima2 = ax2.get_xlim()
            limb2 = ax2.get_ylim()
            hsc.circles(x02,y02,s = r2,fc = '',ec = 'r')
            ax2.scatter(x02,y02, facecolors = '', marker = 'P',edgecolors = 'b')
            ax2.scatter(tx2,ty2,facecolors = '',marker = 'o',edgecolors = 'r')
            ax2.set_title(r'$Ra_{%.3f} \ Dec_{%.3f} \ \lambda_{%.3f} \ z_{%.3f} [%.3f \sim \lambda \sim %.3f]$'
                          %(ra2,dec2,lamda_array[k2],z_ref2,bins_L[1],bins_L[2]),fontsize = 20)
            #ax2.set_ylabel(r'$\lambda_{%.3f \sim %.3f} \ Z_{%.3f \sim %.3f}$'%
            #               (bins_L[1],bins_L[2],z0,z1),fontsize = 20)
            ax2.set_xticklabels(labels = [], fontsize = 10)
            ax2.set_yticklabels(labels = [], fontsize = 10)
            ax2.set_xlim(lima2[0],lima2[1])
            ax2.set_ylim(limb2[0],limb2[1])
            ax2.axes.set_aspect('equal')
            co2 = plt.colorbar(im2, label='flux', fraction = 0.035,pad = 0.001)
            co2.set_ticks(np.logspace(-5,2,6))
            
            wcs3 = awc.WCS(clust3[1])
            ax3 = f.add_subplot(spc[p,2], projection = wcs3)
            ax03 = ax3.axes.coords[0]
            ay03 = ax3.axes.coords[1]
            ax03.set_ticks(spacing = 0.05*U.deg)
            ax03.set_ticklabel(color = 'red')
            ax03.grid(color = 'red',alpha = 0.5)
            ay03.set_ticks(spacing = 0.05*U.deg) 
            ay03.set_ticklabel(color = 'green')
            ay03.grid(color = 'green',alpha = 0.5)
            ax03.set_major_formatter('d.dd')
            ay03.set_major_formatter('d.dd')
            tx3, ty3 = wcs3.all_world2pix(posx3*U.deg, posy3*U.deg, 1)
            x03, y03 = wcs3.all_world2pix(ra3*U.deg, dec3*U.deg, 1)
            im3 = ax3.imshow(clust3[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',
                       norm=mpl.colors.LogNorm())
            lima3 = ax3.get_xlim()
            limb3 = ax3.get_ylim()
            hsc.circles(x03,y03,s = r3,fc = '',ec = 'r')
            ax3.scatter(x03,y03, facecolors = '', marker = 'P',edgecolors = 'b')
            ax3.scatter(tx3,ty3,facecolors = '',marker = 'o',edgecolors = 'r')
            ax3.set_title(r'$Ra_{%.3f} \ Dec_{%.3f} \ \lambda_{%.3f} \ z_{%.3f} [%.3f \sim \lambda \sim %.3f]$'
                          %(ra3,dec3,lamda_array[k3],z_ref3,bins_L[2],bins_L[3]),fontsize = 20)
            #ax3.set_ylabel(r'$\lambda_{%.3f \sim %.3f} \ Z_{%.3f \sim %.3f}$'%
            #               (bins_L[2],bins_L[3],z0,z1),fontsize = 20)
            ax3.set_xticklabels(labels = [], fontsize = 10)
            ax3.set_yticklabels(labels = [], fontsize = 10)
            ax3.set_xlim(lima3[0],lima3[1])
            ax3.set_ylim(limb3[0],limb3[1])
            ax3.axes.set_aspect('equal')
            co3 = plt.colorbar(im3, label='flux', fraction = 0.035,pad = 0.001)
            co3.set_ticks(np.logspace(-5,2,6))
            
            wcs4 = awc.WCS(clust4[1])
            ax4 = f.add_subplot(spc[p,3], projection = wcs4)
            ax04 = ax4.axes.coords[0]
            ay04 = ax4.axes.coords[1]
            ax04.set_ticks(spacing = 0.05*U.deg)
            ax04.set_ticklabel(color = 'red')
            ax04.grid(color = 'red',alpha = 0.5)
            ay04.set_ticks(spacing = 0.05*U.deg) 
            ay04.set_ticklabel(color = 'green')
            ay04.grid(color = 'green',alpha = 0.5)
            ax04.set_major_formatter('d.dd')
            ay04.set_major_formatter('d.dd')
            tx4, ty4 = wcs4.all_world2pix(posx4*U.deg, posy4*U.deg, 1)
            x04, y04 = wcs4.all_world2pix(ra4*U.deg, dec4*U.deg, 1)
            im4 = ax4.imshow(clust4[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',
                       norm=mpl.colors.LogNorm())
            lima4 = ax4.get_xlim()
            limb4 = ax4.get_ylim()
            hsc.circles(x04,y04,s = r4,fc = '',ec = 'r')
            ax4.scatter(x04,y04, facecolors = '', marker = 'P',edgecolors = 'b')
            ax4.scatter(tx4,ty4,facecolors = '',marker = 'o',edgecolors = 'r')
            ax4.set_title(r'$Ra_{%.3f} \ Dec_{%.3f} \ \lambda_{%.3f} \ z_{%.3f} [%.3f \sim \lambda \sim %.3f]$'
                          %(ra4,dec4,lamda_array[k4],z_ref4,bins_L[3],bins_L[4]),fontsize = 20)
            #ax4.set_ylabel(r'$\lambda_{%.3f \sim %.3f} \ Z_{%.3f \sim %.3f}$'%
            #               (bins_L[3],bins_L[4],z0,z1),fontsize = 20)
            ax4.set_xticklabels(labels = [], fontsize = 10)
            ax4.set_yticklabels(labels = [], fontsize = 10)
            ax4.set_xlim(lima4[0],lima4[1])
            ax4.set_ylim(limb4[0],limb4[1])
            ax4.axes.set_aspect('equal')
            co4 = plt.colorbar(im4, label='flux', fraction = 0.035,pad = 0.001)
            co4.set_ticks(np.logspace(-5,2,6))
            
            wcs5 = awc.WCS(clust5[1])
            ax5 = f.add_subplot(spc[p,4], projection = wcs5)
            ax05 = ax5.axes.coords[0]
            ay05 = ax5.axes.coords[1]
            ax05.set_ticks(spacing = 0.05*U.deg)
            ax05.set_ticklabel(color = 'red')
            ax05.grid(color = 'red',alpha = 0.5)
            ay05.set_ticks(spacing = 0.05*U.deg) 
            ay05.set_ticklabel(color = 'green')
            ay05.grid(color = 'green',alpha = 0.5)
            ax05.set_major_formatter('d.dd')
            ay05.set_major_formatter('d.dd')
            tx5, ty5 = wcs5.all_world2pix(posx5*U.deg, posy5*U.deg, 1)
            x05, y05 = wcs5.all_world2pix(ra5*U.deg, dec5*U.deg, 1)
            im5 = ax5.imshow(clust5[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,origin = 'lower',
                       norm=mpl.colors.LogNorm())
            lima5 = ax5.get_xlim()
            limb5 = ax5.get_ylim()
            hsc.circles(x05,y05,s = r5,fc = '',ec = 'r')
            ax5.scatter(x05,y05, facecolors = '', marker = 'P',edgecolors = 'b')
            ax5.scatter(tx5,ty5,facecolors = '',marker = 'o',edgecolors = 'r')
            ax5.set_title(r'$Ra_{%.3f} \ Dec_{%.3f} \ \lambda_{%.3f} \ z_{%.3f} [%.3f \sim \lambda \sim %.3f]$'
                          %(ra5,dec5,lamda_array[k5],z_ref5,bins_L[4],bins_L[5]),fontsize = 20)
            #ax5.set_ylabel(r'$\lambda_{%.3f \sim %.3f} \ Z_{%.3f \sim %.3f}$'%
            #               (bins_L[4],bins_L[5],z0,z1),fontsize = 20)
            ax5.set_xticklabels(labels = [], fontsize = 10)
            ax5.set_yticklabels(labels = [], fontsize = 10)
            ax5.set_xlim(lima5[0],lima5[1])
            ax5.set_ylim(limb5[0],limb5[1])
            ax5.axes.set_aspect('equal')
            co5 = plt.colorbar(im5, label='flux', fraction = 0.035,pad = 0.001)
            co5.set_ticks(np.logspace(-5,2,6))
        
        except ValueError:
            continue
    plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_summary/S_S0_z_richness%.0f.pdf'%k,dpi=600)
    plt.close()
    