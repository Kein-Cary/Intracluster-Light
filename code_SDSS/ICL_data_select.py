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

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
goal_data = aft.getdata(load + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
ID = np.array(goal_data.ID)
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
com_ID = ID[(redshift >= 0.2) & (redshift <= 0.3)]

band = ['r', 'g', 'i', 'u', 'z']
zN, bN = len(com_z), len(band)

for kk in range(bN):
    ## bad images
    with h5py.File(load + 'mpi_h5/Except_%s_sample.h5' % band[kk], 'r') as f:
        except_cat = np.array(f['a'])
    except_ra = ['%.3f' % ll for ll in except_cat[0,:] ]
    except_dec = ['%.3f' % ll for ll in except_cat[1,:] ]
    #except_z = ['%.3f' % ll for ll in except_cat[2,:] ]

    ## special mask
    with h5py.File(load + 'mpi_h5/special_mask_cat.h5', 'r') as f:
        special_cat = np.array(f['a'])
    speci_ra = ['%.3f' % ll for ll in special_cat[0,:] ]
    speci_dec = ['%.3f' % ll for ll in special_cat[1,:] ]
    #speci_z = ['%.3f' % ll for ll in special_cat[2,:] ]

    sub_z = []
    sub_ra = []
    sub_dec = []
    sub_rich = []
    sub_ID = []
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
        ID_g = com_ID[jj]

        r_mag, r_err = com_r_Mag[jj], com_r_Mag_err[jj]
        g_mag, g_err = com_g_Mag[jj], com_g_Mag_err[jj]
        i_mag, i_err = com_i_Mag[jj], com_i_Mag_err[jj]
        u_mag, u_err = com_u_Mag[jj], com_u_Mag_err[jj]
        z_mag, z_err = com_z_Mag[jj], com_z_Mag_err[jj]

        ## rule out bad image (once a cluster image is bad in a band, all the five band image will be ruled out!)
        identi0 = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec)# & ('%.3f'%z_g in except_z) ## use this only for "img_data_select"
        identi1 = ('%.3f'%ra_g in speci_ra) & ('%.3f'%dec_g in speci_dec)# & ('%.3f'%z_g in speci_z)
        if  identi0 == True: 
            continue
        elif identi1 == True:
            continue
        else:
            sub_z.append(z_g)
            sub_ra.append(ra_g)
            sub_dec.append(dec_g)
            sub_rich.append(rich_g)
            sub_ID.append(ID_g)

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
    sub_ID = np.array(sub_ID)

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
            'r_Mag_err', 'g_Mag_err', 'i_Mag_err', 'u_Mag_err', 'z_Mag_err', 'CAT_ID']
    values = [sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
            sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr, sub_ID]
    fill = dict(zip(keys, values))
    data = pds.DataFrame(fill)
    #data.to_csv(load + 'selection/%s_band_img_data_select.csv' % band[kk])
    data.to_csv(load + 'selection/%s_band_stack_catalog.csv' % band[kk]) ## also rule out special_mask sample

    ## save h5py for mpirun
    sub_array = np.array([sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
            sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr])
    #with h5py.File(load + 'mpi_h5/%s_band_img_data_select.h5' % band[kk], 'w') as f:
    with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'w') as f:
        f['a'] = np.array(sub_array)
    #with h5py.File(load + 'mpi_h5/%s_band_img_data_select.h5' % band[kk]) as f:
    with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk]) as f:
        for tt in range( len(sub_array) ):
            f['a'][tt,:] = sub_array[tt,:]

########################################################
# Based on the image data sample, seletcing the sky image sample
##### selected sky sample
for kk in range(bN):

    data_cat = pds.read_csv(load + 'selection/%s_band_stack_catalog.csv' % band[kk])
    com_z = data_cat['z']
    com_ra = data_cat['ra']
    com_dec = data_cat['dec']
    com_rich = data_cat['rich']
    com_id = data_cat['CAT_ID']

    com_r_Mag = data_cat['r_Mag']
    com_r_Mag_err = data_cat['r_Mag_err']

    com_g_Mag = data_cat['g_Mag']
    com_g_Mag_err = data_cat['g_Mag_err']

    com_i_Mag = data_cat['i_Mag']
    com_i_Mag_err = data_cat['i_Mag_err']

    com_u_Mag = data_cat['u_Mag']
    com_u_Mag_err = data_cat['u_Mag_err']

    com_z_Mag = data_cat['z_Mag']
    com_z_Mag_err = data_cat['z_Mag_err']

    ## sky rule out catalogue
    with h5py.File(load + 'mpi_h5/sky_rule_out_cat.h5', 'r') as f:
        sky_cat = np.array(f['a'])
    sky_ra = ['%.3f' % ll for ll in sky_cat[0,:] ]
    sky_dec = ['%.3f' % ll for ll in sky_cat[1,:] ]
    #sky_z = ['%.3f' % ll for ll in sky_cat[2,:] ]

    sub_z = []
    sub_ra = []
    sub_dec = []
    sub_rich = []
    sub_id = []

    sub_r_mag, sub_r_Merr = [], []
    sub_g_mag, sub_g_Merr = [], []
    sub_i_mag, sub_i_Merr = [], []
    sub_u_mag, sub_u_Merr = [], []
    sub_z_mag, sub_z_Merr = [], []

    zN = len(com_z)

    for jj in range(zN):
        ra_g = com_ra[jj]
        dec_g = com_dec[jj]
        z_g = com_z[jj]
        rich_g = com_rich[jj]
        id_g = com_id[jj]

        r_mag, r_err = com_r_Mag[jj], com_r_Mag_err[jj]
        g_mag, g_err = com_g_Mag[jj], com_g_Mag_err[jj]
        i_mag, i_err = com_i_Mag[jj], com_i_Mag_err[jj]
        u_mag, u_err = com_u_Mag[jj], com_u_Mag_err[jj]
        z_mag, z_err = com_z_Mag[jj], com_z_Mag_err[jj]

        identi = ('%.3f'%ra_g in sky_ra) & ('%.3f'%dec_g in sky_dec)# & ('%.3f'%z_g in sky_z)
        if  identi == True: 
            continue
        else:
            sub_z.append(z_g)
            sub_ra.append(ra_g)
            sub_dec.append(dec_g)
            sub_rich.append(rich_g)
            sub_id.append(id_g)

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
    sub_id = np.array(sub_id)

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
            'r_Mag_err', 'g_Mag_err', 'i_Mag_err', 'u_Mag_err', 'z_Mag_err', 'CAT_ID']
    values = [sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
            sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr, sub_id]
    fill = dict(zip(keys, values))
    data = pds.DataFrame(fill)
    data.to_csv(load + 'selection/%s_band_sky_catalog.csv' % band[kk])
    ## save h5py for mpirun
    sub_array = np.array([sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag,
            sub_r_Merr, sub_g_Merr, sub_i_Merr, sub_u_Merr, sub_z_Merr])
    with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'w') as f:
        f['a'] = np.array(sub_array)
    with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk]) as f:
        for tt in range( len(sub_array) ):
            f['a'][tt,:] = sub_array[tt,:]

print('done!')
