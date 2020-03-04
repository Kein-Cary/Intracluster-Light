import h5py
import numpy as np
import pandas as pds

def phot_z_rule_out():
	csv_1 = 'photo_z_Except_r_sample.csv'
	csv_2 = 'photo_z_Except_g_sample.csv'
	csv_3 = 'photo_z_Except_i_sample.csv'

	except_ra_r = [ '34.973',  '252.554', '246.426', '207.646', '129.406', '139.099', '351.802',
					'352.107', '128.996', '214.717', '16.278',  '359.055', '26.113',  '346.410', 
					'126.625', '10.912',  '30.057',  '114.941', '223.140', '12.748',  '134.888', 
					'191.188', '221.385', '249.109', '133.700', '350.748', '248.012', '233.085', 
					'13.471',  '223.495', '355.128', '345.588', '162.936', '320.771', '150.984', 
					'346.197', '253.513', '126.625', '352.919', '30.057',  '20.490',  '143.162', 
					'177.961', '27.776',  '15.793',  '127.879', '30.705',  ]

	except_dec_r = ['2.785',  '51.060', '51.761', '5.321',  '15.083', '52.906', '31.133', 
					'-5.446', '15.592', '39.966', '34.623', '22.743', '13.926', '23.084', 
					'24.725', '9.838',  '-3.483', '34.678', '40.806', '25.484', '16.546', 
					'53.856', '13.735', '20.521', '8.789',  '33.946', '21.411', '40.801', 
					'29.638', '-0.253', '9.749',  '24.659', '8.767',  '-5.672', '12.181', 
					'-7.609', '29.490', '24.725', '23.721', '-3.483', '12.758', '23.016', 
					'52.214', '27.592', '26.353', '37.158', '-4.036', ]

	except_z_r = [  '0.277', '0.262', '0.299', '0.241', '0.267', '0.201', '0.297', 
					'0.287', '0.295', '0.299', '0.217', '0.214', '0.206', '0.220', 
					'0.279', '0.277', '0.298', '0.247', '0.287', '0.265', '0.275', 
					'0.266', '0.247', '0.252', '0.279', '0.257', '0.240', '0.277', 
					'0.289', '0.216', '0.295', '0.242', '0.292', '0.277', '0.287', 
					'0.299', '0.214', '0.279', '0.296', '0.298', '0.275', '0.272', 
					'0.285', '0.250', '0.248', '0.204', '0.293', ]

	x_ra = np.array( [ np.float(ll) for ll in except_ra_r] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_r] )
	x_z = np.array( [np.float(ll) for ll in except_z_r] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(csv_1)

	except_ra_g = [ '139.473', '34.973',  '252.554', '246.426', '207.646', '129.406', '139.099',
					'351.802', '352.107', '128.996', '214.717', '16.278',  '359.055', '346.410', 
					'126.625', '134.572', '30.057',  '114.941', '223.140', '12.748',  '134.888', 
					'147.765', '221.385', '249.109', '133.700', '350.748', '248.012', '233.085', 
					'13.471',  '223.495', '355.128', '345.588', '320.771', '346.197', '126.625', 
					'352.919', '30.057',  '143.162', '177.961', '15.793',  '127.879', ]

	except_dec_g = ['51.727', '2.785',  '51.060', '51.761', '5.321',  '15.083', '52.906', 
					'31.133', '-5.446', '15.592', '39.966', '34.623', '22.743', '23.084', 
					'24.725', '48.522', '-3.483', '34.678', '40.806', '25.484', '16.546', 
					'45.454', '13.735', '20.521', '8.789',  '33.946', '21.411', '40.801', 
					'29.638', '-0.253', '9.749',  '24.659', '-5.672', '-7.609', '24.725', 
					'23.721', '-3.483', '23.016', '52.214', '26.353', '37.158', ]

	except_z_g = [  '0.227', '0.277', '0.262', '0.299', '0.241', '0.267', '0.201', 
					'0.297', '0.287', '0.295', '0.299', '0.217', '0.214', '0.220', 
					'0.279', '0.292', '0.298', '0.247', '0.287', '0.265', '0.275', 
					'0.269', '0.247', '0.252', '0.279', '0.257', '0.240', '0.277', 
					'0.289', '0.216', '0.295', '0.242', '0.277', '0.299', '0.279', 
					'0.296', '0.298', '0.272', '0.285', '0.248', '0.204', ]

	x_ra = np.array( [ np.float(ll) for ll in except_ra_g] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_g] )
	x_z = np.array( [np.float(ll) for ll in except_z_g] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(csv_2)

	except_ra_i = [ '34.973',  '252.554', '246.426', '207.646', '129.406', '139.099', '351.802', 
					'352.107', '128.996', '214.717', '16.278',  '359.055', '201.144', '26.113',  
					'346.410', '126.625', '30.057',  '114.941', '14.803',  '223.140', '12.748', 
					'134.888', '221.385', '249.109', '350.748', '248.012', '233.085', '13.471', 
					'24.881',  '24.514',  '223.495', '355.128', '345.588', '320.771', '150.984', 
					'346.197', '253.513', '126.625', '352.919', '30.057',  '20.490',  '143.162', 
					'177.961', '27.776',  '15.793',  '27.208',  '10.708',  '111.749', '11.871',  
					'240.586', '20.656',  '30.705',  ]

	except_dec_i = ['2.785',  '51.060', '51.761', '5.321',  '15.083', '52.906', '31.133', 
					'-5.446', '15.592', '39.966', '34.623', '22.743', '50.142', '13.926', 
					'23.084', '24.725', '-3.483', '34.678', '19.208', '40.806', '25.484', 
					'16.546', '13.735', '20.521', '33.946', '21.411', '40.801', '29.638', 
					'-1.608', '-3.635', '-0.253', '9.749',  '24.659', '-5.672', '12.181', 
					'-7.609', '29.490', '24.725', '23.721', '-3.483', '12.758', '23.016', 
					'52.214', '27.592', '26.353', '23.029', '22.239', '35.567', '-6.988', 
					'56.446', '34.660', '-4.036', ]

	except_z_i = [  '0.277', '0.262', '0.299', '0.241', '0.267', '0.201', '0.297', 
					'0.287', '0.295', '0.299', '0.217', '0.214', '0.277', '0.206', 
					'0.220', '0.279', '0.298', '0.247', '0.277', '0.287', '0.265', 
					'0.275', '0.247', '0.252', '0.257', '0.240', '0.277', '0.289', 
					'0.251', '0.237', '0.216', '0.295', '0.242', '0.277', '0.287', 
					'0.299', '0.214', '0.279', '0.296', '0.298', '0.275', '0.272', 
					'0.285', '0.250', '0.248', '0.206', '0.248', '0.293', '0.215', 
					'0.261', '0.295', '0.293', ]

	x_ra = np.array( [ np.float(ll) for ll in except_ra_i] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_i] )
	x_z = np.array( [np.float(ll) for ll in except_z_i] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(csv_3)

	return

def phot_z_use_sample():
	band = ['r', 'g', 'i']
	file = 'photo_z_difference_sample.csv'
	cat = pds.read_csv(file)
	com_ra, com_dec, com_z, com_rich = np.array(cat.ra), np.array(cat.dec), np.array(cat.z), np.array(cat.rich)
	( com_r_Mag, com_g_Mag, com_i_Mag, com_u_Mag, com_z_Mag ) = (
	np.array(cat.r_Mag), np.array(cat.g_Mag), np.array(cat.i_Mag), np.array(cat.u_Mag), np.array(cat.z_Mag) )
	zN, bN = len(com_z), len(band)

	for kk in range(bN):
		## bad images
		sub_cat = pds.read_csv('photo_z_Except_%s_sample.csv' % band[kk])
		tx_ra, tx_dec, tx_z = np.array(sub_cat.ra), np.array(sub_cat.dec), np.array(sub_cat.z)

		except_ra = ['%.3f' % ll for ll in tx_ra ]
		except_dec = ['%.3f' % ll for ll in tx_dec ]
		#except_z = ['%.3f' % ll for ll in tx_z ]

		sub_z = []
		sub_ra = []
		sub_dec = []
		sub_rich = []
		sub_r_mag = []
		sub_g_mag = []
		sub_i_mag = []
		sub_u_mag = []
		sub_z_mag = []

		for jj in range(zN):
			ra_g = com_ra[jj]
			dec_g = com_dec[jj]
			z_g = com_z[jj]
			rich_g = com_rich[jj]

			r_mag = com_r_Mag[jj]
			g_mag = com_g_Mag[jj]
			i_mag = com_i_Mag[jj]
			u_mag = com_u_Mag[jj]
			z_mag = com_z_Mag[jj]

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

		sub_z = np.array(sub_z)
		sub_ra = np.array(sub_ra)
		sub_dec = np.array(sub_dec)
		sub_rich = np.array(sub_rich)

		sub_r_mag = np.array(sub_r_mag)
		sub_g_mag = np.array(sub_g_mag)
		sub_i_mag = np.array(sub_i_mag)
		sub_u_mag = np.array(sub_u_mag)
		sub_z_mag = np.array(sub_z_mag)

		## save the csv file
		keys = ['ra', 'dec', 'z', 'rich', 'r_Mag', 'g_Mag', 'i_Mag', 'u_Mag', 'z_Mag']
		values = [sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv('phot_z_%s_band_stack_cat.csv' % band[kk])

		## save h5py for mpirun
		sub_array = np.array([sub_ra, sub_dec, sub_z, sub_rich, sub_r_mag, sub_g_mag, sub_i_mag, sub_u_mag, sub_z_mag])
		with h5py.File('phot_z_%s_band_stack_cat.h5' % band[kk], 'w') as f:
			f['a'] = np.array(sub_array)

		with h5py.File('phot_z_%s_band_stack_cat.h5' % band[kk]) as f:
			for tt in range( len(sub_array) ):
				f['a'][tt,:] = sub_array[tt,:]

	return

def main():
	#phot_z_rule_out()
	phot_z_use_sample()

if __name__ == "__main__":
	main()
