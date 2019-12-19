import h5py
import numpy as np
import pandas as pds

def data_select():
	lod = '/home/xkchen/mywork/ICL/data/redmapper/'
	csv_1 = 'Except_r_sample.csv'
	csv_2 = 'Except_g_sample.csv'
	csv_3 = 'Except_i_sample.csv'
	csv_4 = 'Except_z_sample.csv'
	csv_5 = 'Except_u_sample.csv'

	except_ra_r = [ '1.980', '1.721', '7.110', '8.320', '17.260', '32.906', '34.535', '38.382',
					'39.468', '116.612', '128.611', '130.537', '135.374', '136.374', '147.185',
					'153.728', '158.222', '162.079', '168.057', '168.650', '172.033', '177.809',
					'182.398', '183.061', '183.580', '186.684', '202.923', '208.380', '210.752',
					'210.871', '212.342', '221.403', '221.731', '224.878', '237.591', '242.344',
					'248.482', '250.342', '255.705', '330.691', '351.680', '354.643', '180.927',
					'154.207', '233.938', '133.229', '10.814',  '186.372', '231.786', '12.742',  
					'190.946', '145.632', '223.211', '18.302',  '238.323', '37.662' , '145.632', 
					'186.372']

	except_dec_r = ['-8.845', '-10.586', '6.724', '-0.633', '15.237', '0.387', '-6.379', '1.911',
					'-4.182', '48.404', '36.515', '38.226', '61.733', '5.180', '27.627',
					'22.351', '40.602', '38.948', '15.760', '15.610', '41.322', '33.577',
					'38.482', '65.307', '32.835', '48.232', '13.325', '53.575', '22.709',
					'22.397', '39.952', '8.243',  '31.858', '42.382', '2.756',  '36.393',
					'43.091', '24.974', '31.842', '-8.544', '1.134',  '-1.781', '1.032', 
					'21.965', '0.083',  '18.196', '2.531',  '0.726',  '50.305', '-9.490', 
					'51.831', '30.025', '15.746', '25.084', '10.319', '-4.991', '30.025', 
					'0.726' ]

	except_z_r = [	'0.260', '0.291', '0.264', '0.261', '0.296', '0.295', '0.236', '0.243',
					'0.250', '0.213', '0.289', '0.244', '0.298', '0.258', '0.237',
					'0.257', '0.257', '0.287', '0.275', '0.274', '0.288', '0.212',
					'0.264', '0.247', '0.226', '0.261', '0.242', '0.225', '0.221',
					'0.224', '0.267', '0.268', '0.247', '0.286', '0.240', '0.281',
					'0.270', '0.274', '0.273', '0.210', '0.277', '0.251', '0.255',
					'0.218', '0.295', '0.257', '0.262', '0.238', '0.278', '0.200', 
					'0.268', '0.271', '0.297', '0.255', '0.226', '0.292', '0.271', 
					'0.238']

	x_ra = np.array( [ np.float(ll) for ll in except_ra_r] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_r] )
	x_z = np.array( [np.float(ll) for ll in except_z_r] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_1)

	except_ra_g = [ '1.980',   '8.320',   '10.814',  '15.970',  '18.217',  '29.544',  '34.675',  '35.888',
					'37.662',  '113.119', '121.566', '124.701', '128.611', '130.537', '133.229', '136.374',
					'138.569', '145.632', '150.040', '154.207', '154.265', '158.222', '160.149', '162.793',
					'168.215', '168.650', '172.033', '177.809', '179.842', '180.927', '181.262', '183.580',
					'186.666', '188.950', '190.946', '208.380', '210.871', '211.215', '215.244', '216.308',
					'221.403', '224.024', '226.406', '227.367', '233.938', '238.323', '242.344', '250.342',
					'255.705', '330.691', '340.792', '351.680', '10.814',  '186.372', '231.786', '12.742',  
					'190.946', '145.632', '223.211', '18.302' , '238.323', '37.662' , '145.632', '186.372']

	except_dec_g = ['-8.845', '-0.633', '2.531', '-0.960', '3.247',  '6.212',  '0.114',   '-7.228',
					'-4.991', '37.617', '19.247', '4.898',  '36.515', '38.226', '18.196', '5.180',
					'3.203',  '30.025', '17.588', '21.965', '39.047', '40.602', '4.409',  '9.066',
					'56.464', '15.610', '41.322', '33.577', '-2.627', '1.032',  '11.653', '32.835',
					'3.383',  '15.556', '51.831', '53.575', '22.397', '47.084', '0.229',  '7.192',
					'8.243',  '16.449', '51.449', '0.899',  '0.083',  '10.319', '36.393', '24.974',
					'31.842', '-8.544', '-9.211', '1.134',  '2.531',  '0.726',  '50.305', '-9.490', 
					'51.831', '30.025', '15.746', '25.084', '10.319', '-4.991', '30.025', '0.726' ]

	except_z_g = [  '0.260', '0.261', '0.262', '0.208', '0.261', '0.298', '0.272', '0.279',
					'0.292', '0.203', '0.285', '0.252', '0.289', '0.244', '0.257', '0.258',
					'0.231', '0.271', '0.291', '0.218', '0.206', '0.257', '0.272', '0.221',
					'0.227', '0.274', '0.288', '0.212', '0.213', '0.255', '0.261', '0.226',
					'0.226', '0.285', '0.268', '0.225', '0.224', '0.238', '0.277', '0.284',
					'0.268', '0.287', '0.268', '0.263', '0.295', '0.226', '0.281', '0.274',
					'0.273', '0.210', '0.266', '0.277', '0.262', '0.238', '0.278', '0.200', 
					'0.268', '0.271', '0.297', '0.255', '0.226', '0.292', '0.271', '0.238']

	x_ra = np.array( [ np.float(ll) for ll in except_ra_g] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_g] )
	x_z = np.array( [np.float(ll) for ll in except_z_g] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_2)

	except_ra_i = [ '5.200',   '8.320',   '10.708',  '10.720',  '12.742',  '21.224',  '22.289', 
					'24.320',  '32.425',  '37.108',  '37.662',  '43.747',  '117.410', '117.585',
					'126.251', '128.611', '130.537', '131.524', '136.374', '141.919', '143.950',
					'145.632', '148.541', '158.222', '162.793', '173.917', '174.080', '175.435', 
					'175.790', '177.089', '177.809', '179.024', '179.970', '180.927', '183.580',
					'184.271', '186.372', '190.460', '190.946', '196.951', '196.056', '201.264',
					'206.148', '208.380', '210.871', '214.913', '218.217', '218.931', '219.728',
					'221.403', '222.506', '234.949', '235.950', '240.309', '241.192', '242.344',
					'242.452', '250.342', '255.705', '259.786', '330.691', '347.415', '351.680',
					'129.462', '154.207', '233.938', '10.814',  '186.372', '231.786', '12.742',  
					'190.946', '145.632', '223.211', '18.302' , '238.323', '37.662' , '145.632', 
					'186.372']

	except_dec_i = ['1.585',  '-0.633', '0.215',  '0.718',  '-9.490', '2.508',  '3.630', 
					'7.882',  '2.377',  '1.503',  '-4.991', '1.212',  '27.287', '26.293', 
					'4.430',  '36.515', '38.226', '36.948', '5.180',  '4.352',  '12.151',
					'30.025', '36.054', '40.602', '9.066',  '30.155', '50.425', '26.890',
					'26.859', '37.966', '33.577', '42.024', '26.444', '1.032',  '32.835',
					'61.133', '0.726',  '19.782', '51.831', '43.153', '7.031',  '6.481',
					'51.631', '53.575', '22.397', '43.195', '30.489', '43.626', '53.601',
					'8.243',  '14.797', '36.326', '47.502', '49.412', '52.640', '36.393',
					'26.578', '24.974', '31.842', '30.735', '-8.544', '-3.423', '1.134', 
					'6.777',  '21.965', '0.083',  '2.531',  '0.726',  '50.305', '-9.490', 
					'51.831', '30.025', '15.746', '25.084', '10.319', '-4.991', '30.025', 
					'0.726' ]

	except_z_i = [  '0.208', '0.261', '0.269', '0.270', '0.200', '0.217', '0.269',
			  		'0.258', '0.269', '0.264', '0.292', '0.235', '0.243', '0.205',
					'0.224', '0.289', '0.244', '0.286', '0.258', '0.275', '0.255',
					'0.271', '0.291', '0.257', '0.221', '0.210', '0.286', '0.295',
					'0.292', '0.203', '0.212', '0.245', '0.241', '0.255', '0.226',
					'0.266', '0.238', '0.287', '0.268', '0.210', '0.212', '0.285',
					'0.237', '0.225', '0.224', '0.219', '0.266', '0.267', '0.292',
					'0.268', '0.300', '0.279', '0.278', '0.246', '0.219', '0.281',
					'0.287', '0.274', '0.273', '0.280', '0.210', '0.274', '0.277',
					'0.236', '0.218', '0.295', '0.262', '0.238', '0.278', '0.200', 
					'0.268', '0.271', '0.297', '0.255', '0.226', '0.292', '0.271', 
					'0.238']

	x_ra = np.array( [ np.float(ll) for ll in except_ra_i] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_i] )
	x_z = np.array( [np.float(ll) for ll in except_z_i] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_3)

	except_ra_z = [ '8.320',   '8.546',   '18.302',  '24.320',  '26.775',  '37.894',  '117.585', '122.573', '126.251',
					'127.509', '128.611', '130.537', '129.462', '136.374', '140.078', '141.507', '145.632',
					'148.100', '148.541', '153.281', '155.763', '158.222', '159.589', '162.793', '170.262', 
					'175.790', '176.698', '177.809', '179.505', '181.328', '182.343', '183.580', '186.372', 
					'186.760', '189.244', '190.640', '192.743', '195.341', '196.056', '196.951', '208.380', 
					'212.420', '221.403', '221.695', '222.069', '222.506', '224.104', '234.233', '242.344', 
					'244.422', '250.342', '330.545', '330.691', '334.357', '351.883', '10.814',  '186.372',
					'231.786', '12.742',  '190.946', '145.632', '223.211', '18.302' , '238.323', '37.662' , 
					'145.632', '186.372']

	except_dec_z = ['-0.633', '-0.243', '25.084', '7.882',  '15.805', '-1.258', '26.293', '37.200', '4.430',
					'8.460',  '36.515', '38.226', '6.777',  '5.180',  '37.105', '24.074', '30.025', 
					'61.161', '36.054', '17.932', '55.701', '40.602', '42.127', '9.066',  '46.414', 
					'26.859', '9.874',  '33.577', '48.772', '4.034',  '65.523', '32.835', '0.726',  
					'56.723', '14.695', '35.537', '9.056',  '24.035', '7.031',  '43.153', '53.575', 
					'43.790', '8.243',  '16.112', '13.622', '14.797', '40.059', '34.858', '36.393', 
					'42.539', '24.974', '9.516',  '-8.544', '27.836', '0.943',  '2.531',  '0.726',  
					'50.305', '-9.490', '51.831', '30.025', '15.746', '25.084', '10.319', '-4.991', 
					'30.025', '0.726' ]

	except_z_z = [  '0.261', '0.249', '0.255', '0.258', '0.209', '0.299', '0.205', '0.297', '0.224',
					'0.258', '0.289', '0.244', '0.236', '0.258', '0.235', '0.217', '0.271', 
					'0.293', '0.291', '0.261', '0.256', '0.257', '0.211', '0.221', '0.294', 
					'0.292', '0.221', '0.212', '0.278', '0.275', '0.204', '0.226', '0.238', 
					'0.292', '0.231', '0.269', '0.298', '0.276', '0.212', '0.210', '0.225', 
					'0.300', '0.268', '0.212', '0.226', '0.300', '0.274', '0.235', '0.281', 
					'0.294', '0.274', '0.284', '0.210', '0.298', '0.279', '0.262', '0.238', 
					'0.278', '0.200', '0.268', '0.271', '0.297', '0.255', '0.226', '0.292', 
					'0.271', '0.238']

	x_ra = np.array( [ np.float(ll) for ll in except_ra_z] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_z] )
	x_z = np.array( [np.float(ll) for ll in except_z_z] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_4)

	except_ra_u = [ '3.333',   '18.302',  '28.393',  '35.888',  '116.809', '128.611', '130.537', '131.189', 
					'145.545', '150.379', '151.716', '155.763', '155.783', '158.222', '162.913', '168.650', 
					'174.028', '177.809', '183.580', '186.305', '186.372', '189.574', '190.000', '192.056', 
					'195.231', '199.485', '200.248', '201.089', '208.380', '210.752', '219.787', '221.695', 
					'223.099', '237.131', '238.323', '242.344', '244.422', '250.342', '252.835', '257.082', 
					'260.123', '322.517', '323.850', '327.180', '330.691', '334.357', '350.567', '351.680', 
					'346.734', '10.814',  '186.372', '231.786', '12.742',  '190.946', '145.632', '223.211', 
					'18.302' , '238.323', '37.662' , '145.632', '186.372']

	except_dec_u = ['14.517', '25.084', '-1.302', '-7.228', '18.166', '36.515', '38.226', '21.492', 
					'5.595',  '19.691', '27.110', '55.701', '7.108',  '40.602', '29.176', '15.610', 
					'-2.717', '33.577', '32.835', '-0.444', '0.726',  '16.583', '28.296', '18.629', 
					'23.735', '31.760', '50.298', '4.322',  '53.575', '22.709', '29.039', '16.112', 
					'28.189', '16.972', '10.319', '36.393', '42.539', '24.974', '20.606', '21.182', 
					'28.876', '-0.352', '-0.741', '7.024',  '-8.544', '27.836', '-3.376', '1.134' , 
					'15.375', '2.531',  '0.726',  '50.305', '-9.490', '51.831', '30.025', '15.746', 
					'25.084', '10.319', '-4.991', '30.025', '0.726' ]

	except_z_u = [  '0.228', '0.255', '0.244', '0.279', '0.297', '0.289', '0.244', '0.272', 
					'0.219', '0.248', '0.248', '0.256', '0.290', '0.257', '0.228', '0.274', 
					'0.201', '0.212', '0.226', '0.300', '0.238', '0.257', '0.267', '0.227', 
					'0.267', '0.208', '0.234', '0.260', '0.225', '0.221', '0.251', '0.212', 
					'0.246', '0.300', '0.226', '0.281', '0.294', '0.274', '0.264', '0.213', 
					'0.290', '0.237', '0.211', '0.282', '0.210', '0.298', '0.274', '0.277', 
					'0.220', '0.262', '0.238', '0.278', '0.200', '0.268', '0.271', '0.297', 
					'0.255', '0.226', '0.292', '0.271', '0.238']

	x_ra = np.array( [ np.float(ll) for ll in except_ra_u] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_u] )
	x_z = np.array( [np.float(ll) for ll in except_z_u] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_5)

	## use h5py data form. for mpi4py run
	t_ra_r = [np.float(ll) for ll in except_ra_r]
	t_dec_r = [np.float(ll) for ll in except_dec_r]
	t_z_r = [np.float(ll) for ll in except_z_r]
	t_data_r = np.array([t_ra_r, t_dec_r, t_z_r])
	with h5py.File(lod + 'Except_r_sample.h5', 'w') as f:
		f['a'] = np.array(t_data_r)
	with h5py.File(lod + 'Except_r_sample.h5') as f:
		for tt in range(len(t_data_r)):
			f['a'][tt,:] = np.array(t_data_r[tt,:])

	t_ra_g = [np.float(ll) for ll in except_ra_g]
	t_dec_g = [np.float(ll) for ll in except_dec_g]
	t_z_g = [np.float(ll) for ll in except_z_g]
	t_data_g = np.array([t_ra_g, t_dec_g, t_z_g])
	with h5py.File(lod + 'Except_g_sample.h5', 'w') as f:
		f['a'] = np.array(t_data_g)
	with h5py.File(lod + 'Except_g_sample.h5') as f:
		for tt in range(len(t_data_g)):
			f['a'][tt,:] = np.array(t_data_g[tt,:])

	t_ra_i = [np.float(ll) for ll in except_ra_i]
	t_dec_i = [np.float(ll) for ll in except_dec_i]
	t_z_i = [np.float(ll) for ll in except_z_i]
	t_data_i = np.array([t_ra_i, t_dec_i, t_z_i])
	with h5py.File(lod + 'Except_i_sample.h5', 'w') as f:
		f['a'] = np.array(t_data_i)
	with h5py.File(lod + 'Except_i_sample.h5') as f:
		for tt in range(len(t_data_i)):
			f['a'][tt,:] = np.array(t_data_i[tt,:])

	t_ra_u = [np.float(ll) for ll in except_ra_u]
	t_dec_u = [np.float(ll) for ll in except_dec_u]
	t_z_u = [np.float(ll) for ll in except_z_u]
	t_data_u = np.array([t_ra_u, t_dec_u, t_z_u])
	with h5py.File(lod + 'Except_u_sample.h5', 'w') as f:
		f['a'] = np.array(t_data_u)
	with h5py.File(lod + 'Except_u_sample.h5') as f:
		for tt in range(len(t_data_u)):
			f['a'][tt,:] = np.array(t_data_u[tt,:])

	t_ra_z = [np.float(ll) for ll in except_ra_z]
	t_dec_z = [np.float(ll) for ll in except_dec_z]
	t_z_z = [np.float(ll) for ll in except_z_z]
	t_data_z = np.array([t_ra_z, t_dec_z, t_z_z])
	with h5py.File(lod + 'Except_z_sample.h5', 'w') as f:
		f['a'] = np.array(t_data_z)
	with h5py.File(lod + 'Except_z_sample.h5') as f:
		for tt in range(len(t_data_z)):
			f['a'][tt,:] = np.array(t_data_z[tt,:])

	## dr7 select [mainly include "over-masking" part]
	except_ra = ['1.980', '6.566', '29.162', '34.432', '119.661', '129.197', '142.096', '162.870', 
				'167.538', '180.610', '247.334', '337.717', '180.211', '242.027']
	except_dec = ['-8.845', '1.221', '0.842', '-9.105', '13.152', '15.150', '18.003', '17.351', 
				'8.920', '10.562', '47.148', '-0.538', '-1.465', '25.346']
	except_z = ['0.260', '0.292', '0.217', '0.239', '0.230', '0.279', '0.291', '0.202', '0.225', 
				'0.229', '0.211', '0.282', '0.267', '0.270']
	x_ra = np.array( [ np.float(ll) for ll in except_ra] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec] )
	x_z = np.array( [np.float(ll) for ll in except_z] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + 'Bad_match_dr7_cat.csv')

	return

def spec_cat():
	lod = '/home/xkchen/mywork/ICL/data/redmapper/'
	## this mask is mainly for g, r, i band
	cat_ra = ['328.403', '196.004', '140.502', '180.973', '168.198', '140.296', '180.610', 
			'182.343', '241.928', '10.814',  '247.330', '186.372', '14.652', '125.865', 
			'162.469', '199.052', '211.730', '118.804', '200.095', '5.781',   '202.887', 
			'117.585', '176.039', '180.996', '231.786', '189.718', '162.793', '12.742', 
			'190.946', '154.684', '329.477', '167.124', '325.814', '340.792', '236.778', 
			'238.323', '37.662',  '222.321', '145.632', '237.132', '242.433', '244.912', 
			'189.884', '146.241', '170.810', '191.012', '127.750', '231.949', '200.200', 
			'117.736', '151.358', '169.450', '223.211', '132.843', '29.594',  '201.413']

	cat_dec = ['17.695', '67.507', '51.922', '1.783', '2.503', '34.860', '12.262', 
			'65.523', '7.482',  '2.531',  '26.183', '0.726', '4.352',  '54.168', 
			'16.156', '11.363', '55.067', '30.775', '8.433',  '-9.291', '62.642', 
			'26.293', '51.969', '31.968', '50.305', '34.660', '9.066',  '-9.490', 
			'51.831', '60.833', '14.251', '29.294', '8.540',  '-9.211', '19.257', 
			'10.319', '-4.991', '35.293', '30.025', '11.364', '7.709',  '9.767',  
			'42.660', '7.254',  '18.236', '67.781', '13.509', '9.711',  '4.054',  
			'15.913', '50.978', '4.246',  '15.746', '18.194', '16.536', '50.632']

	cat_z = ['0.230', '0.221', '0.204', '0.237', '0.268', '0.238', '0.226', 
			'0.204', '0.224', '0.262', '0.223', '0.238', '0.282', '0.243', 
			'0.209', '0.267', '0.251', '0.287', '0.227', '0.296', '0.219', 
			'0.205', '0.287', '0.204', '0.278', '0.226', '0.221', '0.200', 
			'0.268', '0.200', '0.272', '0.215', '0.255', '0.266', '0.270', 
			'0.226', '0.292', '0.287', '0.271', '0.226', '0.218', '0.237', 
			'0.287', '0.299', '0.270', '0.275', '0.258', '0.245', '0.211', 
			'0.286', '0.228', '0.253', '0.297', '0.250', '0.216', '0.249']

	keys = ['ra', 'dec', 'z']
	values = [cat_ra, cat_dec, cat_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + 'special_mask_cat.csv')
	# use h5py data form. for mpi4py run
	t_ra = [np.float(ll) for ll in cat_ra]
	t_dec = [np.float(ll) for ll in cat_dec]
	t_z = [np.float(ll) for ll in cat_z]
	t_data = np.array([t_ra, t_dec, t_z])
	with h5py.File(lod + 'special_mask_cat.h5', 'w') as f:
		f['a'] = np.array(t_data)
	with h5py.File(lod + 'special_mask_cat.h5') as f:
		for tt in range(len(t_data)):
			f['a'][tt,:] = np.array(t_data[tt,:])

	return	

def sky_rule_out():
	lod = '/home/xkchen/mywork/ICL/data/redmapper/'

	cat_ra = [  '172.584', '155.825', '192.342', '149.544', '227.637', '144.349', 
				'331.843', '357.192', '180.927', '16.543',  '134.612', '189.716', 
				'130.438', '149.878', '131.473', '155.763', '141.359', '188.471', 
				'29.487',  '324.562', '121.334', '329.959', '199.559', '142.417']

	cat_dec = [ '10.462', '11.059', '49.795', '55.255',  '6.838',  '38.844', 
				'23.245', '-2.401', '1.032',  '0.856',   '56.870', '55.646', 
				'56.356', '54.220', '56.191', '55.701',  '60.528', '54.049', 
				'9.404',  '9.908', '48.945',  '22.984',  '1.713',  '8.052' ]

	cat_z = [   '0.206', '0.236', '0.286', '0.212', '0.223', '0.250', 
				'0.223', '0.239', '0.255', '0.263', '0.245', '0.278', 
				'0.229', '0.248', '0.261', '0.256', '0.259', '0.221', 
				'0.210', '0.221', '0.208', '0.200', '0.225', '0.214']

	t_ra = [np.float(ll) for ll in cat_ra]
	t_dec = [np.float(ll) for ll in cat_dec]
	t_z = [np.float(ll) for ll in cat_z]

	sky_cat = np.array([t_ra, t_dec, t_z])
	with h5py.File(lod + 'sky_rule_out_cat.h5', 'w') as f:
		f['a'] = np.array(sky_cat)
	with h5py.File(lod + 'sky_rule_out_cat.h5') as f:
		for tt in range( len(sky_cat) ):
			f['a'][tt,:] = np.array( sky_cat[tt,:] )
	print('saved!')

def main():
	#data_select()
	#spec_cat()
	sky_rule_out()

if __name__ == "__main__":
	main()
