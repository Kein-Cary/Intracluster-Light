#!/usr/bin/python2

import pickle
import numpy as np

import pickle
fitdatalist = pickle.load( open('/home/xkchen/figs/ffitdata_jointfit_shan_long2.p', 'r') )

Nl = len( fitdatalist )

cat_lis = ['low_BCG_M', 'high_BCG_M']

for ii in range( Nl ):
	p, rp, delsig, delsig_pred, xirp, xirpmean, rp_data, delsig_data, covmat, r, xihm, rvir, rs, sigoff = fitdatalist[ii]

	data_arr = np.array([ rp, xirp ]).T
	np.savetxt('/home/xkchen/%s_xi-rp.txt' % cat_lis[ii], data_arr, fmt = '%.8f, %.8f')

	print rp_data.shape
