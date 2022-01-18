import h5py
import numpy as np
import pandas as pds

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy

from light_measure import jack_SB_func


# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
rad2arcsec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# band information of SDSS
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])


### === light profile measurement
def PA_SB_Zx_func(data, weit_data, pix_size, cx, cy, z0, R_bins):
	"""
	measure liht profile in physical coordinate (radius in units of kpc)
	---------------------
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	R_bins : radius bin edges for SB measurement, in unit of pixels
	cx, cy: cluster central position in image frame (in inuit pixel)
	pix_size: pixel size
	z : the redshift of data
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	"""
	Da0 = Test_model.angular_diameter_distance(z0).value ## in unit 'Mpc'
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc


	#. surface brightness along the row direction
	intens_r_h = np.zeros(len(rbin), dtype = np.float)
	intens_h = np.zeros(len(rbin), dtype = np.float)
	intens_h_err = np.zeros(len(rbin), dtype = np.float)

	N_pix_h = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio_h = np.zeros(len(rbin), dtype = np.float)

	#. surface brightness along the column direction
	intens_r_v = np.zeros(len(rbin), dtype = np.float)
	intens_v = np.zeros(len(rbin), dtype = np.float)
	intens_v_err = np.zeros(len(rbin), dtype = np.float)

	N_pix_v = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio_v = np.zeros(len(rbin), dtype = np.float)

	dr = np.sqrt( ( (2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ( (2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)
	diff_x = (2 * pix_id[0] + 1) / 2 - (2 * xn + 1) / 2
	diff_y = (2 * pix_id[1] + 1) / 2 - (2 * yn + 1) / 2


	for k in range(len(rbin) - 1):

		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		r_iner = set_r[k] ## useing radius in unit of kpc
		r_out = set_r[k + 1]

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		if bool_sum == 0:
			intens_r_h[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			intens_r_v[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			continue

		else:
			#. points located in radius bin
			id_vx = (dr >= rbin[k]) & (dr < rbin[k + 1])

			#. points along the row direction
			id_ux_0 = np.abs( diff_x ) < rbin[ k ]
			id_uy_0 = ( np.abs( diff_y ) >= rbin[ k ] ) & ( np.abs( diff_y ) < rbin[ k+1 ] )
			id_lim_0 = ( id_ux_0 & id_uy_0 ) & id_vx

			weit_arr_0 = weit_data[ id_lim_0 ]

			samp_flux = data[ id_lim_0 ]
			samp_chi = chi[ id_lim_0 ]
			tot_flux = np.nansum( samp_flux * weit_arr_0 ) / np.nansum( weit_arr_0 )

			idnn = np.isnan( samp_flux )
			N_pix_h[ k ] = np.sum( idnn == False )
			nsum_ratio_h[ k ] = np.nansum( weit_arr_0 ) / np.sum( idnn == False )			

			intens_h[ k ] = tot_flux
			cen_r = np.nansum( dr[ id_lim_0 ] * weit_arr_0 ) / np.nansum( weit_arr_0 ) * pix_size
			intens_r_h[ k ] = cen_r * Da0 * 1e3 / rad2arcsec

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				if np.sum( iv ) == 0:
					continue

				else:
					set_samp = samp_flux[ iv ]
					set_weit = weit_arr_0[ iv ]

					ttf = np.nansum( set_samp * set_weit ) / np.nansum( set_weit )
					tmpf.append( ttf )

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
			tmpf[ id_inf ] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan( tmpf )
			id_fals = id_nan == False
			Tmpf = tmpf[ id_fals ]

			RMS = np.std(Tmpf)

			if len(Tmpf) > 1:
				intens_h_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_h_err[k] = RMS


			#. points along the columns direction
			id_ux_1 = np.abs( diff_y ) < rbin[ k ]
			id_uy_1 = ( np.abs( diff_x ) >= rbin[ k ] ) & ( np.abs( diff_x ) < rbin[ k+1 ] )
			id_lim_1 = ( id_ux_1 & id_uy_1 ) & id_vx

			weit_arr_1 = weit_data[ id_lim_1 ]

			samp_flux = data[ id_lim_1 ]
			samp_chi = chi[ id_lim_1 ]
			tot_flux = np.nansum( samp_flux * weit_arr_1 ) / np.nansum( weit_arr_1 )

			idnn = np.isnan( samp_flux )
			N_pix_v[ k ] = np.sum( idnn == False )
			nsum_ratio_v[ k ] = np.nansum( weit_arr_1 ) / np.sum( idnn == False )			

			intens_v[ k ] = tot_flux
			cen_r = np.nansum( dr[ id_lim_1 ] * weit_arr_1 ) / np.nansum( weit_arr_1 ) * pix_size
			intens_r_v[ k ] = cen_r * Da0 * 1e3 / rad2arcsec

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				if np.sum( iv ) == 0:
					continue

				else:
					set_samp = samp_flux[ iv ]
					set_weit = weit_arr_1[ iv ]

					ttf = np.nansum( set_samp * set_weit ) / np.nansum( set_weit )
					tmpf.append( ttf )

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
			tmpf[ id_inf ] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan( tmpf )
			id_fals = id_nan == False
			Tmpf = tmpf[ id_fals ]

			RMS = np.std(Tmpf)

			if len(Tmpf) > 1:
				intens_v_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_v_err[k] = RMS

	#..
	idzo = N_pix_h < 1

	SB_h = intens_h.copy()
	SB_h[idzo] = 0.
	SB_h_err = intens_h_err.copy()
	SB_h_err[idzo] = 0.

	SB_h_R = intens_r_h.copy()
	nsum_ratio_h[idzo] = 0.
	SB_h, SB_h_err = SB_h / pix_size**2, SB_h_err / pix_size**2


	idzo = N_pix_v < 1

	SB_v = intens_v.copy()
	SB_v[idzo] = 0.
	SB_v_err = intens_v_err.copy()
	SB_v_err[idzo] = 0.

	SB_v_R = intens_r_v.copy()
	nsum_ratio_v[idzo] = 0.
	SB_v, SB_v_err = SB_v / pix_size**2, SB_v_err / pix_size**2

	h_array = [ SB_h_R, SB_h, SB_h_err, N_pix_h, nsum_ratio_h ]
	v_array = [ SB_v_R, SB_v, SB_v_err, N_pix_v, nsum_ratio_v ]

	return h_array, v_array


### === average of jackknife
def aveg_jack_PA_SB_func( J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, N_bin, n_rbins, pix_size, zx, band_str):
	"""
	measure the average SB profile of jackknife subsample
	-----------------------------------
	J_sub_img, J_sub_pix_cont : the stacked image and corresponding pixel counts of subsamples
	J_sub_sb : .csv file, the output file of surface brightness profile of subsamples
	jack_SB_file : .csv file, the output of average jackknife subsample surface brightness

	N_bin, n_rbins : the number of subsample (N_bin), and number of radii bins (n_rbins)
	pix_size, zx : pixel scale (in units of arcsec) and redshift of measurement
	band_str : filter information (i.e., 'g', 'r', 'i')
	"""

	#. radius bin
	lim_r = 0

	for nn in range( N_bin ):

		with h5py.File( J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

		id_nn = np.isnan(tmp_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r = np.max( [lim_r, dR_max] )

	r_bins = np.logspace(0, np.log10(lim_r), n_rbins)

	#. SB measure
	for nn in range( N_bin ):

		with h5py.File( J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])

		with h5py.File( J_sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int( tmp_img.shape[1] / 2), np.int( tmp_img.shape[0] / 2)

		h_array, v_array = PA_SB_Zx_func( tmp_img, tmp_cont, pix_size, xn, yn, zx, r_bins )

		SB_h_R, SB_h, SB_h_err, N_pix_h, nsum_ratio_h = h_array[:]
		SB_v_R, SB_v, SB_v_err, N_pix_v, nsum_ratio_v = v_array[:]

		id_hx = N_pix_h < 1.
		SB_h_R[ id_hx ] = np.nan
		SB_h[ id_hx ] = np.nan 
		SB_h_err[ id_hx ] = np.nan

		id_vx = N_pix_v < 1.
		SB_v_R[ id_vx ] = np.nan
		SB_v[ id_vx ] = np.nan
		SB_v_err[ id_vx ] = np.nan

		#. save
		keys = [ 'r_h', 'sb_h', 'sb_err_h', 'r_v', 'sb_v', 'sb_err_v' ]
		values = [ SB_h_R, SB_h, SB_h_err, SB_v_R, SB_v, SB_v_err ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( J_sub_sb % nn )

	#. average of jackknife
	tmp_h_sb = []
	tmp_h_r = []

	tmp_v_sb = []
	tmp_v_r = []

	for nn in range( N_bin ):

		n_dat = pds.read_csv( J_sub_sb % nn )

		r_arr = np.array( n_dat['r_h'] )
		sb_arr = np.array( n_dat['sb_h'] )
		sb_err = np.array( n_dat['sb_err_h'] )

		tmp_h_sb.append( sb_arr )
		tmp_h_r.append( r_arr )

		r_arr = np.array( n_dat['r_v'] )
		sb_arr = np.array( n_dat['sb_v'] )
		sb_err = np.array( n_dat['sb_err_v'] )

		tmp_v_sb.append( sb_arr )
		tmp_v_r.append( r_arr )

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R_h, tt_jk_SB_h, tt_jk_err_h, lim_R_h = jack_SB_func(tmp_h_sb, tmp_h_r, band_str, N_bin )[4:]
	tt_jk_R_v, tt_jk_SB_v, tt_jk_err_v, lim_R_v = jack_SB_func(tmp_v_sb, tmp_v_r, band_str, N_bin )[4:]

	sb_lim_r_h = np.ones( len( tt_jk_R_h ) ) * lim_R_h
	sb_lim_r_v = np.ones( len( tt_jk_R_v ) ) * lim_R_v

	#.
	keys = [ 'r_h', 'sb_h', 'sb_err_h', 'lim_R_h', 'r_v', 'sb_v', 'sb_err_v', 'lim_R_v' ]
	values = [ tt_jk_R_h, tt_jk_SB_h, tt_jk_err_h, sb_lim_r_h, tt_jk_R_v, tt_jk_SB_v, tt_jk_err_v, sb_lim_r_v ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( jack_SB_file )

	return

