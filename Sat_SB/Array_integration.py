import math
import numpy as np
from scipy import integrate as integ
from scipy import interpolate as interp
from scipy.misc import derivative


### === ### 
def print_row(lst):
	print( ' '.join('%11.8f' % x for x in lst) )

def romberg_integ_func( f, a, b, eps = 1e-8 ):
	"""
	Approximate the definite integral of f from a to b by Romberg's method.
	eps is the desired accuracy.
	"""

	##. R[0][0]
	R = [ [0.5 * (b - a) * ( f(a) + f(b) ) ] ]

	# print_row(R[0])

	n = 1

	while True:

		h = float(b - a) / 2 ** n

		##. Add an empty row.
		R.append( [None] * (n + 1) )

		R[n][0] = 0.5 * R[n-1][0] + h * sum( f( a + (2*k-1) * h ) for k in range( 1, 2**(n-1) + 1 ) )

		for m in range(1, n+1):

			R[n][m] = R[n][m-1] + (R[n][m-1] - R[n-1][m-1]) / (4 ** m - 1)

		print_row(R[n])

		if abs(R[n][n-1] - R[n][n]) < eps:

			return R[n][n]

		n += 1

	return

def arr_romber_integ_func( x_arr, y_arr, a, b, eps = 1e-8 ):

	tmp_F = interp.splrep( x_arr, y_arr, s = 0 )

	##. R[0][0]
	mf_a = interp.splev( a, tmp_F, der = 0 )
	mf_b = interp.splev( b, tmp_F, der = 0 )

	R = [ [0.5 * (b - a) * ( mf_a + mf_b ) ] ]

	n = 1

	while True:

		h = float(b - a) / 2 ** n

		##. Add an empty row.
		R.append( [None] * (n + 1) )

		R[n][0] = 0.5 * R[n-1][0] + h * sum( 
					interp.splev( a + (2*k-1) * h, tmp_F, der = 0 ) for k in range( 1, 2**(n-1) + 1 ) )

		for m in range(1, n+1):

			R[n][m] = R[n][m-1] + (R[n][m-1] - R[n-1][m-1]) / (4 ** m - 1)

		if abs(R[n][n-1] - R[n][n]) < eps:

			return R[n][n]

		n += 1

	return


### === ### any type function integration
##. trapezoidal rule
def sum_fun_xk( xk, func ):
	return sum( [ func( each ) for each in xk ] )

def trap_rule_func( a, b, n, func ):
	"""
	func : function to integrate
	a, b : Lower and upper limit of integration
	n : number of points to binned the varables for given function
	"""
	h = ( b - a ) / float( n )

	xk = [ a + i * h for i in range( 1, n ) ]

	return h / 2 * ( func( a ) + 2 * sum_fun_xk( xk, func ) + func( b ) )


##. with Simpson formula
def multi_Simpson_func( a, b, n, func ):
	"""
	func : function to integrate
	a, b : Lower and upper limit of integration
	n : number of points to binned the varables for given function
	"""
	h = (b - a) / n

	temp1 = 0

	temp2 = 0

	for ii in range( 1, n ):

		xk1 = a + h * ii

		xk2 = a + h * (ii + 1)

		xk12 = (xk1 + xk2) / 2

		temp1 += func( xk1 )

		temp2 += func( xk12 )

	temp2 += func( (a + a + h) / 2 )

	return (b - a) / (6 * n) * ( func( a ) + 4 * temp2 + 2 * temp1 + func( b ) )

def arr_multi_Simpson_func( x_arr, y_arr, a, b, n ):
	"""
	x_arr, y_arr : sampled varables and function values	
	"""
	
	tmp_F = interp.splrep( x_arr, y_arr, s = 0 )

	h = (b - a) / n

	temp1 = 0

	temp2 = 0

	for ii in range( 1, n ):

		xk1 = a + h * ii

		xk2 = a + h * (ii + 1)

		xk12 = (xk1 + xk2) / 2

		temp1 += interp.splev( xk1, tmp_F, der = 0 )
		temp2 += interp.splev( xk12, tmp_F, der = 0 )

	temp2 += interp.splev( (a + a + h) / 2, tmp_F, der = 0 )

	mf0 = interp.splev( a, tmp_F, der = 0 )
	mf1 = interp.splev( b, tmp_F, der = 0 )

	II = (b - a) / (6 * n) * ( mf0 + 4 * temp2 + 2 * temp1 + mf1 )

	return II


### === testing
def tx_f( x ):
	return np.exp( x ) * x

if __name__ == "__main__":

	I0 = trap_rule_func( 0, 1, 20, tx_f )
	I1 = multi_Simpson_func( 0, 1, 20, tx_f )

	x0 = np.linspace( 0, 1, 129 )
	y0 = tx_f( x0 )
	I2 = arr_multi_Simpson_func( x0, y0, 0, 1, 20 )

	ddx = np.diff( x0 )
	ddx = np.r_[ ddx, ddx[-1] ]
	I3 = integ.romb( y0, dx = ddx )

	print( I0 / 1 )
	print( I1 / 1 )
	print( I2 / 1 )
	print( I3[-1] / 1 )

	It0 = romberg_integ_func( tx_f, 0, 1, eps = 1e-8 )
	It1 = arr_romber_integ_func( x0, y0, 0, 1, eps = 1e-8 )

	print( It0 / 1 )
	print( It1 / 1 )

