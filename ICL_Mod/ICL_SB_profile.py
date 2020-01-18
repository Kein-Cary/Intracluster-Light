import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

