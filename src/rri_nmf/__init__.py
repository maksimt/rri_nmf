# from rri_nmf.nmf import *  # now f() in nmf can be accessed by:
# import rri_nmf
# rri_nmf.f()
import rri_nmf.nmf
import rri_nmf.matrixops
import rri_nmf.optimization
__all__ = ['nmf', 'initialization', 'optimization', 'matrixops',
           'sklearn_interface']