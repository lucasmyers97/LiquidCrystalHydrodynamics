"""
This is a test to try to create a *.pyf file from the given fortran file using
the numpy.f2py module

Lucas Myers
Written: July 15, 2020
"""

import numpy.f2py

args = ['fib1.f', '-m', 'fib2', '-h', 'fib1.pyf']
failure = numpy.f2py.run_main(args)
print(failure)