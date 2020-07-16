"""
This test is meant to try to compile a Fortran program which uses a subroutine
from a different file. The expected output is for it to create the fibonacci
numbers and add 1 to everything.

Lucas Myers
July 15, 2020
"""

import numpy.f2py
import numpy as np

with open('fib2.f') as file:
    source = file.read()
module = 'fibadd1'
args = ['fib1.f', '--compiler=mingw32']
failure = numpy.f2py.compile(source, modulename=module, 
                             extra_args=args, verbose=False)
print(failure)

import fibadd1
a = np.zeros(8, 'd')
fibadd1.fibadd(a)
print(a)