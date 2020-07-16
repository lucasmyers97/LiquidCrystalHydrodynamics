"""
This tests a basic fibonacci number generator fortran program.

Lucas Myers
July 15, 2020
"""

import numpy.f2py
import numpy as np

with open('fib1.f') as file:
    source = file.read()
    
args = ['--compiler=mingw32']
module = 'fib1'

failure = numpy.f2py.compile(source, modulename=module, 
                             extra_args=args, verbose=False)
print(failure)

import fib1

a = np.zeros(8, 'd')
fib1.fib(a)
print(a)