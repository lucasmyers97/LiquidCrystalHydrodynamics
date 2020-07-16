"""
This code creates a Fortran extension for python of a function which produces
a fibonacci sequence and then adds one to all the entries. It is meant to test
using subroutines which live in seprate files.

This test failed.

Lucas Myers
Written: July 15, 2020
"""

import numpy.f2py
import numpy as np

with open('fib2.f') as file:
    source = file.read()
    
module = 'fibadd1'
args = ['fib2.pyf','fib1.f', '--compiler=mingw32']

failure = numpy.f2py.compile(source, modulename=module, 
                             extra_args=args, verbose=False)

print(failure)

import fibadd1

a = fibadd.fibadd(8)
print(a)