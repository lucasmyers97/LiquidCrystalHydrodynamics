"""
This file uses the numpy.f2py module to turn a fibonacci *.pyf file into a
useable python module.

Lucas Myers
Written: July 15, 2020
"""

import numpy.f2py

with open('fib1.f') as file:
    source = file.read()
    
args = ['fib2.pyf', '--compiler=mingw32']
module = 'fib2'
failure = numpy.f2py.compile(source, modulename=module, extra_args=args)

print(failure)

import fib2
a = fib2.fib(8)
print(a)