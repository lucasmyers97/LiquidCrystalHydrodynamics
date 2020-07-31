"""
This compiles the bihar.f fortran script using the modified bihar.pyf file
which makes it more pythonic (i.e. no inplace variables, only input and 
output)

Lucas Myers
Written: July 18,2020
"""

import numpy.f2py

with open('bihar.f') as file:
    source = file.read()
    
module = 'bihar'
args = ['bihar.pyf', '--compiler=mingw32']

failure = numpy.f2py.compile(source, extra_args=args, 
                             modulename=module, verbose=False)

print(failure)