"""
This script generates a .pyf file from the bihar.f file.

Lucas Myers
Written July 17, 2020
"""

import numpy.f2py

with open('bihar.f') as file:
    source = file.read()
    
args = ['bihar.f', '-m', 'bihar', '-h', 'bihar.pyf']
r = numpy.f2py.run_main(args)

if not r:
    print("Success")