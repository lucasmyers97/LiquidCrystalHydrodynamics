"""
This tests a quick hello world program for using the f2py python/fortran
package.

Lucas Myers
Written: July 15, 2020
"""
import numpy.f2py

with open('hello.f') as file:
    source = file.read()
    
args = ['--compiler=mingw32']
    
a = numpy.f2py.compile(source, modulename='hello', verbose=0, extra_args=args)
print(a)

import hello
hello.foo()