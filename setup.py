from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="LVAR_calculations_final",
    ext_modules=cythonize('LVAR_calculations_final.pyx'),
    include_dirs=[numpy.get_include()]
)
