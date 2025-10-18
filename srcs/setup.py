from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "fast_count",                        
    sources=["fast_count.pyx"], 
    language="c++",                      
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-std=c++17"],
    extra_link_args=["-std=c++17"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="fast_count",
    ext_modules=cythonize(ext, language_level="3"),
)

