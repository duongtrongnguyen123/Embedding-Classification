from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np



exts = [
    Extension(
        "_count_fast",
        ["_count_fast.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    Extension(
        "_encode_corpus",
        ["_encode_corpus.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="corpus_pipeline",
    ext_modules=cythonize(exts, language_level="3"),
)
