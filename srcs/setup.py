from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np



exts = [
    Extension(
        "fast_count",
        ["fast_count.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    Extension(
        "encode_corpus",
        ["encode_corpus.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="corpus_pipeline",
    ext_modules=cythonize(exts, language_level="3"),
)
