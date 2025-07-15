# -*- coding: utf-8 -*-
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("CCanalyse.pyx")
)