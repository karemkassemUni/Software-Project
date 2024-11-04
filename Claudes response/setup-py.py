from setuptools import setup, Extension

module = Extension('symnmf',
                  sources=['symnmfmodule.c', 'symnmf.c'],
                  include_dirs=['.'])

setup(name='symnmf',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization implementation',
      ext_modules=[module])
