from setuptools import setup, Extension

module = Extension('symnmf',
                  sources=['symnmfmodule.c', 'symnmf.c'],
                  extra_compile_args=['-O3'],
                  extra_link_args=['-lm'])

setup(name='symnmf',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization Implementation',
      ext_modules=[module])