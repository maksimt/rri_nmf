from setuptools import setup, find_packages

setup(name='rri_nmf',
      description='Non-negative matrix factorization using rank-one residue '
                  'iterations.',

      version='0.1', author='Maksim Tsikhanovich',
      url='https://github.com/maksimt/rri_nmf',

      packages=find_packages(where='src'), package_dir={'': 'src'})