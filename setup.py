from setuptools import setup

import os
import sys

with open('README.md') as f:
  long_description = f.read()


setup(
  name = 'EnsembleBench',
  packages = ['EnsembleBench',],
  version = '0.0.0.1',      
  description = 'A set of tools for building good ensemble model teams in machine learning.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Yanzhao Wu',
  author_email = 'yanzhaowumail@gmail.com',
  url = 'https://github.com/git-disl/EnsembleBench',
  download_url = 'https://github.com/git-disl/EnsembleBench/archive/master.zip',
  keywords = ['ENSEMBLE', 'INFERENCE', 'MACHINE LEARNING'],
  install_requires=[
          'numpy',
          'matplotlib',
          'scikit-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
