#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
import platform
import subprocess
import importlib
from wheel.bdist_wheel import bdist_wheel as native_bdist_wheel
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as native_build_ext

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'cffi>=1.12.3',
    'numpy>=1.16.0'
]

test_requirements = [
    'pytest>=2.9.2'
]

os.environ['FC'] = 'gfortran'
os.environ['FCFLAGS'] = "-ffree-line-length-0 -m64 -std=f2003 -march=native -DUSE_CBOOL -fPIC"


operating_system = platform.system()
compiled_base_dir = 'spyrate/_lib'

dir_path = os.getcwd()
compiled_path = os.path.join(dir_path, compiled_base_dir)

ffi_internal_builder = importlib.import_module('spyrate.__build')


# Create a custom build class to build libraries, and patch cython extensions
def build_libraries():
    '''
    Build compiled libraries as part of setuptools build
    '''
    curr_dir = os.getcwd()
    os.chdir(compiled_path)
    os.environ['PWD'] = compiled_path
    if subprocess.call(['make', 'SPYRATE_ARCH='+operating_system]):
        raise RuntimeError('Library build failed, exiting')
    os.chdir(curr_dir)
    os.environ['PWD'] = curr_dir


# Custom build class
class SpyrateBuildExt(native_build_ext):
    '''
    custom build class
    '''

    def run(self):
        if os.environ.get('READTHEDOCS') == 'True':
            return
        build_libraries()
        native_build_ext.run(self)


# Custom bdist_wheel class
class SpyrateBdistWheel(native_bdist_wheel):
    '''
    custom wheel build class
    '''

    def run(self):
        self.run_command('build')
        native_bdist_wheel.run(self)


setup(
    author="Joy Merwin Monteiro",
    author_email='joy.merwin@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python bindings to the Radiative Transfer for Energetics library",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spyrate',
    name='spyrate',
    packages=find_packages(include=['spyrate']),
    ext_modules=[ffi_internal_builder.ffibuilder.distutils_extension()],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JoyMonteiro/spyrate',
    version='0.1.4',
    zip_safe=False,
    cmdclass={
        'build_ext': SpyrateBuildExt,
        'bdist_wheel': SpyrateBdistWheel,
    },
)
