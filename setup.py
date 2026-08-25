#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

with open('README.rst', 'rt', encoding='utf-8') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='ifta-segmentation',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='A Python toolkit for Histopathology Image Analysis',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Kitware, Inc.',
    author_email='developers@digitalslidearchive.net',
    url='https://github.com/DigitalSlideArchive/HistomicsTK',
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={'ifta': 'ifta'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.22,<2.0',
        'scipy>=1.8,<2.0',
        'pandas>=1.5,<3.0',
        'matplotlib>=3.6,<4.0',
        'imageio>=2.21,<3.0',
        'Pillow>=9.5,<11.0',
        'shapely[vectorized]>=2.0,<3.0',
        'scikit-image>=0.20,<0.25',
        'scikit-learn>=1.2,<1.6',
        'joblib>=1.1,<2.0',
        'lxml>=4.9,<5.0',
        'tifffile>=2023.8.12,<2024.5',
        'tiffslide<=2.1.2',
        'tqdm>=4.64,<5.0',
        'umap-learn>=0.5.4,<0.6',
        'openpyxl>=3.0,<4.0',
        'xlrd<2',
        'h5py>=3.8,<4.0',
        'opencv-python-headless>=4.7,<5.0',
        'openslide-python>=1.2,<2.0',
        'pyvips>=2.2,<3.0',
        # retire-girder-dependency: girder-slicer-cli-web/girder-client/ctk-cli dropped —
        # I/O now goes through storage_client.py (StorageClient) against the first-party storage API
        'requests',
        'dask[dataframe]>=2023.1.0,<2025.0',
        'distributed>=2023.1.0,<2025.0',
        'termcolor>=2.0,<3.0',
        'seaborn>=0.12,<0.14',
        'protobuf>=3.20.3,<5',
        'zarr>=2.14,<2.17',
        'numcodecs>=0.12,<0.16',
        'asciitree>=0.3.3'
    ],
    python_requires='>=3.10',
    license='Apache Software License 2.0',
    keywords='ifta',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)