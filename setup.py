# -*- coding: utf-8 -*-
# setup.py
from setuptools import setup, find_packages

setup(
    name='robokhatt',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
        'arabic-reshaper',
        'python-bidi',
        'matplotlib',
        'tqdm'
    ],
)
