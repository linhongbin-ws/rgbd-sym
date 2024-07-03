from setuptools import setup, find_packages

setup(name='rgbd_sym', 
      version='1.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'ipykernel',
        'notebook',
        'scipy',
        'pandas',
        'opencv-python<4.10',
        ], 
      packages=find_packages())
