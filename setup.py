from setuptools import setup, find_packages

setup(name='rgbd_sym', 
      version='1.0',
    install_requires=[
        'matplotlib',
        'numpy<=1.23',
        'ipykernel',
        'notebook',
        'scipy',
        'pandas',
        'opencv-python<4.10',
        'gym<=0.24', 
        'ruamel.yaml<=0.17',
        'pynput'
        ], 
      packages=find_packages())
