from setuptools import setup, find_packages

setup(
    name='CryptoEvolutionPackage',
    version='0.2',
    packages=find_packages(include=['source', 'source.*']),
)
