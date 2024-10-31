from setuptools import setup, find_packages

setup(
    name='barcelona_prediction', 
    version='0.1.0',
    description='Library for barcelona apartment price prediction pipeline',
    author='Matias Borrell, Lucia Sauer, Blanca Jimenez',
    author_email='matias.borrell@bse.eu, Lucia Sauer, Blanca Jimenez',
    url='https://github.com/luciasauer/bcn_aparment_price', 
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
    ], 
)