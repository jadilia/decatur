from setuptools import setup, find_packages

setup(
    name='decatur',

    version='0.1',

    description='Tidal synchronization of Kepler eclipsing binaries',

    url='https://github.com/jlurie/decatur',

    author='John Lurie',

    author_email='luriejcc@gmail.com',

    license='MIT',

    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 3'],

    packages=find_packages(exclude=['docs, tests']),
)
