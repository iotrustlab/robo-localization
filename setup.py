"""
Setup script for robo-localization package.

This package provides a comprehensive framework for 3D rover localization
using redundant sensor fusion with Extended Kalman Filtering.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Extract core requirements (exclude dev dependencies)
core_requirements = []
dev_requirements = []
optional_requirements = []

for req in requirements:
    if any(dev_pkg in req for dev_pkg in ['pytest', 'black', 'flake8', 'mypy', 'sphinx']):
        dev_requirements.append(req)
    elif any(opt_pkg in req for opt_pkg in ['pandas', 'seaborn', 'jupyter', 'ipywidgets']):
        optional_requirements.append(req)
    else:
        core_requirements.append(req)

setup(
    name='robo-localization',
    version='1.0.0',
    description='3D Rover Localization with Redundant Sensor Fusion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Robo Localization Team',
    author_email='team@robo-localization.org',
    url='https://github.com/robo-localization/robo-localization',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Core dependencies
    install_requires=core_requirements,
    
    # Optional dependencies
    extras_require={
        'analytics': optional_requirements,
        'dev': dev_requirements,
        'all': optional_requirements + dev_requirements,
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Package data
    include_package_data=True,
    
    # Entry points for command line usage
    entry_points={
        'console_scripts': [
            'robo-localization=robo_localization.main:main',
            'robo-localization-demo=robo_localization.demo:run_demo',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords for searchability
    keywords='robotics localization kalman-filter sensor-fusion gps imu odometry extended-kalman-filter',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/robo-localization/robo-localization/issues',
        'Source': 'https://github.com/robo-localization/robo-localization',
        'Documentation': 'https://robo-localization.readthedocs.io/',
    },
)