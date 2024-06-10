from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='store_turnover_forecast',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.22.0',
        'joblib>=0.14.0',
        'pytest>=5.0.0',
        'pyyaml>=5.3.0',
        'jupyter>=1.0.0'
    ],
    entry_points={
        'console_scripts': [
            'run_pipeline=scripts.run_pipeline:main',
        ],
    },
    author='Victoire Bonaud',
    author_email='victoirebonaud@gmail.com',
    description='A project to forecast store turnover using machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vbonaud/store_turnover_forecast',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
