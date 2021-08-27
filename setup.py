import setuptools
from io import open

requirements = [
    'numpy',
    'scipy',
    'mrcfile',
    'matplotlib',
    'pathos',
    'natsort',
    'scikit-image'
]

setuptools.setup(
    name='montage',
    maintainer='Ariana Peck',
    version='0.1.0',
    maintainer_email='apeck@slac.stanford.edu',
    description='Montage cryo-electron tomography',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/apeck12/montage.git',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False)
