import os
import glob
from setuptools import setup, find_packages

scripts = glob.glob('./bin/*')
scripts = [os.path.basename(f) for f in scripts if f[-1] != '~']
scripts = [os.path.join('bin', s) for s in scripts]

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "eastlake",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name='eastlake',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
)
