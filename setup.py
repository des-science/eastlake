import os
import tempfile
import contextlib
import subprocess
import glob
from setuptools import setup, find_packages
import setuptools.command.build_ext
import setuptools.command.build_py


# https://stackoverflow.com/questions/6194499/pushd-through-os-system
@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def _run_shell(cmd):
    print(">>>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def _build_swarp():
    cwd = os.path.abspath(os.getcwd())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pushd(tmpdir):
            _run_shell("cp %s/src/swarp-2.40.1.tar.gz ." % cwd)
            _run_shell("tar -xzvf swarp-2.40.1.tar.gz")
            with pushd("swarp-2.40.1"):
                _run_shell("./configure --prefix=%s" % tmpdir)
                _run_shell("make")
                _run_shell("make install")
            _run_shell("cp bin/swarp %s/eastlake/astromatic/." % cwd)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        _build_swarp()
        super().run()


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        _build_swarp()
        super().run()


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
    cmdclass={
        'build_py': build_py,
        'build_ext': build_ext,
    },
    package_data={
        "eastlake": ["astromatic/*"],
    }
)
