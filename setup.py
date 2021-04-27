import os
import tempfile
import contextlib
import subprocess
import glob
import sys


if "--skip-swarp-build" in sys.argv:
    SKIP_SWARP_BUILD = True
    sys.argv.remove("--skip-swarp-build")
else:
    SKIP_SWARP_BUILD = False


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
    if SKIP_SWARP_BUILD:
        return
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


def _build_src_ext():
    if "CONDA_BUILD" in os.environ:
        cc = "${CC}"
        cldflags = "\"${CFLAGS} ${LDFLAGS}\""
    elif all(v in os.environ for v in ["CONDA_PREFIX", "CC", "CFLAGS", "LDFLAGS"]):
        cc = os.environ["CC"]
        cldflags = "\"" + os.environ["CFLAGS"] + " " + os.environ["LDFLAGS"] + "\""
    elif sys.platform == "linux":
        cc = "gcc"
        cldflags = (
            "\"-isystem ${CONDA_PREFIX}/include "
            "-Wl,-rpath,${CONDA_PREFIX}/lib "
            "-Wl,-rpath-link,${CONDA_PREFIX}/lib "
            "-L${CONDA_PREFIX}/lib\""
        )
    elif sys.platform == "darwin":
        cc = "clang"
        cldflags = (
            "\"-ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe "
            "-isystem ${CONDA_PREFIX}/include "
            "-Wl,-rpath,${CONDA_PREFIX}/lib "
            "-L${CONDA_PREFIX}/lib "
            "-Wl,-pie -Wl,-headerpad_max_install_names "
            "-Wl,-dead_strip_dylibs\""
        )
    else:
        print(sys.platform, flush=True)
        raise RuntimeError("platform %s not recognized" % sys.platform)

    cwd = os.path.abspath(os.getcwd())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pushd(tmpdir):
            _run_shell("cp %s/src/src-extractor-2.24.4.tar.gz ." % cwd)
            _run_shell("tar -xzvf src-extractor-2.24.4.tar.gz")
            with pushd("sextractor-2.24.4"):
                try:
                    _run_shell("sh autogen.sh")
                    _run_shell(
                        "CC=%s "
                        "CFLAGS=%s "
                        "./configure "
                        "--prefix=%s "
                        "--enable-openblas "
                        "--with-openblas-libdir=${CONDA_PREFIX}/lib "
                        "--with-openblas-incdir=${CONDA_PREFIX}/include " % (
                            cc, cldflags, tmpdir
                        )
                    )
                except Exception:
                    _run_shell("cat config.log")
                    raise

                _run_shell("make")
                _run_shell("make install")

            _run_shell("cp bin/sex %s/eastlake/astromatic/src-extractor" % cwd)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        _build_swarp()
        _build_src_ext()
        super().run()


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        _build_swarp()
        _build_src_ext()
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
