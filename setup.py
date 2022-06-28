import os
import tempfile
import contextlib
import subprocess
import glob
import sys

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


def _get_cc_flags_prefix():
    if "CONDA_BUILD" in os.environ:
        print(">>>  building in conda build", flush=True)
        cc = "${CC}"
        cldflags = "\"${CFLAGS} -fcommon\""
        prefix_var = "PREFIX"
    elif all(v in os.environ for v in ["CONDA_PREFIX", "CC", "CFLAGS", "LDFLAGS"]):
        # assume compilers package is installed
        print(">>>  building w/ conda-forge compilers package", flush=True)
        cc = os.environ["CC"]
        cldflags = "\"" + os.environ["CFLAGS"] + " -fcommon\""
        prefix_var = "CONDA_PREFIX"
    elif sys.platform == "linux":
        if "CONDA_PREFIX" in os.environ:
            print(">>>  building in generic conda prefix", flush=True)
            cc = "gcc"
            cldflags = (
                "\"-isystem ${CONDA_PREFIX}/include "
                "-Wl,-rpath,${CONDA_PREFIX}/lib "
                "-Wl,-rpath-link,${CONDA_PREFIX}/lib "
                "-L${CONDA_PREFIX}/lib "
                "-ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 "
                "-ffunction-sections -pipe -fcommon\""
            )
            prefix_var = "CONDA_PREFIX"
        else:
            print(">>>  building without any extra flags - good luck!", flush=True)
            # good luck!
            cc = None
            cldflags = None
    elif sys.platform == "darwin":
        if "CONDA_PREFIX" in os.environ:
            print(">>>  building in generic conda prefix", flush=True)
            cc = "clang"
            cldflags = (
                "\"-ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe "
                "-isystem ${CONDA_PREFIX}/include "
                "-Wl,-rpath,${CONDA_PREFIX}/lib "
                "-L${CONDA_PREFIX}/lib "
                "-Wl,-pie -Wl,-headerpad_max_install_names "
                "-Wl,-dead_strip_dylibs -fcommon\""
            )
            prefix_var = "CONDA_PREFIX"
        else:
            print(">>>  building without any extra flags - good luck!", flush=True)
            # good luck!
            cc = None
            cldflags = None
            prefix_var = None
    else:
        print(sys.platform, flush=True)
        raise RuntimeError("platform %s not recognized" % sys.platform)

    return cc, cldflags, prefix_var


def _build_swarp():
    cc, cldflags, prefix_var = _get_cc_flags_prefix()
    cwd = os.path.abspath(os.getcwd())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pushd(tmpdir):
            _run_shell("cp %s/src/swarp-2.40.1.tar.gz ." % cwd)
            _run_shell("tar -xzvf swarp-2.40.1.tar.gz")
            with pushd("swarp-2.40.1"):
                if cc is not None and cldflags is not None:
                    _run_shell(
                        "CC=%s "
                        "CFLAGS=%s "
                        "./configure --prefix=%s" % (cc, cldflags, tmpdir)
                    )
                else:
                    _run_shell("./configure --prefix=%s" % tmpdir)
                _run_shell("make")
                _run_shell("make install")
            _run_shell("cp bin/swarp %s/eastlake/astromatic/." % cwd)


def _build_src_ext():
    cc, cldflags, prefix_var = _get_cc_flags_prefix()
    cwd = os.path.abspath(os.getcwd())
    with tempfile.TemporaryDirectory() as tmpdir:
        with pushd(tmpdir):
            _run_shell("cp %s/src/src-extractor-2.24.4.tar.gz ." % cwd)
            _run_shell("tar -xzvf src-extractor-2.24.4.tar.gz")
            with pushd("sextractor-2.24.4"):
                try:
                    _run_shell("sh autogen.sh")
                    if cc is not None and cldflags is not None:
                        _run_shell(
                            "CC=%s "
                            "CFLAGS=%s "
                            "./configure "
                            "--prefix=%s "
                            "--enable-openblas "
                            "--with-openblas-libdir=${%s}/lib "
                            "--with-openblas-incdir=${%s}/include " % (
                                cc, cldflags, tmpdir, prefix_var, prefix_var
                            )
                        )
                    else:
                        _run_shell(
                            "./configure "
                            "--prefix=%s " % (
                                tmpdir
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

setup(
    name='eastlake',
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    cmdclass={
        'build_py': build_py,
        'build_ext': build_ext,
    },
    package_data={
        "eastlake": ["astromatic/*", "data/*", "config/*"],
    },
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
