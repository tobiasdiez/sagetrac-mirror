## requirements.txt for creating venvs with sagelib
##
## Usage:
##
##                   $ ../sage -sh
##         (sage-sh) $ python3 -m venv venv1
##         (sage-sh) $ source venv1/bin/activate
## (venv1) (sage-sh) $ pip install -r requirements.txt
## (venv1) (sage-sh) $ pip install -e .

dnl FIXME: Including the whole package-version.txt does not work for packages that have a patchlevel....
dnl We need a better tool to format this information.

sage-conf==include(`../sage_conf/package-version.txt')
dnl sage_setup     # Will be split out later.

dnl From build/pkgs/sagelib/dependencies
cypari2==include(`../cypari/package-version.txt')
dnl ... but building bdist_wheel of cypari2 fails with recent pip... https://github.com/sagemath/cypari2/issues/93
cysignals==include(`../cysignals/package-version.txt')
Cython==include(`../cython/package-version.txt')
gmpy2==include(`../gmpy2/package-version.txt')
jinja2==include(`../jinja2/package-version.txt')
dnl ... for sage_setup.autogen.interpreters
jupyter_core==include(`../jupyter_core/package-version.txt')
numpy==include(`../numpy/package-version.txt')
dnl ... already needed by sage.env
pkgconfig==include(`../pkgconfig/package-version.txt')
dnl pplpy==include(`../pplpy/package-version.txt')
include(`../pplpy/install-requires.txt')
pycygwin==esyscmd(`printf $(cat ../pycygwin/package-version.txt)'); sys_platform == 'cygwin'
dnl pynac       # after converting to a pip-installable package


dnl From Makefile.in: SAGERUNTIME
ipython==include(`../ipython/package-version.txt')
pexpect==include(`../pexpect/package-version.txt')
psutil==include(`../psutil/package-version.txt')

dnl From Makefile.in: DOC_DEPENDENCIES
sphinx==include(`../sphinx/package-version.txt')
networkx==include(`../networkx/package-version.txt')
scipy==include(`../scipy/package-version.txt')
sympy==include(`../sympy/package-version.txt')
matplotlib==include(`../matplotlib/package-version.txt')
pillow==include(`../pillow/package-version.txt')
mpmath==include(`../mpmath/package-version.txt')
ipykernel==include(`../ipykernel/package-version.txt')
jupyter_client==include(`../jupyter_client/package-version.txt')
ipywidgets==include(`../ipywidgets/package-version.txt')

dnl Other Python packages that are standard spkg, used in doctests
cvxopt==include(`../cvxopt/package-version.txt')
rpy2==include(`../rpy2/package-version.txt')
dnl fpylll       # does not install because it does not declare its build dependencies correctly. Reported upstream: https://github.com/fplll/fpylll/issues/185
dnl pycryptosat  # Sage distribution installs it as part of cryptominisat. According to its README on https://pypi.org/project/pycryptosat/: "The pycryptosat python package compiles while compiling CryptoMiniSat. It cannot be compiled on its own, it must be compiled at the same time as CryptoMiniSat."
