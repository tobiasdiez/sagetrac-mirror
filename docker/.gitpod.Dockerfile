##
## Install system packages
##
FROM gitpod/workspace-base as prepare

USER gitpod
# Only copy build, for package information needed for the system package install.
# configure.ac is needed because build/sage_bootstrap uses it to recognize SAGE_ROOT.
COPY --chown=gitpod:gitpod ./configure.ac ./configure.ac
COPY --chown=gitpod:gitpod ./build ./build

# Install system packages
RUN sudo install-packages \
        python3-pip \
        $(build/bin/sage-get-system-packages debian \
            _bootstrap \
            $(PATH=build/bin:$PATH build/bin/sage-package list \
                 --has-file=spkg-configure.m4 :standard: \
              | grep -E -v "pari|tox|flint" ))
    # As of 2021-12, gitpod uses ubuntu-focal. To save space, we filter out some packages (pari, flint) that are
    # too old and will be rejected by our configure script.
    # We do not install tox, since it pulls in javascript-common which does not install for some reason
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade \
        setuptools wheel virtualenv pipenv pylint rope flake8 mypy autopep8 pep8 pydocstyle notebook

## Homebrew has some more up-to-date packages (but sage is not yet able to find them)
### RUN brew update && brew upgrade
### RUN brew install arb flint fplll tox
### We do not install ecl from brew, since this breaks the build of maxima
### Installing pari from brew doesn't work as gitpod gp executable is then hidden by pari/gp
### RUN brew install pari pari-elldata pari-galdata pari-galpol pari-seadata
### Give prio to brew over other system packages
### ENV PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"

##
## Prebuild non-Python packages that have no (working) system-installed package 
##
FROM prepare as prebuild
USER gitpod
### We cannot copy everything due to https://github.com/gitpod-io/gitpod/issues/7157
### COPY --chown=gitpod:gitpod . .
### Thus only selectively copy the files we need
COPY --chown=gitpod:gitpod ./bootstrap ./bootstrap
COPY --chown=gitpod:gitpod ./src/doc/bootstrap ./src/doc/bootstrap
COPY --chown=gitpod:gitpod ./src/bin ./src/bin
COPY --chown=gitpod:gitpod ./m4 ./m4
COPY --chown=gitpod:gitpod ./pkgs ./pkgs
COPY --chown=gitpod:gitpod ./sage ./sage
COPY --chown=gitpod:gitpod ./Makefile ./Makefile
RUN ./bootstrap
RUN ./configure --prefix=/home/gitpod/sage-prebuild
### V=0 since otherwise we would reach log limit
### Gitpod also puts a timeout at 1h
### So we use the construction timeout ... || true
### to make sure we are below this limit and ensure that the docker build doesn't fail due to hitting this limit
RUN MAKE='make -j16' timeout 45m make build-local V=0  || true

##
## Build final image
##
FROM prepare
# Reuse the prebuild packages
COPY --chown=gitpod:gitpod --from=prebuild /home/gitpod/sage-prebuild /home/gitpod/sage-prebuild
ENV PATH=/home/gitpod/sage-prebuild/bin:$PATH
ENV PKG_CONFIG_PATH=/home/gitpod/sage-prebuild/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CPPFLAGS="-I/home/gitpod/sage-prebuild/include $CPPFLAGS"
ENV LDFLAGS="-L/home/gitpod/sage-prebuild/lib $LDFLAGS"
ENV LD_LIBRARY_PATH="/home/gitpod/sage-prebuild/lib:$LD_LIBRARY_PATH"

# Configure 
## Gitpod sets PIP_USER: yes by default (in the full workspace image), which leads to problems during build (e.g pip not being installed in the venv)
## RUN unset PIP_USER
## Gitpod installs pyenv by default (in the full workspace image), and sage's pip install targets the pyenv python for some reason
## RUN pyenv global system
