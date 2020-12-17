## Pipfile with all dependencies of sagelib and version information as free as possible
## (for developers to generate a dev environment)
[[source]]
name = "pypi"
url = "https://pypi.org/simple"
verify_ssl = true

[dev-packages]
pkgconfig = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../pkgconfig/install-requires.txt))"')"
cython = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../cython/install-requires.txt))"')"
pycodestyle = "*"
ipykernel = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../ipykernel/install-requires.txt))"')"
tox = "*"
jinja2 = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../jinja2/install-requires.txt))"')"
pytest = "*"
ipywidgets = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../ipywidgets/install-requires.txt))"')"
sphinx = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../sphinx/install-requires.txt))"')"
rope = "*"
six = "*"
jupyter-core = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../jupyter_core/install-requires.txt))"')"

[packages]
numpy = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../numpy/install-requires.txt))"')"
cysignals = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../cysignals/install-requires.txt))"')"
cypari2 = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../cypari/install-requires.txt))"')"
gmpy2 = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../gmpy2/install-requires.txt))"')"
psutil = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../psutil/install-requires.txt))"')"
pexpect = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../pexpect/install-requires.txt))"')"
ipython = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../ipython/install-requires.txt))"')"
sympy = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../sympy/install-requires.txt))"')"
scipy = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../scipy/install-requires.txt))"')"
pplpy = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../pplpy/install-requires.txt))"')"
matplotlib = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../matplotlib/install-requires.txt))"')"
cvxopt = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../cvxopt/install-requires.txt))"')"
rpy2 = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../rpy2/install-requires.txt))"')"
networkx = "esyscmd(`printf "$(echo $(sed "s/#.*//;s/^[-_a-zA-Z0-9]* *//;" ../networkx/install-requires.txt))"')"

[requires]
python_version = "3.8"

[packages.e1839a8]
path = "."
editable = true
