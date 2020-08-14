
# This script loads packages from /src/sage instead of site-packages/sage to make development easier
# For this to work, the following preparation is necessary:
# - Compile sage using `./sage -b`
# - Remove the imports of the packages from site-packages/sage/all.py
# - Delete (or rename) the packages from site-packages/sage
# Moreover, this file has to reside in `src/sage`

# %% Load sage
import sys
import importlib

# Import sage from site-packages
import sage.all

# Load manifolds and tensor packages from src folder, and put them at "sage.manifolds" and "sage.tensor"
spec = importlib.util.find_spec("manifolds")
print(spec) # This should show that the module is loaded from /sage/src/
module = importlib.util.module_from_spec(spec)
sys.modules["manifolds"] = module
sys.modules["sage.manifolds"] = module
sys.modules["__main__.manifolds"] = module

spec = importlib.util.find_spec("tensor")
print(spec) # This should show that the module is loaded from /sage/src/
module = importlib.util.module_from_spec(spec)
sys.modules["sage.tensor"] = module
sys.modules["__main__.tensor"] = module

# %% 
# Reload manifold packages (actually not needed on first run, use this if you change code in /src/manifolds)
for k,v in sys.modules.items():
    if k.startswith('__main__.manifolds'):
        print(k)
        importlib.reload(v)

# Actual imports
from .manifolds.differentiable.tensorfield import TensorField

from .manifolds.differentiable.symplectic_form import SymplecticForm
from .manifolds.differentiable.metric import PseudoRiemannianMetric
from .manifolds.manifold import Manifold

# %%
M = Manifold(2, 'M')
omega = SymplecticForm(M, 'omega', '\omega'); omega

# %%
from .manifolds.catalog import Sphere

M = Sphere(2, 1)
omega = SymplecticForm(M, 'omega', '\omega', M.metric().volume_form())
M.metric().volume_form()
M.metric().volume_form(2).ex
omega[0,1] = 1
omega.display()

# %%
omega.display()

# %%
M.metric().volume_form().display()

# %%
