from typing import Optional
from sage.manifolds.differentiable.symplectic_form import SymplecticForm, SymplecticFormParal
from sage.manifolds.differentiable.examples.euclidean import EuclideanSpace


class SymplecticVectorSpace(EuclideanSpace):

    _symplectic_form: SymplecticForm

    def __init__(self, n: int, name=None, latex_name=None,
                 coordinates='Cartesian', symbols: Optional[str] = None, symplectic_name='omega',
                 symplectic_latex_name=None, start_index=1, base_manifold=None,
                 category=None, init_coord_methods=None,
                 unique_tag=None):
        r"""
        EXAMPLES:

        Standard symplectic form on `\RR^2`::

            sage: M.<q, p> = SymplecticVectorSpace(2, symplectic_name='omega')
            sage: omega = M.symplectic_form()
            sage: omega.display()
            omega = -dq/\dp
        """
        dim_half = n // 2

        if symbols is None:
            if dim_half == 1:
                symbols = r"q:q p:p"
            else:
                symbols_list = [f"q{i}:q^{i} p{i}:p_{i}" for i in range(1, dim_half + 1)]
                symbols = ' '.join(symbols_list)

        EuclideanSpace.__init__(self, n, name, latex_name=latex_name,
                                coordinates=coordinates, symbols=symbols, start_index=start_index,
                                base_manifold=base_manifold, category=category, init_coord_methods=init_coord_methods,
                                unique_tag=unique_tag)

        self._symplectic_form = SymplecticFormParal(self, symplectic_name, symplectic_latex_name)
        for i in range(0, dim_half):
            q_index = 2 * i + 1
            self._symplectic_form.set_comp()[q_index, q_index + 1] = -1

    def symplectic_form(self) -> SymplecticForm:
        r"""
        EXAMPLES:

        Standard symplectic form on `\RR^2`::

            sage: M.<q, p> = SymplecticVectorSpace(2, symplectic_name='omega')
            sage: omega = M.symplectic_form()
            sage: omega.display()
            omega = -dq/\dp
        """
        return self._symplectic_form
