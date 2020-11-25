from sage.manifolds.differentiable.tensorfield_paral import TensorFieldParal
from sage.manifolds.differentiable.diff_form import DiffForm
from sage.manifolds.differentiable.vectorfield import VectorField
from sage.manifolds.differentiable.scalarfield import DiffScalarField
from sage.manifolds.differentiable.vectorfield_module import VectorFieldModule
from typing import Optional, Union
from sage.manifolds.differentiable.manifold import DifferentiableManifold
from sage.manifolds.differentiable.tensorfield import TensorField


class PoissonTensorField(TensorField):
    """
    A Poisson bivector field `\varpi` on a differentiable manifold.

    That is, at each point `m \in M`, `\varpi_m` is a bilinear map of the type:

    .. MATH::

        \varpi_m:\ T^*_m M \times T^*_m M  \to \RR

    where `T^*_m M` stands for the cotangent space to the
    manifold `M` at the point `m`, such that `\varpi_m` is skew-symmetric and the
    Schouten bracket of `\varpi` with itself vanishes.

    """
    def __init__(self, manifold: Union[DifferentiableManifold, VectorFieldModule], name: Optional[str] = None, latex_name: Optional[str] = None):
        r"""
        Construct a Poisson bivector field.

        INPUT:

        - ``manifold`` -- module `\mathfrak{X}(M)` of vector
        fields on the manifold `M`, or the manifold `M` itself
        - ``name`` -- (default: ``varpi``) name given to the Poisson tensor
        - ``latex_name`` -- (default: ``None``) LaTeX symbol to denote the Poisson tensor;
        if ``None``, it is formed from ``name``

        EXAMPLES:

        Standard Poisson tensor on `\RR^2`::

            sage: M.<q, p> = EuclideanSpace(2, "R2", r"\mathbb{R}^2", symbols=r"q:q p:p")
            sage: varpi = PoissonTensorField(M, 'varpi', r'\varpi')
            sage: varpi.set_comp()[1,2] = -1
            TODO check sign conventions!
            sage: varpi.display()
            varpi = -dq/\dp

        """
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold

        if name is None:
            name = "varpi"
            if latex_name is None:
                latex_name = "\\varpi"

        TensorField.__init__(self, vector_field_module, (2, 0), name=name, latex_name=latex_name, antisym=(0, 1))

    def hamiltonian_vector_field(self, function: DiffScalarField) -> VectorField:
        r"""
        X_f \contr \omega + \dif f = 0
        """
        vector_field = - self.sharp(function.exterior_derivative())
        vector_field.set_name('X' + function._name, 'X_{' + function._latex_name + '}')
        return vector_field

    def sharp(self, form: DiffForm) -> VectorField:
        r"""
        \omega^\sharp: T^*M -> T M
        defined by `<\alpha, X> = \omega_m (\omega^\sharp(\alpha), X)`
        for all `X \in T_m M` and `\alpha \in T^*_m M`.
        inverse to flat
        In indicies, `\alpha^i = \pi^{ij} \alpha_j` where `\pi` is the Poisson tensor associated to the symplectic form.
        """

        if form.degree() != 1:
            raise ValueError(f"the degree of the differential form must be one but it is {form.degree()}")

        vector_field = form.up(self)
        vector_field.set_name(form._name + '_sharp', form._latex_name + '^\\sharp')
        return vector_field

    def poisson_bracket(self, f: DiffScalarField, g: DiffScalarField) -> DiffScalarField:
        r"""
        {f, g} = 
        = X_f (g) = -X_g(f) = \pi(\dif f, \dif g)

        [X_f, X_g] = X_{{f,g}}
        """
        poisson_bracket = self.contract(0, f.exterior_derivative()).contract(0, g.exterior_derivative())
        poisson_bracket.set_name(f"poisson({f._name}, {g._name})", '\\{' + f'{f._latex_name}, {g._latex_name}' + '\\}')
        return poisson_bracket


class PoissonTensorFieldParal(PoissonTensorField, TensorFieldParal):
    
    def __init__(self, manifold: Union[DifferentiableManifold, VectorFieldModule], name: Optional[str] = None, latex_name: Optional[str] = None):
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold

        if name is None:
            name = "varpi"
            if latex_name is None:
                latex_name = "\\varpi"

        TensorFieldParal.__init__(self, vector_field_module, (2, 0), name=name, latex_name=latex_name, antisym=(0, 1))

