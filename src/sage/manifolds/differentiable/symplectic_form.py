r"""
Symplectic structures

The class :class:`SymplecticForm` implements symplectic structures
on differentiable manifolds over `\RR`. The derived class
:class:`SymplecticFormParal` is devoted to symplectic forms on a
parallelizable manifold.

AUTHORS:

- Tobias Diez (2020) : initial version

REFERENCES:

- [AM1990]_
- [RS2007]_

"""
# *****************************************************************************
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# *****************************************************************************
from sage.manifolds.differentiable.poisson_tensor import PoissonTensorField
from six.moves import range
from typing import Dict, Union, overload, Optional

from sage.rings.integer import Integer
from sage.manifolds.differentiable.diff_form import DiffForm, DiffFormParal
from sage.manifolds.differentiable.diff_map import DiffMap
from sage.manifolds.differentiable.vectorfield_module import VectorFieldModule

from sage.manifolds.differentiable.tensorfield import TensorField
from sage.manifolds.differentiable.tensorfield_paral import TensorFieldParal
from sage.manifolds.differentiable.metric import PseudoRiemannianMetric
from sage.manifolds.differentiable.manifold import DifferentiableManifold
from sage.manifolds.scalarfield import ScalarField
from sage.manifolds.differentiable.scalarfield import DiffScalarField
from sage.manifolds.differentiable.vectorfield import VectorField


class SymplecticForm(DiffForm):
    r"""
    A symplectic form on a differentiable manifold.

    An instance of this class is a field `\omega` of nondegenerate skew-symmetric bilinear
    forms on a differentiable manifold `M` over `\RR`.

    That is, at each point `m \in M`, `\omega_m` is a bilinear map of the type:

    .. MATH::

        \omega_m:\ T_m M \times T_m M  \to \RR

    where `T_m M` stands for the tangent space to the
    manifold `M` at the point `m`, such that `\omega_m` is skew-symmetric:
    `\forall u,v \in T_m M, \ \omega_m(v,u) = - \omega_m(u,v)`
    and nondegenerate:
    `(\forall v \in T_m M,\ \ \omega_m(u,v) = 0) \Longrightarrow u=0`.

    .. NOTE::

        If `M` is parallelizable, the class :class:`SymplecticFormParal`
        should be used instead.
    
    TESTS:
    sage: import pytest
    sage: pytest.main("symplectic_form_test.py")
    TODO: add output

    """

    _name: str
    _latex_name: str
    _dim_half: int
    _poisson: Optional[PoissonTensorField]
    _vol_form: Optional[DiffForm]
    _restrictions: Dict[DifferentiableManifold, 'SymplecticForm']

    def __init__(self, manifold: Union[DifferentiableManifold, VectorFieldModule], name: Optional[str] = None, latex_name: Optional[str] = None):
        r"""
        Construct a symplectic form.

        INPUT:

        - ``manifold`` -- module `\mathfrak{X}(M)` of vector
        fields on the manifold `M`, or the manifold `M` itself
        - ``name`` -- (default: ``omega``) name given to the symplectic form
        - ``latex_name`` -- (default: ``None``) LaTeX symbol to denote the symplectic form;
        if ``None``, it is formed from ``name``

        EXAMPLES:

        Standard symplectic form on `\RR^2`::

            sage: M.<q, p> = EuclideanSpace(2, "R2", r"\mathbb{R}^2", symbols=r"q:q p:p")
            sage: omega = SymplecticForm(M, 'omega', r'\omega')
            sage: omega.set_comp()[1,2] = -1
            sage: omega.display()
            omega = -dq/\dp

        """
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold

        if name is None:
            name = "omega"
            if latex_name is None:
                latex_name = "\\omega"

        if latex_name is None:
            latex_name = name

        DiffForm.__init__(self, vector_field_module, 2, name=name, latex_name=latex_name)

        # Check that manifold is even dimensional
        dim = self._ambient_domain.dimension()
        if dim % 2 == 1:
            raise ValueError(f"the dimension of the manifold must be even but it is {dim}")
        self._dim_half = dim // 2

        # Initialization of derived quantities
        SymplecticForm._init_derived(self)

    def _repr_(self):
        r"""
        String representation of the object.
        """
        return self._final_repr(f"Symplectic form {self._name} ")

    def _new_instance(self):
        r"""
        Create an instance of the same class as ``self``.
        """
        return type(self)(self._vmodule, 'unnamed symplectic form',
                          latex_name=r'\mbox{unnamed symplectic form}')

    def _init_derived(self):
        r"""
        Initialize the derived quantities.
        """
        # Initialization of quantities pertaining to the mother class
        DiffForm._init_derived(self)

        self._poisson = None
        self._vol_form = None

    def _del_derived(self):
        r"""
        Delete the derived quantities.
        """
        # Delete the derived quantities from the mother class
        DiffForm._del_derived(self)

        # Clear the Poisson tensor
        if self._poisson is not None:
            self._poisson._restrictions.clear()
            self._poisson._del_derived()
            self._poisson = None

        # Delete the volume form
        if self._vol_form is not None:
            self._vol_form._restrictions.clear()
            self._vol_form._del_derived()
            self._vol_form = None

    def restrict(self, subdomain: DifferentiableManifold, dest_map: Optional[DiffMap] = None) -> 'SymplecticForm':
        r"""
        Return the restriction of the symplectic form to some subdomain.

        If the restriction has not been defined yet, it is constructed here.

        INPUT:

        - ``subdomain`` -- open subset `U` of the symplectic form's domain
        - ``dest_map`` -- (default: ``None``) destination map
          `\Phi:\ U \to V`, where `V` is a subdomain of the symplectic form's domain
          If None, the restriction of the initial vector field module is used.

        OUTPUT:

        - the restricted symplectic form.

        EXAMPLES::

            sage: M = Manifold(6, 'M')
            sage: omega = SymplecticForm(M)
            sage: U = M.open_subset('U')
            sage: omega.restrict(U)
            Symplectic form omega on the Open subset U of the
             6-dimensional differentiable manifold M
        """
        if subdomain == self._domain:
            return self
        
        if subdomain not in self._restrictions:
            # Construct the restriction at the tensor field level
            restriction = DiffForm.restrict(self, subdomain, dest_map=dest_map)
            
            # Restrictions of derived quantities
            if self._poisson is not None:
                restriction._poisson = self._poisson.restrict(subdomain)
            if self._vol_form is not None:
                restriction._vol_form = self._vmodule.restrict(subdomain)

            # The restriction is ready
            self._restrictions[subdomain] = restriction
            return restriction
        else:
            return self._restrictions[subdomain]

    @staticmethod
    def wrap(form: DiffForm, name: Optional[str] = None, latex_name: Optional[str] = None) -> 'SymplecticForm':
        r"""
        Define the symplectic form from a differential form.

        INPUT:

        - ``form`` -- differential `2`-form

        EXAMPLES:

        Volume form on the sphere as a symplectic form:

            sage: M = Sphere(2)
            sage: vol_form = M.metric().volume_form()
            sage: omega = SymplecticForm.wrap(vol_form, 'omega', r'\omega')
            sage: omega.display()
            TODO
        """
        if form.degree() != 2:
            raise TypeError("the argument must be a form of degree 2")
        
        if isinstance(form, DiffFormParal):
            return SymplecticFormParal.wrap(form, name, latex_name)

        if name is None:
            name = form._name
        if latex_name is None:
            latex_name = form._latex_name

        symplectic_form = SymplecticForm(form.base_module(), name, latex_name)

        for dom, rst in form._restrictions.items():
            if isinstance(rst, DiffFormParal):
                symplectic_form._restrictions[dom] = SymplecticFormParal.wrap(rst)
            else:
                symplectic_form._restrictions[dom] = SymplecticForm.wrap(rst)
        return symplectic_form

    def poisson(self, expansion_symbol=None, order=1) -> PoissonTensorField:
        r"""
        Return the Poisson tensor associated to the symplectic form.

        INPUT:

        - ``expansion_symbol`` -- (default: ``None``) symbolic variable; if
          specified, the inverse will be expanded in power series with respect
          to this variable (around its zero value)
        - ``order`` -- integer (default: 1); the order of the expansion
          if ``expansion_symbol`` is not ``None``; the *order* is defined as
          the degree of the polynomial representing the truncated power series
          in ``expansion_symbol``; currently only first order inverse is
          supported

        If ``expansion_symbol`` is set, then the zeroth order symplecic form must be
        invertible. Moreover, subsequent calls to this method will return
        a cached value, even when called with the default value (to enable
        computation of derived quantities). To reset, use :meth:`_del_derived`.

        OUTPUT:

        - the Poisson tensor

        EXAMPLES:

        Poisson tensor of `2`-dimensional symplectic vector space::

            sage: M = SymplecticVectorSpace(2)
            sage: omega = M.symplectic_form()
            sage: poisson = omega.poisson(); poisson
            Tensor field inv_g of type (2,0) on the 2-dimensional differentiable manifold S^2
            sage: poisson.display()
            inv_g = (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dx*d/dx
             + (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dy*d/dy
        """

        if self._poisson is None:
            # Initialize the Poisson tensor
            poisson_name = f'poisson_{self._name}'
            poisson_latex_name = f'{self._latex_name}^{{-1}}'
            self._poisson = PoissonTensorField(self._vmodule, poisson_name, poisson_latex_name)

        # Update the Poisson tensor
        # TODO: Should this be done instead when a new restriction is added?
        for domain, restriction in self._restrictions.items():
            # Forces the update of the restriction
            self._poisson._restrictions[domain] = restriction.poisson(
                                             expansion_symbol=expansion_symbol,
                                             order=order) 
        return self._poisson

    def hamiltonian_vector_field(self, function: DiffScalarField) -> VectorField:
        return self.poisson().hamiltonian_vector_field(function)
    
    def flat(self, vector_field: VectorField) -> DiffForm:
        r"""
        \omega^\flat: TM -> T^* M
        defined by `<\omega^\flat(X), Y> = \omega_m (X, Y)`
        for all `X, Y \in T_m M`.
        In indicies, `X_i = \omega_{ji} X^j`.
        """
        form = vector_field.down(self)
        form.set_name(vector_field._name + '_flat', vector_field._latex_name + '^\\flat')
        return form

    def sharp(self, form: DiffForm) -> VectorField:
        r"""
        \omega^\sharp: T^*M -> T M
        defined by `<\alpha, X> = \omega_m (\omega^\sharp(\alpha), X)`
        for all `X \in T_m M` and `\alpha \in T^*_m M`.
        inverse to flat
        In indicies, `\alpha^i = \pi^{ij} \alpha_j` where `\pi` is the Poisson tensor associated to the symplectic form.
        """
        return self.poisson().sharp(form)
    
    def poisson_bracket(self, f: DiffScalarField, g: DiffScalarField) -> DiffScalarField:
        r"""
        {f, g} = \omega(X_f, X_g)
        = X_f (g) = -X_g(f) = \pi(\dif f, \dif g)

        [X_f, X_g] = X_{{f,g}}
        """
        return self.poisson().poisson_bracket(f, g)

    def volume_form(self) -> DiffForm:
        r"""
        Liouville volume form `\omega^n` associated with the symplectic form `\omega`,
        where `2n` is the dimension of the manifold).
        TODO: Decide about normalization

        OUTPUT:

        - volume form associated to the symplectic form

        EXAMPLES:

        Volume form on `\RR^4`::

            sage: M = SymplecticVectorSpace(4, 'R4')
            sage: omega = M.symplectic_form()
            sage: vol = omega.volume_form() ; vol
            4-form omega^2 on the 4-dimensional differentiable manifold R4
            sage: vol.display()
            omega^2 = TODO: add
        """
        if self._vol_form is None:
            self._vol_form = self
            for _ in range(1, self._dim_half):
                self._vol_form = self._vol_form.wedge(self)

            # TODO: Or use something as vol_omega as name?
            volume_name = f'{self._name}^{self._dim_half}'
            volume_latex_name = f'{self._latex_name}^{{{self._dim_half}}}'
            self._vol_form.set_name(volume_name, volume_latex_name)

        return self._vol_form

    def hodge_star(self, pform):
        r"""
        TODO: Add symplectic hodge star
        Compute the Hodge dual of a differential form with respect to the
        metric.

        If the differential form is a `p`-form `A`, its *Hodge dual* with
        respect to the metric `g` is the
        `(n-p)`-form `*A` defined by

        .. MATH::

            *A_{i_1\ldots i_{n-p}} = \frac{1}{p!} A_{k_1\ldots k_p}
                \epsilon^{k_1\ldots k_p}_{\qquad\ i_1\ldots i_{n-p}}

        where `n` is the manifold's dimension, `\epsilon` is the volume
        `n`-form associated with `g` (see :meth:`volume_form`) and the indices
        `k_1,\ldots, k_p` are raised with `g`.

        INPUT:

        - ``pform``: a `p`-form `A`; must be an instance of
          :class:`~sage.manifolds.differentiable.scalarfield.DiffScalarField`
          for `p=0` and of
          :class:`~sage.manifolds.differentiable.diff_form.DiffForm` or
          :class:`~sage.manifolds.differentiable.diff_form.DiffFormParal`
          for `p\geq 1`.

        OUTPUT:

        - the `(n-p)`-form `*A`

        EXAMPLES:

        Hodge dual of a 1-form in the Euclidean space `R^3`::

            sage: M = Manifold(3, 'M', start_index=1)
            sage: X.<x,y,z> = M.chart()
            sage: g = M.metric('g')
            sage: g[1,1], g[2,2], g[3,3] = 1, 1, 1
            sage: var('Ax Ay Az')
            (Ax, Ay, Az)
            sage: a = M.one_form(Ax, Ay, Az, name='A')
            sage: sa = g.hodge_star(a) ; sa
            2-form *A on the 3-dimensional differentiable manifold M
            sage: sa.display()
            *A = Az dx/\dy - Ay dx/\dz + Ax dy/\dz
            sage: ssa = g.hodge_star(sa) ; ssa
            1-form **A on the 3-dimensional differentiable manifold M
            sage: ssa.display()
            **A = Ax dx + Ay dy + Az dz
            sage: ssa == a  # must hold for a Riemannian metric in dimension 3
            True

        Hodge dual of a 0-form (scalar field) in `R^3`::

            sage: f = M.scalar_field(function('F')(x,y,z), name='f')
            sage: sf = g.hodge_star(f) ; sf
            3-form *f on the 3-dimensional differentiable manifold M
            sage: sf.display()
            *f = F(x, y, z) dx/\dy/\dz
            sage: ssf = g.hodge_star(sf) ; ssf
            Scalar field **f on the 3-dimensional differentiable manifold M
            sage: ssf.display()
            **f: M --> R
               (x, y, z) |--> F(x, y, z)
            sage: ssf == f # must hold for a Riemannian metric
            True

        Hodge dual of a 0-form in Minkowski spacetime::

            sage: M = Manifold(4, 'M')
            sage: X.<t,x,y,z> = M.chart()
            sage: g = M.lorentzian_metric('g')
            sage: g[0,0], g[1,1], g[2,2], g[3,3] = -1, 1, 1, 1
            sage: g.display()  # Minkowski metric
            g = -dt*dt + dx*dx + dy*dy + dz*dz
            sage: var('f0')
            f0
            sage: f = M.scalar_field(f0, name='f')
            sage: sf = g.hodge_star(f) ; sf
            4-form *f on the 4-dimensional differentiable manifold M
            sage: sf.display()
            *f = f0 dt/\dx/\dy/\dz
            sage: ssf = g.hodge_star(sf) ; ssf
            Scalar field **f on the 4-dimensional differentiable manifold M
            sage: ssf.display()
            **f: M --> R
               (t, x, y, z) |--> -f0
            sage: ssf == -f  # must hold for a Lorentzian metric
            True

        Hodge dual of a 1-form in Minkowski spacetime::

            sage: var('At Ax Ay Az')
            (At, Ax, Ay, Az)
            sage: a = M.one_form(At, Ax, Ay, Az, name='A')
            sage: a.display()
            A = At dt + Ax dx + Ay dy + Az dz
            sage: sa = g.hodge_star(a) ; sa
            3-form *A on the 4-dimensional differentiable manifold M
            sage: sa.display()
            *A = -Az dt/\dx/\dy + Ay dt/\dx/\dz - Ax dt/\dy/\dz - At dx/\dy/\dz
            sage: ssa = g.hodge_star(sa) ; ssa
            1-form **A on the 4-dimensional differentiable manifold M
            sage: ssa.display()
            **A = At dt + Ax dx + Ay dy + Az dz
            sage: ssa == a  # must hold for a Lorentzian metric in dimension 4
            True

        Hodge dual of a 2-form in Minkowski spacetime::

            sage: F = M.diff_form(2, name='F')
            sage: var('Ex Ey Ez Bx By Bz')
            (Ex, Ey, Ez, Bx, By, Bz)
            sage: F[0,1], F[0,2], F[0,3] = -Ex, -Ey, -Ez
            sage: F[1,2], F[1,3], F[2,3] = Bz, -By, Bx
            sage: F[:]
            [  0 -Ex -Ey -Ez]
            [ Ex   0  Bz -By]
            [ Ey -Bz   0  Bx]
            [ Ez  By -Bx   0]
            sage: sF = g.hodge_star(F) ; sF
            2-form *F on the 4-dimensional differentiable manifold M
            sage: sF[:]
            [  0  Bx  By  Bz]
            [-Bx   0  Ez -Ey]
            [-By -Ez   0  Ex]
            [-Bz  Ey -Ex   0]
            sage: ssF = g.hodge_star(sF) ; ssF
            2-form **F on the 4-dimensional differentiable manifold M
            sage: ssF[:]
            [  0  Ex  Ey  Ez]
            [-Ex   0 -Bz  By]
            [-Ey  Bz   0 -Bx]
            [-Ez -By  Bx   0]
            sage: ssF.display()
            **F = Ex dt/\dx + Ey dt/\dy + Ez dt/\dz - Bz dx/\dy + By dx/\dz
             - Bx dy/\dz
            sage: F.display()
            F = -Ex dt/\dx - Ey dt/\dy - Ez dt/\dz + Bz dx/\dy - By dx/\dz
             + Bx dy/\dz
            sage: ssF == -F  # must hold for a Lorentzian metric in dimension 4
            True

        Test of the standard identity

        .. MATH::

            *(A\wedge B) = \epsilon(A^\sharp, B^\sharp, ., .)

        where `A` and `B` are any 1-forms and `A^\sharp` and `B^\sharp` the
        vectors associated to them by the metric `g` (index raising)::

            sage: var('Bt Bx By Bz')
            (Bt, Bx, By, Bz)
            sage: b = M.one_form(Bt, Bx, By, Bz, name='B')
            sage: b.display()
            B = Bt dt + Bx dx + By dy + Bz dz
            sage: epsilon = g.volume_form()
            sage: g.hodge_star(a.wedge(b)) == epsilon.contract(0,a.up(g)).contract(0,b.up(g))
            True

        """
        from sage.functions.other import factorial
        from sage.tensor.modules.format_utilities import format_unop_txt, format_unop_latex
        p = pform.tensor_type()[1]
        eps = self.volume_form(p)
        if p == 0:
            dom_resu = self._domain.intersection(pform.domain())
            resu = pform.restrict(dom_resu) * eps.restrict(dom_resu)
        else:
            args = list(range(p)) + [eps] + list(range(p))
            resu = pform.contract(*args)
        if p > 1:
            resu = resu / factorial(p)
        resu.set_name(name=format_unop_txt('*', pform._name),
                    latex_name=format_unop_latex(r'\star ', pform._latex_name))
        return resu


class SymplecticFormParal(SymplecticForm, DiffFormParal):
    r"""
    Pseudo-Riemannian metric with values on a parallelizable manifold.

    An instance of this class is a field of nondegenerate symmetric bilinear
    forms (metric field) along a differentiable manifold `U` with values in a
    parallelizable manifold `M` over `\RR`, via a differentiable mapping
    `\Phi: U \rightarrow M`. The standard case of a metric field *on* a
    manifold corresponds to `U=M` and `\Phi = \mathrm{Id}_M`. Other common
    cases are `\Phi` being an immersion and `\Phi` being a curve in `M` (`U` is
    then an open interval of `\RR`).

    A *metric* `g` is a field on `U`, such that at each
    point `p\in U`, `g(p)` is a bilinear map of the type:

    .. MATH::

        g(p):\ T_q M\times T_q M  \longrightarrow \RR

    where `T_q M` stands for the tangent space to manifold `M` at the point
    `q=\Phi(p)`, such that `g(p)` is symmetric:
    `\forall (u,v)\in  T_q M\times T_q M, \ g(p)(v,u) = g(p)(u,v)`
    and nondegenerate:
    `(\forall v\in T_q M,\ \ g(p)(u,v) = 0) \Longrightarrow u=0`.

    .. NOTE::

        If `M` is not parallelizable, the class :class:`PseudoRiemannianMetric`
        should be used instead.

    INPUT:

    - ``vector_field_module`` -- free module `\mathfrak{X}(U,\Phi)` of vector
      fields along `U` with values on `\Phi(U)\subset M`
    - ``name`` -- name given to the metric
    - ``signature`` -- (default: ``None``) signature `S` of the metric as a
      single integer: `S = n_+ - n_-`, where `n_+` (resp. `n_-`) is the number
      of positive terms (resp. number of negative terms) in any diagonal
      writing of the metric components; if ``signature`` is ``None``, `S` is
      set to the dimension of manifold `M` (Riemannian signature)
    - ``latex_name`` -- (default: ``None``) LaTeX symbol to denote the metric;
      if ``None``, it is formed from ``name``

    EXAMPLES:

    Metric on a 2-dimensional manifold::

        sage: M = Manifold(2, 'M', start_index=1)
        sage: c_xy.<x,y> = M.chart()
        sage: g = M.metric('g') ; g
        Riemannian metric g on the 2-dimensional differentiable manifold M
        sage: latex(g)
        g

    A metric is a special kind of tensor field and therefore inheritates all the
    properties from class
    :class:`~sage.manifolds.differentiable.tensorfield.TensorField`::

        sage: g.parent()
        Free module T^(0,2)(M) of type-(0,2) tensors fields on the
         2-dimensional differentiable manifold M
        sage: g.tensor_type()
        (0, 2)
        sage: g.symmetries()  # g is symmetric:
        symmetry: (0, 1);  no antisymmetry

    Setting the metric components in the manifold's default frame::

        sage: g[1,1], g[1,2], g[2,2] = 1+x, x*y, 1-x
        sage: g[:]
        [ x + 1    x*y]
        [   x*y -x + 1]
        sage: g.display()
        g = (x + 1) dx*dx + x*y dx*dy + x*y dy*dx + (-x + 1) dy*dy

    Metric components in a frame different from the manifold's default one::

        sage: c_uv.<u,v> = M.chart()  # new chart on M
        sage: xy_to_uv = c_xy.transition_map(c_uv, [x+y, x-y]) ; xy_to_uv
        Change of coordinates from Chart (M, (x, y)) to Chart (M, (u, v))
        sage: uv_to_xy = xy_to_uv.inverse() ; uv_to_xy
        Change of coordinates from Chart (M, (u, v)) to Chart (M, (x, y))
        sage: M.atlas()
        [Chart (M, (x, y)), Chart (M, (u, v))]
        sage: M.frames()
        [Coordinate frame (M, (d/dx,d/dy)), Coordinate frame (M, (d/du,d/dv))]
        sage: g[c_uv.frame(),:]  # metric components in frame c_uv.frame() expressed in M's default chart (x,y)
        [ 1/2*x*y + 1/2          1/2*x]
        [         1/2*x -1/2*x*y + 1/2]
        sage: g.display(c_uv.frame())
        g = (1/2*x*y + 1/2) du*du + 1/2*x du*dv + 1/2*x dv*du
         + (-1/2*x*y + 1/2) dv*dv
        sage: g[c_uv.frame(),:,c_uv]   # metric components in frame c_uv.frame() expressed in chart (u,v)
        [ 1/8*u^2 - 1/8*v^2 + 1/2            1/4*u + 1/4*v]
        [           1/4*u + 1/4*v -1/8*u^2 + 1/8*v^2 + 1/2]
        sage: g.display(c_uv.frame(), c_uv)
        g = (1/8*u^2 - 1/8*v^2 + 1/2) du*du + (1/4*u + 1/4*v) du*dv
         + (1/4*u + 1/4*v) dv*du + (-1/8*u^2 + 1/8*v^2 + 1/2) dv*dv

    As a shortcut of the above command, on can pass just the chart ``c_uv``
    to ``display``, the vector frame being then assumed to be the coordinate
    frame associated with the chart::

        sage: g.display(c_uv)
        g = (1/8*u^2 - 1/8*v^2 + 1/2) du*du + (1/4*u + 1/4*v) du*dv
         + (1/4*u + 1/4*v) dv*du + (-1/8*u^2 + 1/8*v^2 + 1/2) dv*dv

    The inverse metric is obtained via :meth:`inverse`::

        sage: ig = g.inverse() ; ig
        Tensor field inv_g of type (2,0) on the 2-dimensional differentiable
         manifold M
        sage: ig[:]
        [ (x - 1)/(x^2*y^2 + x^2 - 1)      x*y/(x^2*y^2 + x^2 - 1)]
        [     x*y/(x^2*y^2 + x^2 - 1) -(x + 1)/(x^2*y^2 + x^2 - 1)]
        sage: ig.display()
        inv_g = (x - 1)/(x^2*y^2 + x^2 - 1) d/dx*d/dx
         + x*y/(x^2*y^2 + x^2 - 1) d/dx*d/dy + x*y/(x^2*y^2 + x^2 - 1) d/dy*d/dx
         - (x + 1)/(x^2*y^2 + x^2 - 1) d/dy*d/dy

    """
    _poisson: TensorFieldParal

    def __init__(self, manifold: Union[VectorFieldModule, DifferentiableManifold], name: Optional[str], latex_name: Optional[str] = None):
        r"""
        Construct a metric on a parallelizable manifold.

        TESTS::

            sage: M = Manifold(2, 'M')
            sage: X.<x,y> = M.chart()  # makes M parallelizable
            sage: XM = M.vector_field_module()
            sage: from sage.manifolds.differentiable.metric import \
            ....:                                   PseudoRiemannianMetricParal
            sage: g = PseudoRiemannianMetricParal(XM, 'g', signature=0); g
            Lorentzian metric g on the 2-dimensional differentiable manifold M
            sage: g[0,0], g[1,1] = -(1+x^2), 1+y^2
            sage: TestSuite(g).run(skip='_test_category')

        .. TODO::

            - add a specific parent to the metrics, to fit with the category
              framework

        """
        try:
            vector_field_module = manifold.vector_field_module()
        except AttributeError:
            vector_field_module = manifold

        if name is None:
            name = "omega"

        DiffFormParal.__init__(self, vector_field_module, 2, name=name, latex_name=latex_name)
        
        # Check that manifold is even dimensional
        dim = self._ambient_domain.dimension()
        if dim % 2 == 1:
            raise ValueError(f"the dimension of the manifold must be even but it is {dim}")
        
        # Initialization of derived quantities
        SymplecticFormParal._init_derived(self)

    def _init_derived(self):
        r"""
        Initialize the derived quantities.

        TESTS::

            sage: M = Manifold(3, 'M')
            sage: X.<x,y,z> = M.chart()  # makes M parallelizable
            sage: g = M.metric('g')
            sage: g._init_derived()

        """
        # Initialization of quantities pertaining to mother classes
        DiffFormParal._init_derived(self)
        SymplecticForm._init_derived(self)

    def _del_derived(self, del_restrictions: bool = True):
        r"""
        Delete the derived quantities.

        INPUT:

        - ``del_restrictions`` -- (default: True) determines whether the
          restrictions of ``self`` to subdomains are deleted.

        TESTS::

            sage: M = Manifold(3, 'M')
            sage: X.<x,y,z> = M.chart()  # makes M parallelizable
            sage: g = M.metric('g')
            sage: g._del_derived(del_restrictions=False)
            sage: g._del_derived()

        """
        # Delete derived quantities from mother classes
        DiffFormParal._del_derived(self, del_restrictions=del_restrictions)

        # Clear the Poisson tensor
        if self._poisson is not None:
            self._poisson._components.clear()
            self._poisson._del_derived()

        SymplecticForm._del_derived(self)

    def restrict(self, subdomain: DifferentiableManifold, dest_map: Optional[DiffMap] = None) -> 'SymplecticFormParal':
        r"""
        Return the restriction of the metric to some subdomain.

        If the restriction has not been defined yet, it is constructed here.

        INPUT:

        - ``subdomain`` -- open subset `U` of ``self._domain``
        - ``dest_map`` -- (default: ``None``) smooth destination map
          `\Phi:\ U \rightarrow V`, where `V` is a subdomain of
          ``self.codomain``
          If None, the restriction of ``self._vmodule._dest_map`` to `U` is
          used.

        OUTPUT:

        - the restricted symplectic form.

        EXAMPLES:

        Restriction of a Lorentzian metric on `\RR^2` to the upper half plane::

            sage: M = Manifold(2, 'M')
            sage: X.<x,y> = M.chart()
            sage: g = M.lorentzian_metric('g')
            sage: g[0,0], g[1,1] = -1, 1
            sage: U = M.open_subset('U', coord_def={X: y>0})
            sage: gU = g.restrict(U); gU
            Lorentzian metric g on the Open subset U of the 2-dimensional
             differentiable manifold M
            sage: gU.signature()
            0
            sage: gU.display()
            g = -dx*dx + dy*dy

        """
        if subdomain == self._domain:
            return self
        if subdomain not in self._restrictions:
            # Construct the restriction at the tensor field level:
            resu = DiffFormParal.restrict(self, subdomain, dest_map=dest_map)

            self._restrictions[subdomain] = SymplecticFormParal.wrap(resu)
        return self._restrictions[subdomain]


    @classmethod
    def wrap(cls, form:DiffFormParal, name: Optional[str] = None, latex_name: Optional[str] = None) -> 'SymplecticFormParal':
        r"""
        Define the metric from a field of symmetric bilinear forms.

        INPUT:

        - ``symbiform`` -- instance of
          :class:`~sage.manifolds.differentiable.tensorfield_paral.TensorFieldParal`
          representing a field of symmetric bilinear forms

        EXAMPLES::

            sage: M = Manifold(2, 'M')
            sage: X.<x,y> = M.chart()
            sage: s = M.sym_bilin_form_field(name='s')
            sage: s[0,0], s[0,1], s[1,1] = 1+x^2, x*y, 1+y^2
            sage: g = M.metric('g')
            sage: g.set(s)
            sage: g.display()
            g = (x^2 + 1) dx*dx + x*y dx*dy + x*y dy*dx + (y^2 + 1) dy*dy

        """
        if form.degree() != 2:
            raise TypeError("the argument must be a form of degree 2")
    
        if name is None:
            name = form._name
        if latex_name is None:
            latex_name = form._latex_name

        symplectic_form = cls(form.base_module(), name, latex_name)
        for frame in form._components:
            symplectic_form._components[frame] = form._components[frame].copy()
        for dom, form_rst in form._restrictions.items():
            symplectic_form._restrictions[dom] = SymplecticFormParal.wrap(form_rst)
        return symplectic_form

    def poisson(self, expansion_symbol=None, order=1) -> TensorFieldParal:
        r"""
        Return the inverse metric.

        INPUT:

        - ``expansion_symbol`` -- (default: ``None``) symbolic variable; if
          specified, the inverse will be expanded in power series with respect
          to this variable (around its zero value)
        - ``order`` -- integer (default: 1); the order of the expansion
          if ``expansion_symbol`` is not ``None``; the *order* is defined as
          the degree of the polynomial representing the truncated power series
          in ``expansion_symbol``; currently only first order inverse is
          supported

        If ``expansion_symbol`` is set, then the zeroth order metric must be
        invertible. Moreover, subsequent calls to this method will return
        a cached value, even when called with the default value (to enable
        computation of derived quantities). To reset, use :meth:`_del_derived`.

        OUTPUT:

        - Poisson tensor (which has degree (2,0)) associated to the symplectic form

        EXAMPLES:

        Inverse metric on a 2-dimensional manifold::

            sage: M = Manifold(2, 'M', start_index=1)
            sage: c_xy.<x,y> = M.chart()
            sage: g = M.metric('g')
            sage: g[1,1], g[1,2], g[2,2] = 1+x, x*y, 1-x
            sage: g[:]  # components in the manifold's default frame
            [ x + 1    x*y]
            [   x*y -x + 1]
            sage: ig = g.inverse() ; ig
            Tensor field inv_g of type (2,0) on the 2-dimensional
              differentiable manifold M
            sage: ig[:]
            [ (x - 1)/(x^2*y^2 + x^2 - 1)      x*y/(x^2*y^2 + x^2 - 1)]
            [     x*y/(x^2*y^2 + x^2 - 1) -(x + 1)/(x^2*y^2 + x^2 - 1)]

        If the metric is modified, the inverse metric is automatically updated::

            sage: g[1,2] = 0 ; g[:]
            [ x + 1      0]
            [     0 -x + 1]
            sage: g.inverse()[:]
            [ 1/(x + 1)          0]
            [         0 -1/(x - 1)]

        Using SymPy as symbolic engine::

            sage: M.set_calculus_method('sympy')
            sage: g[1,1], g[1,2], g[2,2] = 1+x, x*y, 1-x
            sage: g[:]  # components in the manifold's default frame
            [x + 1   x*y]
            [  x*y 1 - x]
            sage: g.inverse()[:]
            [ (x - 1)/(x**2*y**2 + x**2 - 1)      x*y/(x**2*y**2 + x**2 - 1)]
            [     x*y/(x**2*y**2 + x**2 - 1) -(x + 1)/(x**2*y**2 + x**2 - 1)]

        Demonstration of the series expansion capabilities::

            sage: M = Manifold(4, 'M', structure='Lorentzian')
            sage: C.<t,x,y,z> = M.chart()
            sage: e = var('e')
            sage: g = M.metric()
            sage: h = M.tensor_field(0, 2, sym=(0,1))
            sage: g[0, 0], g[1, 1], g[2, 2], g[3, 3] = -1, 1, 1, 1
            sage: h[0, 1], h[1, 2], h[2, 3] = 1, 1, 1
            sage: g.set(g + e*h)

        If ``e`` is a small parameter, ``g`` is a tridiagonal approximation of
        the Minkowski metric::

            sage: g[:]
            [-1  e  0  0]
            [ e  1  e  0]
            [ 0  e  1  e]
            [ 0  0  e  1]

        The inverse, truncated to first order in ``e``, is::

            sage: g.inverse(expansion_symbol=e)[:]
            [-1  e  0  0]
            [ e  1 -e  0]
            [ 0 -e  1 -e]
            [ 0  0 -e  1]

        If ``inverse()`` is called subsequently, the result will be the same.
        This allows for all computations to be made to first order::

            sage: g.inverse()[:]
            [-1  e  0  0]
            [ e  1 -e  0]
            [ 0 -e  1 -e]
            [ 0  0 -e  1]

        """
        super().poisson()

        if expansion_symbol is not None:
            if (self._poisson is not None and bool(self._poisson._components)
                and list(self._poisson._components.values())[0][0,0]._expansion_symbol
                    == expansion_symbol
                and list(self._poisson._components.values())[0][0,0]._order == order):
                return self._poisson

            if order != 1:
                raise NotImplementedError("only first order inverse is implemented")
            decompo = self.series_expansion(expansion_symbol, order)
            g0 = decompo[0]
            g1 = decompo[1]

            g0m = self._new_instance()   # needed because only metrics have
            g0m.set_comp()[:] = g0[:]    # an "inverse" method.

            contraction = g1.contract(0, g0m.inverse(), 0)
            contraction = contraction.contract(1, g0m.inverse(), 1)
            self._poisson = - (g0m.inverse() - expansion_symbol * contraction)
            self._poisson.set_calc_order(expansion_symbol, order)
            return self._poisson

        from sage.matrix.constructor import matrix
        from sage.tensor.modules.comp import CompFullyAntiSym
        # Is the inverse metric up to date ?
        for frame in self._components:
            if frame not in self._poisson._components:
                # the computation is necessary
                fmodule = self._fmodule
                si = fmodule._sindex ; nsi = fmodule._rank + si
                dom = self._domain
                comp_poisson = CompFullyAntiSym(fmodule._ring, frame, 2, start_index=si,
                                    output_formatter=fmodule._output_formatter)
                comp_poisson_scal = {}  # dict. of scalars representing the components
                                # of the poisson tensor (keys: comp. indices)
                for i in range(si, nsi):
                    for j in range(i, nsi):   # symmetry taken into account
                        comp_poisson_scal[(i,j)] = dom.scalar_field()
                for chart in dom.top_charts():
                    # TODO: do the computation without the 'SR' enforcement
                    try:
                        self_matrix = matrix(
                                  [[self.comp(frame)[i, j, chart].expr(method='SR')
                                  for j in range(si, nsi)] for i in range(si, nsi)])
                        self_matrix_inv = self_matrix.inverse()
                    except (KeyError, ValueError):
                        continue
                    for i in range(si, nsi):
                        for j in range(i, nsi):
                            val = chart.simplify(- self_matrix_inv[i-si,j-si], method='SR')
                            comp_poisson_scal[(i,j)].add_expr(val, chart=chart)
                for i in range(si, nsi):
                    for j in range(i, nsi):
                        comp_poisson[i,j] = comp_poisson_scal[(i,j)]
                self._poisson._components[frame] = comp_poisson
        return self._poisson

#****************************************************************************************
