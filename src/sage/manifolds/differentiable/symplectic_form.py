r"""
Pseudo-Riemannian Metrics and Degenerate Metrics

The class :class:`PseudoRiemannianMetric` implements pseudo-Riemannian metrics
on differentiable manifolds over `\RR`. The derived class
:class:`PseudoRiemannianMetricParal` is devoted to metrics with values on a
parallelizable manifold.

The class :class:`DegenerateMetric` implements degenerate (or null or lightlike)
metrics on differentiable manifolds over `\RR`. The derived class
:class:`DegenerateMetricParal` is devoted to metrics with values on a
parallelizable manifold.

AUTHORS:

- Eric Gourgoulhon, Michal Bejger (2013-2015) : initial version
- Pablo Angulo (2016) : Schouten, Cotton and Cotton-York tensors
- Florentin Jaffredo (2018) : series expansion for the inverse metric
- Hans Fotsing Tetsing (2019) : degenerate metrics

REFERENCES:

- [KN1963]_
- [Lee1997]_
- [ONe1983]_
- [DB1996]_
- [DS2010]_

"""
# *****************************************************************************
#  Copyright (C) 2015 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#  Copyright (C) 2015 Michal Bejger <bejger@camk.edu.pl>
#  Copyright (C) 2016 Pablo Angulo <pang@cancamusa.net>
#  Copyright (C) 2018 Florentin Jaffredo <florentin.jaffredo@polytechnique.edu>
#  Copyright (C) 2019 Hans Fotsing Tetsing <hans.fotsing@aims-cameroon.org>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  https://www.gnu.org/licenses/
# *****************************************************************************
from six.moves import range
from typing import overload, Optional

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
    Pseudo-Riemannian metric with values on an open subset of a
    differentiable manifold.

    An instance of this class is a field of nondegenerate symmetric bilinear
    forms (metric field) along a differentiable manifold `U` with
    values on a differentiable manifold `M` over `\RR`, via a differentiable
    mapping `\Phi: U \rightarrow M`.
    The standard case of a metric field *on* a manifold corresponds to `U=M`
    and `\Phi = \mathrm{Id}_M`. Other common cases are `\Phi` being an
    immersion and `\Phi` being a curve in `M` (`U` is then an open interval
    of `\RR`).

    A *metric* `g` is a field on `U`, such that at each point `p\in U`, `g(p)`
    is a bilinear map of the type:

    .. MATH::

        g(p):\ T_q M\times T_q M  \longrightarrow \RR

    where `T_q M` stands for the tangent space to the
    manifold `M` at the point `q=\Phi(p)`, such that `g(p)` is symmetric:
    `\forall (u,v)\in  T_q M\times T_q M, \ g(p)(v,u) = g(p)(u,v)`
    and nondegenerate:
    `(\forall v\in T_q M,\ \ g(p)(u,v) = 0) \Longrightarrow u=0`.

    .. NOTE::

        If `M` is parallelizable, the class :class:`PseudoRiemannianMetricParal`
        should be used instead.

    INPUT:

    - ``vector_field_module`` -- module `\mathfrak{X}(U,\Phi)` of vector
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

    Standard metric on the sphere `S^2`::

        sage: M = Manifold(2, 'S^2', start_index=1)

    The two open domains covered by stereographic coordinates (North and South)::

        sage: U = M.open_subset('U') ; V = M.open_subset('V')
        sage: M.declare_union(U,V)   # S^2 is the union of U and V
        sage: c_xy.<x,y> = U.chart() ; c_uv.<u,v> = V.chart() # stereographic coord
        sage: xy_to_uv = c_xy.transition_map(c_uv, (x/(x^2+y^2), y/(x^2+y^2)),
        ....:                 intersection_name='W', restrictions1= x^2+y^2!=0,
        ....:                 restrictions2= u^2+v^2!=0)
        sage: uv_to_xy = xy_to_uv.inverse()
        sage: W = U.intersection(V) # The complement of the two poles
        sage: eU = c_xy.frame() ; eV = c_uv.frame()
        sage: c_xyW = c_xy.restrict(W) ; c_uvW = c_uv.restrict(W)
        sage: eUW = c_xyW.frame() ; eVW = c_uvW.frame()
        sage: g = M.metric('g') ; g
        Riemannian metric g on the 2-dimensional differentiable manifold S^2

    The metric is considered as a tensor field of type (0,2) on `S^2`::

        sage: g.parent()
        Module T^(0,2)(S^2) of type-(0,2) tensors fields on the 2-dimensional
         differentiable manifold S^2

    We define g by its components on domain U (factorizing them to have a nicer
    view)::

        sage: g[eU,1,1], g[eU,2,2] = 4/(1+x^2+y^2)^2, 4/(1+x^2+y^2)^2
        sage: g.display(eU)
        g = 4/(x^2 + y^2 + 1)^2 dx*dx + 4/(x^2 + y^2 + 1)^2 dy*dy

    A matrix view of the components::

        sage: g[eU,:]
        [4/(x^2 + y^2 + 1)^2                   0]
        [                  0 4/(x^2 + y^2 + 1)^2]

    The components of g on domain V expressed in terms of (u,v) coordinates are
    similar to those on domain U expressed in (x,y) coordinates, as we can
    check explicitly by asking for the component transformation on the
    common subdomain W::

        sage: g.display(eVW, c_uvW)
        g = 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) du*du
         + 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) dv*dv

    Therefore, we set::

        sage: g[eV,1,1], g[eV,2,2] = 4/(1+u^2+v^2)^2, 4/(1+u^2+v^2)^2
        sage: g[eV,1,1].factor() ; g[eV,2,2].factor()
        4/(u^2 + v^2 + 1)^2
        4/(u^2 + v^2 + 1)^2
        sage: g.display(eV)
        g = 4/(u^2 + v^2 + 1)^2 du*du + 4/(u^2 + v^2 + 1)^2 dv*dv

    At this stage, the metric is fully defined on the whole sphere. Its
    restriction to some subdomain is itself a metric (by default, it bears the
    same symbol)::

        sage: g.restrict(U)
        Riemannian metric g on the Open subset U of the 2-dimensional
         differentiable manifold S^2
        sage: g.restrict(U).parent()
        Free module T^(0,2)(U) of type-(0,2) tensors fields on the Open subset
         U of the 2-dimensional differentiable manifold S^2

    The parent of `g|_U` is a free module because is `U` is a parallelizable
    domain, contrary to `S^2`. Actually, `g` and `g|_U` have different Python
    type::

        sage: type(g)
        <class 'sage.manifolds.differentiable.metric.PseudoRiemannianMetric'>
        sage: type(g.restrict(U))
        <class 'sage.manifolds.differentiable.metric.PseudoRiemannianMetricParal'>

    As a field of bilinear forms, the metric acts on pairs of tensor fields,
    yielding a scalar field::

        sage: a = M.vector_field({eU: [x, 2+y]}, name='a')
        sage: a.add_comp_by_continuation(eV, W, chart=c_uv)
        sage: b = M.vector_field({eU: [-y, x]}, name='b')
        sage: b.add_comp_by_continuation(eV, W, chart=c_uv)
        sage: s = g(a,b) ; s
        Scalar field g(a,b) on the 2-dimensional differentiable manifold S^2
        sage: s.display()
        g(a,b): S^2 --> R
        on U: (x, y) |--> 8*x/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1)
        on V: (u, v) |--> 8*(u^3 + u*v^2)/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1)

    The inverse metric is::

        sage: ginv = g.inverse() ; ginv
        Tensor field inv_g of type (2,0) on the 2-dimensional differentiable
         manifold S^2
        sage: ginv.parent()
        Module T^(2,0)(S^2) of type-(2,0) tensors fields on the 2-dimensional
         differentiable manifold S^2
        sage: latex(ginv)
        g^{-1}
        sage: ginv.display(eU) # again the components are expanded
        inv_g = (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dx*d/dx
         + (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dy*d/dy
        sage: ginv.display(eV)
        inv_g = (1/4*u^4 + 1/4*v^4 + 1/2*(u^2 + 1)*v^2 + 1/2*u^2 + 1/4) d/du*d/du
         + (1/4*u^4 + 1/4*v^4 + 1/2*(u^2 + 1)*v^2 + 1/2*u^2 + 1/4) d/dv*d/dv

    We have::

        sage: ginv.restrict(U) is g.restrict(U).inverse()
        True
        sage: ginv.restrict(V) is g.restrict(V).inverse()
        True
        sage: ginv.restrict(W) is g.restrict(W).inverse()
        True

    The volume form (Levi-Civita tensor) associated with `g`::

        sage: eps = g.volume_form() ; eps
        2-form eps_g on the 2-dimensional differentiable manifold S^2
        sage: eps.display(eU)
        eps_g = 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) dx/\dy
        sage: eps.display(eV)
        eps_g = 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) du/\dv

    The unique non-trivial component of the volume form is nothing but the
    square root of the determinant of g in the corresponding frame::

        sage: eps[[eU,1,2]] == g.sqrt_abs_det(eU)
        True
        sage: eps[[eV,1,2]] == g.sqrt_abs_det(eV)
        True

    The Levi-Civita connection associated with the metric `g`::

        sage: nabla = g.connection() ; nabla
        Levi-Civita connection nabla_g associated with the Riemannian metric g
         on the 2-dimensional differentiable manifold S^2
        sage: latex(nabla)
        \nabla_{g}

    The Christoffel symbols `\Gamma^i_{\ \, jk}` associated with some
    coordinates::

        sage: g.christoffel_symbols(c_xy)
        3-indices components w.r.t. Coordinate frame (U, (d/dx,d/dy)), with
         symmetry on the index positions (1, 2)
        sage: g.christoffel_symbols(c_xy)[:]
        [[[-2*x/(x^2 + y^2 + 1), -2*y/(x^2 + y^2 + 1)],
          [-2*y/(x^2 + y^2 + 1), 2*x/(x^2 + y^2 + 1)]],
         [[2*y/(x^2 + y^2 + 1), -2*x/(x^2 + y^2 + 1)],
          [-2*x/(x^2 + y^2 + 1), -2*y/(x^2 + y^2 + 1)]]]
        sage: g.christoffel_symbols(c_uv)[:]
        [[[-2*u/(u^2 + v^2 + 1), -2*v/(u^2 + v^2 + 1)],
          [-2*v/(u^2 + v^2 + 1), 2*u/(u^2 + v^2 + 1)]],
         [[2*v/(u^2 + v^2 + 1), -2*u/(u^2 + v^2 + 1)],
          [-2*u/(u^2 + v^2 + 1), -2*v/(u^2 + v^2 + 1)]]]

    The Christoffel symbols are nothing but the connection coefficients w.r.t.
    the coordinate frame::

        sage: g.christoffel_symbols(c_xy) is nabla.coef(c_xy.frame())
        True
        sage: g.christoffel_symbols(c_uv) is nabla.coef(c_uv.frame())
        True

    Test that `\nabla` is the connection compatible with `g`::

        sage: t = nabla(g) ; t
        Tensor field nabla_g(g) of type (0,3) on the 2-dimensional
         differentiable manifold S^2
        sage: t.display(eU)
        nabla_g(g) = 0
        sage: t.display(eV)
        nabla_g(g) = 0
        sage: t == 0
        True

    The Riemann curvature tensor of `g`::

        sage: riem = g.riemann() ; riem
        Tensor field Riem(g) of type (1,3) on the 2-dimensional differentiable
         manifold S^2
        sage: riem.display(eU)
        Riem(g) = 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) d/dx*dy*dx*dy
         - 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) d/dx*dy*dy*dx
         - 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) d/dy*dx*dx*dy
         + 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) d/dy*dx*dy*dx
        sage: riem.display(eV)
        Riem(g) = 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) d/du*dv*du*dv
         - 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) d/du*dv*dv*du
         - 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) d/dv*du*du*dv
         + 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) d/dv*du*dv*du

    The Ricci tensor of `g`::

        sage: ric = g.ricci() ; ric
        Field of symmetric bilinear forms Ric(g) on the 2-dimensional
         differentiable manifold S^2
        sage: ric.display(eU)
        Ric(g) = 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) dx*dx
         + 4/(x^4 + y^4 + 2*(x^2 + 1)*y^2 + 2*x^2 + 1) dy*dy
        sage: ric.display(eV)
        Ric(g) = 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) du*du
         + 4/(u^4 + v^4 + 2*(u^2 + 1)*v^2 + 2*u^2 + 1) dv*dv
        sage: ric == g
        True

    The Ricci scalar of `g`::

        sage: r = g.ricci_scalar() ; r
        Scalar field r(g) on the 2-dimensional differentiable manifold S^2
        sage: r.display()
        r(g): S^2 --> R
        on U: (x, y) |--> 2
        on V: (u, v) |--> 2

    In dimension 2, the Riemann tensor can be expressed entirely in terms of
    the Ricci scalar `r`:

    .. MATH::

        R^i_{\ \, jlk} = \frac{r}{2} \left( \delta^i_{\ \, k} g_{jl}
            - \delta^i_{\ \, l} g_{jk} \right)

    This formula can be checked here, with the r.h.s. rewritten as
    `-r g_{j[k} \delta^i_{\ \, l]}`::

        sage: delta = M.tangent_identity_field()
        sage: riem == - r*(g*delta).antisymmetrize(2,3)
        True

    """
    @overload
    def __init__(self, vector_field_module: DifferentiableManifold, name: Optional[str], latex_name: Optional[str]):
        pass
    @overload
    def __init__(self, vector_field_module: VectorFieldModule, name: Optional[str], latex_name: Optional[str]):
        pass
    def __init__(self, vector_field_module, name: None, latex_name: None):
        r"""
        Construct a metric.

        TESTS::

            sage: M = Manifold(2, 'M')
            sage: U = M.open_subset('U') ; V = M.open_subset('V')
            sage: M.declare_union(U,V)   # M is the union of U and V
            sage: c_xy.<x,y> = U.chart() ; c_uv.<u,v> = V.chart()
            sage: xy_to_uv = c_xy.transition_map(c_uv, (x+y, x-y),
            ....:               intersection_name='W', restrictions1= x>0,
            ....:               restrictions2= u+v>0)
            sage: uv_to_xy = xy_to_uv.inverse()
            sage: W = U.intersection(V)
            sage: e_xy = c_xy.frame() ; e_uv = c_uv.frame()
            sage: XM = M.vector_field_module()
            sage: from sage.manifolds.differentiable.metric import \
            ....:                                        PseudoRiemannianMetric
            sage: g = PseudoRiemannianMetric(XM, 'g', signature=0); g
            Lorentzian metric g on the 2-dimensional differentiable
             manifold M
            sage: g[e_xy,0,0], g[e_xy,1,1] = -(1+x^2), 1+y^2
            sage: g.add_comp_by_continuation(e_uv, W, c_uv)
            sage: TestSuite(g).run(skip=['_test_category', '_test_pickling'])

        .. TODO::

            - fix _test_pickling (in the superclass TensorField)
            - add a specific parent to the metrics, to fit with the category
              framework

              Defines the metric from a field of symmetric bilinear forms

        INPUT:

        - ``symbiform`` -- instance of
          :class:`~sage.manifolds.differentiable.tensorfield.TensorField`
          representing a field of symmetric bilinear forms

        EXAMPLES:

        Metric defined from a field of symmetric bilinear forms on a
        non-parallelizable 2-dimensional manifold::

            sage: M = Manifold(2, 'M')
            sage: U = M.open_subset('U') ; V = M.open_subset('V')
            sage: M.declare_union(U,V)   # M is the union of U and V
            sage: c_xy.<x,y> = U.chart() ; c_uv.<u,v> = V.chart()
            sage: xy_to_uv = c_xy.transition_map(c_uv, (x+y, x-y), intersection_name='W',
            ....:                              restrictions1= x>0, restrictions2= u+v>0)
            sage: uv_to_xy = xy_to_uv.inverse()
            sage: W = U.intersection(V)
            sage: eU = c_xy.frame() ; eV = c_uv.frame()
            sage: h = M.sym_bilin_form_field(name='h')
            sage: h[eU,0,0], h[eU,0,1], h[eU,1,1] = 1+x, x*y, 1-y
            sage: h.add_comp_by_continuation(eV, W, c_uv)
            sage: h.display(eU)
            h = (x + 1) dx*dx + x*y dx*dy + x*y dy*dx + (-y + 1) dy*dy
            sage: h.display(eV)
            h = (1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2) du*du + 1/4*u du*dv
             + 1/4*u dv*du + (-1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2) dv*dv
            sage: g = M.metric('g')
            sage: g.set(h)
            sage: g.display(eU)
            g = (x + 1) dx*dx + x*y dx*dy + x*y dy*dx + (-y + 1) dy*dy
            sage: g.display(eV)
            g = (1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2) du*du + 1/4*u du*dv
             + 1/4*u dv*du + (-1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2) dv*dv

        """
        if isinstance(vector_field_module, DifferentiableManifold):
            vector_field_module = vector_field_module.vector_field_module()

        DiffForm.__init__(self, vector_field_module, 2, name=name, latex_name=latex_name)

        # Check that manifold is even dimensional
        dim = self._ambient_domain.dimension()
        if dim % 2 == 1:
            raise ValueError(f"the dimension of the manifold must be even but it is {dim}")
                
        # Initialization of derived quantities
        SymplecticForm._init_derived(self)

    def _repr_(self):
        r"""
        String representation of the object.

        TESTS::

            sage: M = Manifold(5, 'M')
            sage: g = M.metric('g')
            sage: g._repr_()
            'Riemannian metric g on the 5-dimensional differentiable manifold M'
            sage: g = M.metric('g', signature=3)
            sage: g._repr_()
            'Lorentzian metric g on the 5-dimensional differentiable manifold M'
            sage: g = M.metric('g', signature=1)
            sage: g._repr_()
            'Pseudo-Riemannian metric g on the 5-dimensional differentiable manifold M'

        """
        return self._final_repr("Symplectic form " + self._name + " ")

    def _new_instance(self):
        r"""
        Create an instance of the same class as ``self`` with the same
        signature.

        TESTS::

            sage: M = Manifold(5, 'M')
            sage: g = M.metric('g', signature=3)
            sage: g1 = g._new_instance(); g1
            Lorentzian metric unnamed metric on the 5-dimensional
             differentiable manifold M
            sage: type(g1) == type(g)
            True
            sage: g1.parent() is g.parent()
            True
            sage: g1.signature() == g.signature()
            True

        """
        return type(self)(self._vmodule, 'unnamed symplectic form',
                          latex_name=r'\mbox{unnamed symplectic form}')

    def _init_derived(self):
        r"""
        Initialize the derived quantities.

        TESTS::

            sage: M = Manifold(5, 'M')
            sage: g = M.metric('g')
            sage: g._init_derived()

        """
        # Initialization of quantities pertaining to the mother class
        DiffForm._init_derived(self)

        # Poisson tensor: TODO skew
        poisson_name = 'poisson_' + self._name
        poisson_latex_name = self._latex_name + r'^{-1}'
        self._poisson = self._vmodule.tensor((2,0), name=poisson_name,
                                             latex_name=poisson_latex_name,
                                             sym=(0,1))

        # Volume form and associated tensors
        self._vol_forms = [] 

    def _del_derived(self):
        r"""
        Delete the derived quantities.

        TESTS::

            sage: M = Manifold(5, 'M')
            sage: g = M.metric('g')
            sage: g._del_derived()

        """
        # Delete the derived quantities from the mother class
        DiffForm._del_derived(self)

        # Clear the Poisson tensor
        self._poisson._restrictions.clear()
        self._poisson._del_derived()

        # Delete the volume form and the associated tensors
        del self._vol_forms[:]        

    # TODO
    def restrict(self, subdomain, dest_map=None):
        r"""
        Return the restriction of the metric to some subdomain.

        If the restriction has not been defined yet, it is constructed here.

        INPUT:

        - ``subdomain`` -- open subset `U` of the metric's domain (must be an
          instance of :class:`~sage.manifolds.differentiable.manifold.DifferentiableManifold`)
        - ``dest_map`` -- (default: ``None``) destination map
          `\Phi:\ U \rightarrow V`, where `V` is a subdomain of
          ``self._codomain``
          (type: :class:`~sage.manifolds.differentiable.diff_map.DiffMap`)
          If None, the restriction of ``self._vmodule._dest_map`` to `U` is
          used.

        OUTPUT:

        - instance of :class:`PseudoRiemannianMetric` representing the
          restriction.

        EXAMPLES::

            sage: M = Manifold(5, 'M')
            sage: g = M.metric('g', signature=3)
            sage: U = M.open_subset('U')
            sage: g.restrict(U)
            Lorentzian metric g on the Open subset U of the
             5-dimensional differentiable manifold M
            sage: g.restrict(U).signature()
            3

        See the top documentation of :class:`PseudoRiemannianMetric` for more
        examples.

        """
        if subdomain == self._domain:
            return self
        if subdomain not in self._restrictions:
            # Construct the restriction at the tensor field level:
            resu = TensorField.restrict(self, subdomain, dest_map=dest_map)
            # the type is correctly handled by TensorField.restrict, i.e.
            # resu is of type self.__class__, but the signature is not handled
            # by TensorField.restrict; we have to set it here:
            resu._signature = self._signature
            resu._signature_pm = self._signature_pm
            resu._indic_signat = self._indic_signat
            # Restrictions of derived quantities:
            resu._inverse = self.inverse().restrict(subdomain)
            for attr in self._derived_objects:
                derived = self.__getattribute__(attr)
                if derived is not None:
                    resu.__setattr__(attr, derived.restrict(subdomain))
            if self._vol_forms != []:
                for eps in self._vol_forms:
                    resu._vol_forms.append(eps.restrict(subdomain))
            # NB: no initialization of resu._determinants nor
            # resu._sqrt_abs_dets
            # The restriction is ready:
            self._restrictions[subdomain] = resu
        return self._restrictions[subdomain]

    @classmethod
    def wrap(cls, form:DiffForm, name: Optional[str] = None, latex_name: Optional[str] = None) -> 'SymplecticForm':
        r"""
        Defines the metric from a field of symmetric bilinear forms

        INPUT:

        - ``form`` -- instance of
          :class:`~sage.manifolds.differentiable.tensorfield.TensorField`
          representing a field of symmetric bilinear forms

        EXAMPLES:

        Metric defined from a field of symmetric bilinear forms on a
        non-parallelizable 2-dimensional manifold::

            sage: M = Manifold(2, 'M')
            sage: U = M.open_subset('U') ; V = M.open_subset('V')
            sage: M.declare_union(U,V)   # M is the union of U and V
            sage: c_xy.<x,y> = U.chart() ; c_uv.<u,v> = V.chart()
            sage: xy_to_uv = c_xy.transition_map(c_uv, (x+y, x-y), intersection_name='W',
            ....:                              restrictions1= x>0, restrictions2= u+v>0)
            sage: uv_to_xy = xy_to_uv.inverse()
            sage: W = U.intersection(V)
            sage: eU = c_xy.frame() ; eV = c_uv.frame()
            sage: h = M.sym_bilin_form_field(name='h')
            sage: h[eU,0,0], h[eU,0,1], h[eU,1,1] = 1+x, x*y, 1-y
            sage: h.add_comp_by_continuation(eV, W, c_uv)
            sage: h.display(eU)
            h = (x + 1) dx*dx + x*y dx*dy + x*y dy*dx + (-y + 1) dy*dy
            sage: h.display(eV)
            h = (1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2) du*du + 1/4*u du*dv
             + 1/4*u dv*du + (-1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2) dv*dv
            sage: g = M.metric('g')
            sage: g.set(h)
            sage: g.display(eU)
            g = (x + 1) dx*dx + x*y dx*dy + x*y dy*dx + (-y + 1) dy*dy
            sage: g.display(eV)
            g = (1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2) du*du + 1/4*u du*dv
             + 1/4*u dv*du + (-1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2) dv*dv

        """
        if form.degree() != 2:
            raise TypeError("the argument must be a form of degree 2")
        
        if isinstance(form, DiffFormParal):
            return SymplecticFormParal.wrap(form, name, latex_name)

        if name is None:
            name = form._name
        if latex_name is None:
            latex_name = form._latex_name

        symplecticForm = cls(form.base_module(), name, latex_name)

        for dom, rst in form._restrictions.items():
            if isinstance(rst, DiffFormParal):
                symplecticForm._restrictions[dom] = SymplecticFormParal.wrap(rst)
            else:
                symplecticForm._restrictions[dom] = SymplecticForm.wrap(rst)
        return symplecticForm

    def poisson(self, expansion_symbol=None, order=1):
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

        - instance of
          :class:`~sage.manifolds.differentiable.tensorfield.TensorField`
          with ``tensor_type`` = (2,0) representing the inverse metric

        EXAMPLES:

        Inverse of the standard metric on the 2-sphere::

            sage: M = Manifold(2, 'S^2', start_index=1)
            sage: U = M.open_subset('U') ; V = M.open_subset('V')
            sage: M.declare_union(U,V)  # S^2 is the union of U and V
            sage: c_xy.<x,y> = U.chart() ; c_uv.<u,v> = V.chart() # stereographic coord.
            sage: xy_to_uv = c_xy.transition_map(c_uv, (x/(x^2+y^2), y/(x^2+y^2)),
            ....:                 intersection_name='W', restrictions1= x^2+y^2!=0,
            ....:                 restrictions2= u^2+v^2!=0)
            sage: uv_to_xy = xy_to_uv.inverse()
            sage: W = U.intersection(V)  # the complement of the two poles
            sage: eU = c_xy.frame() ; eV = c_uv.frame()
            sage: g = M.metric('g')
            sage: g[eU,1,1], g[eU,2,2] = 4/(1+x^2+y^2)^2, 4/(1+x^2+y^2)^2
            sage: g.add_comp_by_continuation(eV, W, c_uv)
            sage: ginv = g.inverse(); ginv
            Tensor field inv_g of type (2,0) on the 2-dimensional differentiable manifold S^2
            sage: ginv.display(eU)
            inv_g = (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dx*d/dx
             + (1/4*x^4 + 1/4*y^4 + 1/2*(x^2 + 1)*y^2 + 1/2*x^2 + 1/4) d/dy*d/dy
            sage: ginv.display(eV)
            inv_g = (1/4*u^4 + 1/4*v^4 + 1/2*(u^2 + 1)*v^2 + 1/2*u^2 + 1/4) d/du*d/du
             + (1/4*u^4 + 1/4*v^4 + 1/2*(u^2 + 1)*v^2 + 1/2*u^2 + 1/4) d/dv*d/dv

        Let us check that ``ginv`` is indeed the inverse of ``g``::

            sage: s = g.contract(ginv); s  # contraction of last index of g with first index of ginv
            Tensor field of type (1,1) on the 2-dimensional differentiable manifold S^2
            sage: s == M.tangent_identity_field()
            True

        """
        # Is the Poisson tensor up to date?
        for dom, rst in self._restrictions.items():
            self._poisson._restrictions[dom] = rst.poisson(
                                             expansion_symbol=expansion_symbol,
                                             order=order) # forces the update
                                                          # of the restriction
        return self._poisson

    def hamiltonian_vector_field(self, function: DiffScalarField) -> TensorField:
        r"""
        X_f \contr \omega + \dif f = 0
        """
        vector_field = - self.sharp(function.exterior_derivative())
        vector_field.set_name('X' + function._name, 'X_{' + function._latex_name + '}')
        return vector_field
    
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

        if form.degree() != 1:
            raise ValueError(f"the degree of the differential form must be one but it is {form.degree()}")

        vector_field = form.up(self)
        vector_field.set_name(form._name + '_sharp', form._latex_name + '^\\sharp')
        return vector_field
    
    def poisson_bracket(self, f: DiffScalarField, g: DiffScalarField) -> DiffScalarField:
        r"""
        {f, g} = \omega(X_f, X_g)
        = X_f (g) = -X_g(f) = \pi(\dif f, \dif g)

        [X_f, X_g] = X_{{f,g}}
        """
        poisson_bracket = self.contract(0, self.hamiltonian_vector_field(f)).contract(0, self.hamiltonian_vector_field(g))
        poisson_bracket.set_name(f"poisson({f._name}, {g._name})", '\\{' + f'{f._latex_name}, {g._latex_name}' + '\\}')
        return poisson_bracket

    def determinant(self, frame=None):
        r"""
        Determinant of the metric components in the specified frame.

        INPUT:

        - ``frame`` -- (default: ``None``) vector frame with
          respect to which the components `g_{ij}` of the metric are defined;
          if ``None``, the default frame of the metric's domain is used. If a
          chart is provided instead of a frame, the associated coordinate
          frame is used

        OUTPUT:

        - the determinant `\det (g_{ij})`, as an instance of
          :class:`~sage.manifolds.differentiable.scalarfield.DiffScalarField`

        EXAMPLES:

        Metric determinant on a 2-dimensional manifold::

            sage: M = Manifold(2, 'M', start_index=1)
            sage: X.<x,y> = M.chart()
            sage: g = M.metric('g')
            sage: g[1,1], g[1, 2], g[2, 2] = 1+x, x*y , 1-y
            sage: g[:]
            [ x + 1    x*y]
            [   x*y -y + 1]
            sage: s = g.determinant()  # determinant in M's default frame
            sage: s.expr()
            -x^2*y^2 - (x + 1)*y + x + 1

        A shortcut is ``det()``::

            sage: g.det() == g.determinant()
            True

        The notation ``det(g)`` can be used::

            sage: det(g) == g.determinant()
            True

        Determinant in a frame different from the default's one::

            sage: Y.<u,v> = M.chart()
            sage: ch_X_Y = X.transition_map(Y, [x+y, x-y])
            sage: ch_X_Y.inverse()
            Change of coordinates from Chart (M, (u, v)) to Chart (M, (x, y))
            sage: g.comp(Y.frame())[:, Y]
            [ 1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2                            1/4*u]
            [                           1/4*u -1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2]
            sage: g.determinant(Y.frame()).expr()
            -1/4*x^2*y^2 - 1/4*(x + 1)*y + 1/4*x + 1/4
            sage: g.determinant(Y.frame()).expr(Y)
            -1/64*u^4 - 1/64*v^4 + 1/32*(u^2 + 2)*v^2 - 1/16*u^2 + 1/4*v + 1/4

        A chart can be passed instead of a frame::

            sage: g.determinant(X) is g.determinant(X.frame())
            True
            sage: g.determinant(Y) is g.determinant(Y.frame())
            True

        The metric determinant depends on the frame::

            sage: g.determinant(X.frame()) == g.determinant(Y.frame())
            False

        Using SymPy as symbolic engine::

            sage: M.set_calculus_method('sympy')
            sage: g = M.metric('g')
            sage: g[1,1], g[1, 2], g[2, 2] = 1+x, x*y , 1-y
            sage: s = g.determinant()  # determinant in M's default frame
            sage: s.expr()
            -x**2*y**2 + x - y*(x + 1) + 1

        """
        from sage.matrix.constructor import matrix
        dom = self._domain
        if frame is None:
            frame = dom._def_frame
        if frame in dom._atlas:
            # frame is actually a chart and is changed to the associated
            # coordinate frame:
            frame = frame._frame
        if frame not in self._determinants:
            # a new computation is necessary
            resu = frame._domain.scalar_field()
            manif = self._ambient_domain
            gg = self.comp(frame)
            i1 = manif.start_index()
            for chart in gg[[i1, i1]]._express:
                # TODO: do the computation without the 'SR' enforcement
                gm = matrix( [[ gg[i, j, chart].expr(method='SR')
                            for j in manif.irange()] for i in manif.irange()] )
                detgm = chart.simplify(gm.det(), method='SR')
                resu.add_expr(detgm, chart=chart)
            self._determinants[frame] = resu
        return self._determinants[frame]

    det = determinant

    def sqrt_abs_det(self, frame=None):
        r"""
        Square root of the absolute value of the determinant of the metric
        components in the specified frame.

        INPUT:

        - ``frame`` -- (default: ``None``) vector frame with
          respect to which the components `g_{ij}` of ``self`` are defined;
          if ``None``, the domain's default frame is used. If a chart is
          provided, the associated coordinate frame is used

        OUTPUT:

        - `\sqrt{|\det (g_{ij})|}`, as an instance of
          :class:`~sage.manifolds.differentiable.scalarfield.DiffScalarField`

        EXAMPLES:

        Standard metric in the Euclidean space `\RR^3` with spherical
        coordinates::

            sage: M = Manifold(3, 'M', start_index=1)
            sage: U = M.open_subset('U') # the complement of the half-plane (y=0, x>=0)
            sage: c_spher.<r,th,ph> = U.chart(r'r:(0,+oo) th:(0,pi):\theta ph:(0,2*pi):\phi')
            sage: g = U.metric('g')
            sage: g[1,1], g[2,2], g[3,3] = 1, r^2, (r*sin(th))^2
            sage: g.display()
            g = dr*dr + r^2 dth*dth + r^2*sin(th)^2 dph*dph
            sage: g.sqrt_abs_det().expr()
            r^2*sin(th)

        Metric determinant on a 2-dimensional manifold::

            sage: M = Manifold(2, 'M', start_index=1)
            sage: X.<x,y> = M.chart()
            sage: g = M.metric('g')
            sage: g[1,1], g[1, 2], g[2, 2] = 1+x, x*y , 1-y
            sage: g[:]
            [ x + 1    x*y]
            [   x*y -y + 1]
            sage: s = g.sqrt_abs_det() ; s
            Scalar field on the 2-dimensional differentiable manifold M
            sage: s.expr()
            sqrt(-x^2*y^2 - (x + 1)*y + x + 1)

        Determinant in a frame different from the default's one::

            sage: Y.<u,v> = M.chart()
            sage: ch_X_Y = X.transition_map(Y, [x+y, x-y])
            sage: ch_X_Y.inverse()
            Change of coordinates from Chart (M, (u, v)) to Chart (M, (x, y))
            sage: g[Y.frame(),:,Y]
            [ 1/8*u^2 - 1/8*v^2 + 1/4*v + 1/2                            1/4*u]
            [                           1/4*u -1/8*u^2 + 1/8*v^2 + 1/4*v + 1/2]
            sage: g.sqrt_abs_det(Y.frame()).expr()
            1/2*sqrt(-x^2*y^2 - (x + 1)*y + x + 1)
            sage: g.sqrt_abs_det(Y.frame()).expr(Y)
            1/8*sqrt(-u^4 - v^4 + 2*(u^2 + 2)*v^2 - 4*u^2 + 16*v + 16)

        A chart can be passed instead of a frame::

            sage: g.sqrt_abs_det(Y) is g.sqrt_abs_det(Y.frame())
            True

        The metric determinant depends on the frame::

            sage: g.sqrt_abs_det(X.frame()) == g.sqrt_abs_det(Y.frame())
            False

        Using SymPy as symbolic engine::

            sage: M.set_calculus_method('sympy')
            sage: g = M.metric('g')
            sage: g[1,1], g[1, 2], g[2, 2] = 1+x, x*y , 1-y
            sage: g.sqrt_abs_det().expr()
            sqrt(-x**2*y**2 - x*y + x - y + 1)
            sage: g.sqrt_abs_det(Y.frame()).expr()
            sqrt(-x**2*y**2 - x*y + x - y + 1)/2
            sage: g.sqrt_abs_det(Y.frame()).expr(Y)
            sqrt(-u**4 + 2*u**2*v**2 - 4*u**2 - v**4 + 4*v**2 + 16*v + 16)/8

        """
        dom = self._domain
        if frame is None:
            frame = dom._def_frame
        if frame in dom._atlas:
            # frame is actually a chart and is changed to the associated
            # coordinate frame:
            frame = frame._frame
        if frame not in self._sqrt_abs_dets:
            # a new computation is necessary
            detg = self.determinant(frame)
            resu = frame._domain.scalar_field()
            for chart, funct in detg._express.items():
                x = (self._indic_signat * funct).sqrt().expr()
                resu.add_expr(x, chart=chart)
            self._sqrt_abs_dets[frame] = resu
        return self._sqrt_abs_dets[frame]

    def volume_form(self, contra=0):
        r"""
        Volume form (Levi-Civita tensor) `\epsilon` associated with the metric.

        This assumes that the manifold is orientable.

        The volume form `\epsilon` is a `n`-form (`n` being the manifold's
        dimension) such that for any vector basis `(e_i)` that is orthonormal
        with respect to the metric,

        .. MATH::

            \epsilon(e_1,\ldots,e_n) = \pm 1

        There are only two such `n`-forms, which are opposite of each other.
        The volume form `\epsilon` is selected such that the domain's default
        frame is right-handed with respect to it.

        INPUT:

        - ``contra`` -- (default: 0) number of contravariant indices of the
          returned tensor

        OUTPUT:

        - if ``contra = 0`` (default value): the volume `n`-form `\epsilon`, as
          an instance of
          :class:`~sage.manifolds.differentiable.diff_form.DiffForm`
        - if ``contra = k``, with `1\leq k \leq n`, the tensor field of type
          (k,n-k) formed from `\epsilon` by raising the first k indices with
          the metric (see method
          :meth:`~sage.manifolds.differentiable.tensorfield.TensorField.up`);
          the output is then an instance of
          :class:`~sage.manifolds.differentiable.tensorfield.TensorField`, with
          the appropriate antisymmetries, or of the subclass
          :class:`~sage.manifolds.differentiable.multivectorfield.MultivectorField`
          if `k=n`

        EXAMPLES:

        Volume form on `\RR^3` with spherical coordinates::

            sage: M = Manifold(3, 'M', start_index=1)
            sage: U = M.open_subset('U') # the complement of the half-plane (y=0, x>=0)
            sage: c_spher.<r,th,ph> = U.chart(r'r:(0,+oo) th:(0,pi):\theta ph:(0,2*pi):\phi')
            sage: g = U.metric('g')
            sage: g[1,1], g[2,2], g[3,3] = 1, r^2, (r*sin(th))^2
            sage: g.display()
            g = dr*dr + r^2 dth*dth + r^2*sin(th)^2 dph*dph
            sage: eps = g.volume_form() ; eps
            3-form eps_g on the Open subset U of the 3-dimensional
             differentiable manifold M
            sage: eps.display()
            eps_g = r^2*sin(th) dr/\dth/\dph
            sage: eps[[1,2,3]] == g.sqrt_abs_det()
            True
            sage: latex(eps)
            \epsilon_{g}

        The tensor field of components `\epsilon^i_{\ \, jk}` (``contra=1``)::

            sage: eps1 = g.volume_form(1) ; eps1
            Tensor field of type (1,2) on the Open subset U of the
             3-dimensional differentiable manifold M
            sage: eps1.symmetries()
            no symmetry;  antisymmetry: (1, 2)
            sage: eps1[:]
            [[[0, 0, 0], [0, 0, r^2*sin(th)], [0, -r^2*sin(th), 0]],
             [[0, 0, -sin(th)], [0, 0, 0], [sin(th), 0, 0]],
             [[0, 1/sin(th), 0], [-1/sin(th), 0, 0], [0, 0, 0]]]

        The tensor field of components `\epsilon^{ij}_{\ \ k}` (``contra=2``)::

            sage: eps2 = g.volume_form(2) ; eps2
            Tensor field of type (2,1) on the Open subset U of the
             3-dimensional differentiable manifold M
            sage: eps2.symmetries()
            no symmetry;  antisymmetry: (0, 1)
            sage: eps2[:]
            [[[0, 0, 0], [0, 0, sin(th)], [0, -1/sin(th), 0]],
             [[0, 0, -sin(th)], [0, 0, 0], [1/(r^2*sin(th)), 0, 0]],
             [[0, 1/sin(th), 0], [-1/(r^2*sin(th)), 0, 0], [0, 0, 0]]]

        The tensor field of components `\epsilon^{ijk}` (``contra=3``)::

            sage: eps3 = g.volume_form(3) ; eps3
            3-vector field on the Open subset U of the 3-dimensional
             differentiable manifold M
            sage: eps3.tensor_type()
            (3, 0)
            sage: eps3.symmetries()
            no symmetry;  antisymmetry: (0, 1, 2)
            sage: eps3[:]
            [[[0, 0, 0], [0, 0, 1/(r^2*sin(th))], [0, -1/(r^2*sin(th)), 0]],
             [[0, 0, -1/(r^2*sin(th))], [0, 0, 0], [1/(r^2*sin(th)), 0, 0]],
             [[0, 1/(r^2*sin(th)), 0], [-1/(r^2*sin(th)), 0, 0], [0, 0, 0]]]
            sage: eps3[1,2,3]
            1/(r^2*sin(th))
            sage: eps3[[1,2,3]] * g.sqrt_abs_det() == 1
            True

        """
        if self._vol_forms == []:
            # a new computation is necessary
            manif = self._ambient_domain
            dom = self._domain
            ndim = manif.dimension()
            # The result is constructed on the vector field module,
            # so that dest_map is taken automatically into account:
            eps = self._vmodule.alternating_form(ndim, name='eps_'+self._name,
                                latex_name=r'\epsilon_{'+self._latex_name+r'}')
            si = manif.start_index()
            ind = tuple(range(si, si+ndim))
            for frame in dom._top_frames:
                if frame.destination_map() is frame.domain().identity_map():
                    eps.add_comp(frame)[[ind]] = self.sqrt_abs_det(frame)
            self._vol_forms.append(eps)  # Levi-Civita tensor constructed
            # Tensors related to the Levi-Civita one by index rising:
            for k in range(1, ndim+1):
                epskm1 = self._vol_forms[k-1]
                epsk = epskm1.up(self, k-1)
                if k > 1:
                    # restoring the antisymmetry after the up operation:
                    epsk = epsk.antisymmetrize(*range(k))
                self._vol_forms.append(epsk)
        return self._vol_forms[contra]

    def hodge_star(self, pform):
        r"""
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
        from ...tensor.modules.format_utilities import format_unop_txt, format_unop_latex
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


#******************************************************************************

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

    def __init__(self, vector_field_module: VectorFieldModule, name:Optional[str], latex_name: Optional[str]=None):
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

    def _del_derived(self, del_restrictions:bool = True):
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
        SymplecticForm._del_derived(self)

        # Clear the Poisson tensor
        self._poisson._components.clear()
        self._poisson._del_derived()

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
