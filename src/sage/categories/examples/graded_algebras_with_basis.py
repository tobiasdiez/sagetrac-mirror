r"""
Examples of graded algebras with basis
"""
#*****************************************************************************
#  Copyright (C) 2010 John H. Palmieri <palmieri@math.washington.edu>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from sage.misc.cachefunc import cached_method
from sage.categories.all import GradedAlgebrasWithBasis
from sage.combinat.integer_vector_weighted import WeightedIntegerVectors
from sage.combinat.free_module import CombinatorialFreeModule

class GradedPolynomialAlgebra(CombinatorialFreeModule):
    r"""
    This class illustrates an implementation of a graded algebra
    with basis: a polynomial algebra on several generators of varying
    degrees.

    INPUT:

    - ``R`` - base ring

    - ``generators`` - list of strings defining the polynomial
      generators (optional, default ``("a", "b", "c")``)

    - ``degrees`` - a positive integer or a list of positive integers
      (optional, default ``1``).  If it is a list of positive
      integers, then it must be of the same length as ``generators``,
      and its `i`-th entry specifies the degree of the `i`-th
      generator.  If it is an integer `d`, then every generator is
      given degree `d`.

    .. note::

        This is not a very full-featured implementation of a
        polynomial algebra; you can add and multiply elements and
        compute their degrees, but not much else.  For real
        calculations, use Sage's ``PolynomialRing`` construction.

    The implementation involves the following:

    - *A data type for the algebra.* In this case, the algebra is a free module
      with a basis consisting of monomials in the generators. So it is
      constructed as a :class:`CombinatorialFreeModule
      <sage.combinat.free_module.CombinatorialFreeModule>`.
      This means that it inherits all of the functionality associated with
      such objects, including operations such as addition of elements.

    - *Objects indexing the basis elements.* A basis of the algebra consists of
      monomials in the generators, so each monomial can be represented as
      a vector of non-negative integers corresponding to the exponents of the
      generators. Luckily, such vectors can be generated by the function
      ``WeightedIntegerVectors``, so we use these to index the basis elements.

      ::

        sage: A = GradedAlgebrasWithBasis(QQ).example(); A
        An example of a graded algebra with basis: the polynomial algebra on generators ('a', 'b', 'c') of degrees (1, 1, 1) over Rational Field
        sage: A.basis().keys()
        Integer vectors weighted by [1, 1, 1]

      Since ``WeightedIntegerVectors`` is a graded enumerated set, this algebra
      inherits methods that allow us to extract the elements in the basis that
      are of degree `2` as well as homogeneous components::

        sage: A.basis().keys().category()
        Join of Category of infinite enumerated sets and Category of sets with grading
        sage: A.basis(degree=2).list()
        [c^{2}, bc, ac, b^{2}, ab, a^{2}]
        sage: A.homogeneous_component(degree=2) # todo: what this should return?
        Free module generated by Integer vectors of 2 weighted by [1, 1, 1] over Rational Field

    - *Generators of the algebra.* The method :meth:`algebra_generators` must
      return a set of elements that generator the algebra::

        sage: A.algebra_generators()
        Family (a, b, c)

    - *Identity element.* Since the identity element is an element of the basis
      (the monomial whose exponents are all zero), we define the method
      :meth:`one_on_basis` to return the zero vector, since that is the object
      that indexes the basis element corresponding to the identity element.
      The method :meth:`one` uses :meth:`one_on_basis` to compute the identity
      element::

        sage: A.one_basis()
        (0, 0, 0)
        sage: A.one()
        1

    - *Methods describing the algebra structure*. For dealing with basis
      elements, the following methods need to be specified:

      - :meth:`product_on_basis` -- describing how the compute the product of
        two basis elements;
      - :meth:`degree_on_basis` -- describing how to compute the degree of
        an element of the basis;
      - :meth:`_repr_term` (optional) -- controls how a basis element will be
        displayed on the screen.

      These methods form the building blocks for other automatically-defined
      methods. For instance, the method :meth:`product`, which computes the
      product of two arbitrary elements of the algebra, is constructed from
      :meth:`product_on_basis` by extending it linearly.

      ::

        sage: A.product_on_basis((1,0,1), (0,0,2))
        ac^{3}
        sage: (a,b,c) = A.algebra_generators()
        sage: a * (1-b)^2 * c
        ac - 2*abc + ab^{2}c

      The method :meth:`degree` uses :meth:`degree_on_basis` to compute the
      degree for an arbitrary linear combination of basis elements. Similarly,
      the method :meth:`is_homogeneous` will also use :meth:`degree_on_basis`.

        sage: A.degree_on_basis((1,0,1))
        2
        sage: (a + 3*b - c).degree()
        1
        sage: (3*a).is_homogeneous()
        True
        sage: (a^3 - b^2).is_homogeneous()
        False
        sage: ((a + c)^2).is_homogeneous()
        True

      The method :meth:`_repr_term` controls how a basis element will be
      displayed on the screen, which automatically produces the print
      representation for arbitrary elements of the algebra.

        sage: A._repr_term((1,0,1))
        'ac'

    """
    def __init__(self, base_ring, generators=("a", "b", "c"), degrees=1):
        """
        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example(); A
            An example of a graded algebra with basis: the polynomial algebra on generators ('a', 'b', 'c') of degrees (1, 1, 1) over Rational Field
            sage: A = GradedAlgebrasWithBasis(QQ).example(generators=("x", "y"), degrees=(2, 3)); A
            An example of a graded algebra with basis: the polynomial algebra on generators ('x', 'y') of degrees (2, 3) over Rational Field
            sage: TestSuite(A).run()
        """
        from sage.rings.all import Integer
        self._generators = generators
        try:
            Integer(degrees)
            if degrees <= 0:
                raise ValueError, "Degrees must be positive integers"
            degrees = [degrees] * len(generators)
        except TypeError:
            # assume degrees is a list or tuple already.
            if len(degrees) != len(generators):
                raise ValueError, "List of generators and degrees must have the same length"
            try:
                for d in degrees:
                    assert Integer(d) > 0
            except (TypeError, AssertionError):
                raise ValueError, "Degrees must be positive integers"
        self._degrees = tuple(degrees)
        CombinatorialFreeModule.__init__(self, base_ring,
                                         WeightedIntegerVectors(self._degrees),
                                         category = GradedAlgebrasWithBasis(base_ring))

    # FIXME: this is currently required, because the implementation of ``basis``
    # in CombinatorialFreeModule overrides that of GradedAlgebrasWithBasis
    basis = GradedAlgebrasWithBasis.ParentMethods.__dict__['basis']

    @cached_method
    def one_basis(self):
        """
        Returns the integer vector '(0,...,0`), which indexes the one
        of this algebra, as per
        :meth:`AlgebrasWithBasis.ParentMethods.one_basis`

        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example()
            sage: A.one_basis()
            (0, 0, 0)
            sage: A.one()
            1
        """
        # return self.basis().keys().subset(0).first()  # This is generic
        return tuple(self.basis().keys().subset(0).first())

    def product_on_basis(self, t1, t2):
        r"""
        Product of basis elements, as per
        :meth:`AlgebrasWithBasis.ParentMethods.product_on_basis`

        INPUT:

        - ``t1``, ``t2`` - tuples determining monomials (as the
          exponents of the generators) in this algebra

        OUTPUT: the product of the two corresponding monomials, as an
        element of ``self``.

        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example()
            sage: A.product_on_basis((1,0,1), (0,0,2))
            ac^{3}
            sage: (a,b,c) = A.algebra_generators()
            sage: a * (1-b)^2 * c
            ac - 2*abc + ab^{2}c
        """
        return self.monomial(t1.__class__([a+b for a,b in zip(t1, t2)]))

    def degree_on_basis(self, t):
        """
        The degree of the element determined by the tuple ``t`` in
        this graded polynomial algebra.

        INPUT:

        - ``t`` -- the index of an element of the basis of this algebra,
          i.e. an exponent vector of a monomial, written as a tuple

        Output: an integer, the degree of the corresponding monomial

        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example(generators=("x", "y"), degrees=(2, 3))
            sage: A.degree_on_basis((1,1)) # x^1 y^1
            5
            sage: A.degree_on_basis((0,3))
            9
            sage: type(A.degree_on_basis((0,3)))
            <type 'sage.rings.integer.Integer'>
        """
        return self.basis().keys().grading(t)

    @cached_method
    def algebra_generators(self):
        r"""
        Returns the generators of this algebra, as per :meth:`Algebras.ParentMethods.algebra_generators`.

        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example(); A
            An example of a graded algebra with basis: the polynomial algebra on generators ('a', 'b', 'c') of degrees (1, 1, 1) over Rational Field
            sage: A.algebra_generators()
            Family (a, b, c)
        """
        from sage.sets.family import Family
        L = len(self._generators)
        return Family([self.monomial((0,) * i + (1,) + (0,) * (L-i-1)) for i in range(L)])


    # The following makes A[n] the same as A.homogeneous_components(n).
    # While an homogeneous_components
    # method should be implemented for any graded object, whether
    # __getitem__ should be defined and whether it should be the same
    # as homogeneous_components (or return something else, e.g., an
    # element of the algebra as in the case of
    # SymmetricFunctions(QQ).schur()), should be decided on a
    # case-by-case basis.
    __getitem__ = GradedAlgebrasWithBasis.ParentMethods.__dict__['homogeneous_component']

    def _repr_(self):
        """
        Print representation

        EXAMPLES::

            sage: GradedAlgebrasWithBasis(QQ).example() # indirect doctest
            An example of a graded algebra with basis: the polynomial algebra on generators ('a', 'b', 'c') of degrees (1, 1, 1) over Rational Field
        """
        return "An example of a graded algebra with basis: the polynomial algebra on generators %s of degrees %s over %s"%(self._generators, self._degrees, self.base_ring())


    def _repr_term(self, t):
        """
        Print representation for the basis element represented by the
        tuple ``t``.  This governs the behavior of the print
        representation of all elements of the algebra.

        EXAMPLES::

            sage: A = GradedAlgebrasWithBasis(QQ).example(generators=("B", "H", "d"))
            sage: A._repr_term((0,1,2))
            'Hd^{2}'
            sage: A._repr_term((2,3,1))
            'B^{2}H^{3}d'
        """
        if len(t) == 0:
            return "0"
        if max(t) == 0:
            return "1"
        s = ""
        for e,g in zip(t, self._generators):
            if e != 0:
                if e != 1:
                    s += "%s^{%s}" % (g,e)
                else:
                    s += "%s" % g
                s = s.strip()
        return s

Example = GradedPolynomialAlgebra
