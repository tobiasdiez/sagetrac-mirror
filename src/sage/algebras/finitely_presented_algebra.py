"""
Finitely Presented Algebras

Finitely presented algebras are realized as quotients of the free algebra.
"""

#*****************************************************************************
#  Copyright (C) 2005 David Kohel <kohel@maths.usyd.edu>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#    This code is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for more details; the full text
#  is available at:
#
#                  http://www.gnu.org/licenses/
#*****************************************************************************

from sage.misc.cachefunc import cached_method
from sage.algebras.algebra import Algebra
from sage.categories.algebras import Algebras
from sage.structure.parent import Parent
from sage.algebras.algebra_element import AlgebraElement
from sage.structure.element_wrapper import ElementWrapper
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.all import ZZ
from sage.rings.infinity import infinity
from sage.rings.noncommutative_ideals import Ideal_nc

class TwoSidedAlgebraIdeal(Ideal_nc):
    """
    A two-sided ideal of an algebra.
    """
    def __init__(self, free_algebra, gens):
        """
        Initialize ``self``.
        """
        PBW = free_algebra.pbw_basis()
        from sage.algebras.pbw_algebra import TwoSidedPBWIdeal
        self._pbw_ideal = TwoSidedPBWIdeal(PBW, map(PBW, gens))
        Ideal_nc.__init__(self, free_algebra, gens, "twosided")

    def free_algebra(self):
        """
        Return the ambient free algebra of ``self``.
        """
        return self.ring()

    def partial_groebner_basis(self, d):
        """
        Return a partial Groebner basis of ``self`` up to degree ``d``.
        """
        F = self.ring()
        return tuple(map(F, self._pbw_ideal.partial_groebner_basis(d)))

    def groebner_basis(self):
        r"""
        Return a Groebner basis of ``self``.

        .. WARNING::

            This will run forever if the Groebner basis is infinite.
        """
        return self.partial_groebner_basis(infinity)

    def reduce(self, x):
        """
        Return ``x`` modulo ``self``.
        """
        F = self.ring()
        if x == F.zero():
            return x
        PBW = F.pbw_basis()
        p = PBW(x)
        return F(self._pbw_ideal.reduce(p))

class FinitelyPresentedAlgebra(Algebra, UniqueRepresentation):
    """
    A finitely presented algebra realized as a quotient of the free algebra.

    INPUT:

    - ``free_algebra`` -- the ambient free algebra
    - ``ideal`` -- the defining ideal
    - ``category`` -- (optional) the category
    """
    def __init__(self, free_algebra, ideal, names=None, category=None):
        """
        Initialize ``self``.
        """
        self._free_alg = free_algebra
        R = self._free_alg.base_ring()
        self._ideal = ideal
        category = Algebras(R).or_subcategory(category)
        if names is None:
            names = free_algebra.variable_names()
        #Parent.__init__(self, base=R, category=category)
        Algebra.__init__(self, R, names, normalize=True, category=category)

    def _repr_(self):
        """
        Return a string representation of ``self``.
        """
        return "Algebra generated by {} with relations {} over {}".format(
               self.gens(), self._ideal.gens(), self.base_ring())

    def _latex_(self):
        """
        Return a latex representation of ``self``.
        """
        from sage.misc.latex import latex
        ret = latex(self.base_ring()) + "\\langle " + latex(self._free_alg.gens())
        ret += " \mid " + latex(self._ideal.gens()) + "\\rangle"
        return ret

    def ngens(self):
        """
        Return the number of generators of ``self``.
        """
        return self._free_alg.ngens()

    @cached_method
    def gens(self):
        """
        Return the generators of ``self``.
        """
        return tuple(map(lambda x: self.element_class(self, x), self._free_alg.gens()))

    def gen(self, i):
        return self.gens()[i]

    def relations(self):
        """
        Return the relations of ``self`` in the ambient free algebra.
        """
        return self._ideal.gens()

    def defining_ideal(self):
        """
        Return the defining ideal of ``self``.
        """
        return self._ideal

    def quotient(self, I, names=None):
        """
        Return a quotient of ``self`` by the ideal ``I``.
        """
        if I.side() == 'twosided':
            Ip = self._free_alg.ideal(self._ideal.gens() + I.gens())
            return self._free_alg.quotient(Ip, names)
        return super(FinitelyPresentedAlgebra, self).quotient(I, names)

    @cached_method
    def basis(self):
        """
        Return a monomial basis of ``self`` if finite dimensional; otherwise
        this runs forever.

        EXAMPLES::

            sage: F.<x,y,z> = FreeAlgebra(QQ, 3)
            sage: PBW = F.pbw_basis()
            sage: xp,yp,zp = PBW.gens()
            sage: Ip = PBW.ideal(xp^2,yp^2,zp^4, xp*yp + yp*zp, zp*yp, yp*zp, xp*zp - zp*xp)
            sage: Qp = PBW.quotient(Ip)
            sage: Qp.basis() # long time
            (PBW[x],
             PBW[y],
             PBW[z],
             PBW[y]*PBW[x],
             PBW[z]*PBW[x],
             PBW[z]^2,
             PBW[z]^2*PBW[x],
             PBW[z]^3,
             PBW[z]^3*PBW[x])
        """
        mon = reduce(lambda x,y: x.union(set(y.monomials())), self.gens(), set([]))
        todo = set([(x,y) for x in mon for y in mon])
        while len(todo) != 0:
            x,y = todo.pop()
            n = x * y
            m = set(n.monomials())
            add = m.difference(mon)
            mon = mon.union(add)
            for x in mon:
                for y in add:
                    todo.add((x,y))
                    todo.add((y,x))
        mon.discard(self.zero())
        return tuple(sorted(mon, key=lambda x: x.value))

    class Element(AlgebraElement):
        """
        An element in a finitely presented algebra.
        """
        def __init__(self, parent, value, reduce=True):
            if reduce:
                value = parent._ideal.reduce(value)
            self.value = value
            AlgebraElement.__init__(self, parent)

        def _repr_(self):
            return repr(self.value)
        def _latex_(self):
            return latex(self.value)

        def __eq__(self,other):
            return isinstance(other, FinitelyPresentedAlgebra.Element) \
                and self.parent() == other.parent() and self.value == other.value

        def _add_(self, rhs):
            return self.__class__(self.parent(), self.value + rhs.value)

        def _sub_(self, rhs):
            return self.__class__(self.parent(), self.value - rhs.value)

        def _mul_(self, rhs):
            return self.__class__(self.parent(), self.value * rhs.value)

        def _rmul_(self, rhs):
            return self.__class__(self.parent(), rhs * self.value)

        def _lmul_(self, rhs):
            return self.__class__(self.parent(), self.value * rhs)

        def monomials(self):
            """
            Return the monomials of ``self``.
            """
            P = self.parent()
            return map(P, self.value.monomials())

