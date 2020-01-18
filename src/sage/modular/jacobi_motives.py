# -*- coding: utf-8 -*-
"""
Jacobi motives

EXAMPLES::

            sage: from sage.modular.jacobi_motives import JacobiMotive
            sage: M = JacobiMotive((1/5,)*5); M
            Jacobi Motive for (1/5),(1/5),(1/5),(1/5),(1/5)

            sage: M = JacobiMotive((1/2,)*2); M
            Jacobi Motive for (1/2),(1/2)

            sage: M = JacobiMotive((2/3,2/3),(1/3,)); M
            Jacobi Motive for (2/3),(2/3) - (1/3)


REFERENCES:

Mark Watkins, Jacobi sums and Grössencharacters, 2018, p. 111-122.

http://pmb.univ-fcomte.fr/2018/PMB_Watkins.pdf

http://magma.maths.usyd.edu.au/magma/handbook/text/1545

"""
# ****************************************************************************
#       Copyright (C) 2020     Frédéric Chapoton
#                              Kiran S. Kedlaya <kskedl@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#
#                  https://www.gnu.org/licenses/
# ****************************************************************************
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.arith.functions import lcm


class JacobiMotive(object):
    def __init__(self, positive, negative=None):
        r"""
        INPUT:

        - ``positive`` and ``negative`` -- lists of rational numbers        
        """
        if negative is None:
            negative = tuple()
        posi = tuple(QQ(z) for z in positive)
        nega = tuple(QQ(z) for z in negative)
        posi = tuple(z - z.floor() for z in posi)
        nega = tuple(z - z.floor() for z in nega)
        self._posi = tuple(sorted(z for z in posi if z not in ZZ))
        self._nega = tuple(sorted(z for z in nega if z not in ZZ))
        if (sum(self._posi) - sum(self._nega)).denominator() != 1:
            raise ValueError('sum of input is not an integer')
        self._m = lcm(z.denominator() for z in self._posi + self._nega)

    def __repr__(self):
        """
        EXAMPLES::

            sage: from sage.modular.jacobi_motives import JacobiMotive
            sage: M = JacobiMotive((1/5,)*5); M
            Jacobi Motive for (1/5),(1/5),(1/5),(1/5),(1/5)
        """
        text_posi = ",".join("({})".format(z) for z in self._posi)
        text_nega = ",".join(" - ({})".format(z) for z in self._nega)
        return "Jacobi Motive for " + text_posi + text_nega
        
    def scale(self, u):
        new_posi = tuple(u * z for z in self._posi)
        new_nega = tuple(u * z for z in self._nega)
        return JacobiMotive(new_posi, new_nega)

    def weight(self):
        """
        Return the weight of ``self``.

        EXAMPLES::

            sage: from sage.modular.jacobi_motives import JacobiMotive
            sage: M = JacobiMotive((1/5,)*5)
            sage: M.weight()
            5
        """
        one = ZZ.one()
        return sum(one for _ in self._posi) - sum(one for _ in self._nega)

    def hodge_structure(self):
        pass

    def hodge_vector(self):
        pass
    
    def effective_weight(self):
        """
        Return the effective weight of ``self``.

        This is the width of the Hodge structure.

        EXAMPLES::

            sage: from sage.modular.jacobi_motives import JacobiMotive
            sage: M = JacobiMotive((1/5,)*5)
            sage: M.effective_weight()
            3
        """
        pass
    
    def __eq__(self, other):
        return self._posi == other._posi and self._nega == other._nega

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self._posi, self._nega))
    
    def field_of_definition(self):
        """
        TO DO, see :meth:`invariant scalings`
        """
        pass

    def invariant_scalings(self):
        """
        Return the list of scalings that leave the motive invariant.

        EXAMPLES::

            sage: from sage.modular.jacobi_motives import JacobiMotive
            sage: M = JacobiMotive((1/13,3/13,9/13))
            sage: M.invariant_scalings()
            [1, 3, 9]
        """
        m = self._m
        return  [u for u in m.coprime_integers(m)
                 if self.scale(u) == self]

    def jacobi_sum(self):
        pass

    def __mul__(self, other):
        pass

    def __div__(self, other):
        pass
