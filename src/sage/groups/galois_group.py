"""
Galois groups of field extensions

AUTHORS:

- David Roe (2019): initial version
"""

from sage.groups.perm_gps.permgroup import PermutationGroup_generic
from sage.groups.perm_gps.permgroup_element import PermutationGroupElement

class GaloisGroup(PermutationGroup_generic):
    """
    The group of automorphisms of a Galois closure of a given field.

    INPUT:

    - ``field`` -- a field, separable over its base

    - ``names`` -- a string or tuple of length 1, giving a variable name for the splitting field

    - ``gc_numbering`` -- boolean, whether to express permutations in terms of the
        roots of the defining polynomial of the splitting field (versus the defining polynomial
        of the original extension).  The default value may vary based on the type of field.
    """
    def __init__(self, field, names=None, gc_numbering=False):
        self._field = field
        self._base = field.base_field()
        self._gc_numbering = gc_numbering
        if names is None:
            # add a c for Galois closure
            names = field.variable_name() + 'c'
        self._gc_names = normalize_names(1, names)
        # We do only the parts of the initialization of PermutationGroup_generic
        # that don't depend on _gens
        from sage.categories.permutation_groups import PermutationGroups
        category = PermutationGroups().FinitelyGenerated().Finite()
        # Note that we DON'T call the __init__ method for PermutationGroup_generic
        # Instead, the relevant attributes are computed lazily
        super(PermutationGroup_generic, self).__init__(category=category)

    # You should implement the following methods and lazy_attributes
    

    def top_field(self):
        return self._field

    def transitive_label(self):
        return "%sT%s" % (self._field.degree(), self.transitive_number())

    @lazy_attribute
    def _deg(self):
        """
        The number of moved points in the permutation representation.

        This will be the degree of the original number field if `_gc_numbering``
        is ``False``, or the degree of the Galois closure otherwise.

        EXAMPES::

            sage: R.<x> = ZZ[]
            sage: K.<a> = NumberField(x^5-2)
            sage: G = K.galois_group(gc_numbering=False); G
            Galois group 5T3 (5:4) with order 20 of x^5 - 2
            sage: G._deg
            5
            sage: G = K.galois_group(gc_numbering=True); G._deg
            20
        """
        if self._gc_numbering:
            return self.order()
        else:
            return self._field.degree()

    @lazy_attribute
    def _domain(self):
        """
        The integers labeling the roots on which this Galois group acts.

        EXAMPLES::

            sage: R.<x> = ZZ[]
            sage: K.<a> = NumberField(x^5-2)
            sage: G = K.galois_group(gc_numbering=False); G
            Galois group 5T3 (5:4) with order 20 of x^5 - 2
            sage: G._domain
            {1, 2, 3, 4, 5}
            sage: G = K.galois_group(gc_numbering=True); G._domain
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        """
        return FiniteEnumeratedSet(range(1, self._deg+1))

    @lazy_attribute
    def _domain_to_gap(self):
        """
        Dictionary implementing the identity (used by PermutationGroup_generic).

        EXAMPLES::

            sage: R.<x> = ZZ[]
            sage: K.<a> = NumberField(x^5-2)
            sage: G = K.galois_group(gc_numbering=False)
            sage: G._domain_to_gap[5]
            5
        """
        return dict((key, i+1) for i, key in enumerate(self._domain))

    @lazy_attribute
    def _domain_from_gap(self):
        """
        Dictionary implementing the identity (used by PermutationGroup_generic).

        EXAMPLES::

            sage: R.<x> = ZZ[]
            sage: K.<a> = NumberField(x^5-2)
            sage: G = K.galois_group(gc_numbering=True)
            sage: G._domain_from_gap[20]
            20
        """
        return dict((i+1, key) for i, key in enumerate(self._domain))
