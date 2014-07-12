# -*- coding: utf-8 -*-
r"""
Ordered Set Partitions

AUTHORS:

- Mike Hansen

- MuPAD-Combinat developers (for algorithms and design inspiration)

- Travis Scrimshaw (2013-02-28): Removed ``CombinatorialClass`` and added
  entry point through :class:`OrderedSetPartition`.

References:
-----------

.. [NoTh06] Polynomial realizations of some trialgebras,
    J.-C. Novelli and J.-Y. Thibon.

.. [BerZab] The Hopf algebras of symmetric functions and quasi-symmetric
            functions in non-commutative variables are free and co-free},
    N. Bergeron, and M. Zabrocki.

"""
#*****************************************************************************
#       Copyright (C) 2007 Mike Hansen <mhansen@gmail.com>,
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#    This code is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#  The full text of the GPL is available at:
#
#                  http://www.gnu.org/licenses/
#*****************************************************************************
from sage.combinat.shuffle import ShuffleProduct
from sage.combinat.tools import transitive_ideal
from sage.combinat.permutation import to_standard

from sage.rings.arith import factorial
import sage.rings.integer
from sage.sets.set import Set, is_Set
from sage.categories.finite_enumerated_sets import FiniteEnumeratedSets
from sage.misc.classcall_metaclass import ClasscallMetaclass
from sage.misc.misc import prod, uniq
from sage.structure.parent import Parent
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.list_clone import ClonableArray
from sage.combinat.combinatorial_map import combinatorial_map
from sage.combinat.combinat import stirling_number2
from sage.combinat.composition import Composition, Compositions
from sage.combinat.quasi_shuffle import QuasiShuffleProduct
import sage.combinat.permutation as permutation
import itertools
from functools import reduce

class OrderedSetPartition(ClonableArray):
    """
    An ordered partition of a set.

    An ordered set partition `p` of a set `s` is a list of pairwise
    disjoint nonempty subsets of `s` such that the union of these
    subsets is `s`. These subsets are called the parts of the partition.
    We represent an ordered set partition as a list of sets. By
    extension, an ordered set partition of a nonnegative integer `n` is
    the set partition of the integers from `1` to `n`. The number of
    ordered set partitions of `n` is called the `n`-th ordered Bell
    number.

    There is a natural integer composition associated with an ordered
    set partition, that is the sequence of sizes of all its parts in
    order.

    The number `T_n` of ordered set partitions of
    `\{ 1, 2, ..., n \}` is the so-called `n`-th *Fubini number* (also
    known as the `n`-th ordered Bell number; see
    :wikipedia:`Ordered Bell number`). Its exponential generating
    function is
    
    .. MATH::
    
        \sum_n {T_n \over n!} x^n = {1 \over 2-e^x}.

    (See sequence A000670 in OEIS.)

    EXAMPLES:

    There are 13 ordered set partitions of `\{1,2,3\}`::

        sage: OrderedSetPartitions(3).cardinality()
        13

    Here is the list of them::

        sage: OrderedSetPartitions(3).list()
        [[{1}, {2}, {3}],
         [{1}, {3}, {2}],
         [{2}, {1}, {3}],
         [{3}, {1}, {2}],
         [{2}, {3}, {1}],
         [{3}, {2}, {1}],
         [{1}, {2, 3}],
         [{2}, {1, 3}],
         [{3}, {1, 2}],
         [{1, 2}, {3}],
         [{1, 3}, {2}],
         [{2, 3}, {1}],
         [{1, 2, 3}]]

    There are 12 ordered set partitions of `\{1,2,3,4\}` whose underlying
    composition is `[1,2,1]`::

        sage: OrderedSetPartitions(4,[1,2,1]).list()
        [[{1}, {2, 3}, {4}],
         [{1}, {2, 4}, {3}],
         [{1}, {3, 4}, {2}],
         [{2}, {1, 3}, {4}],
         [{2}, {1, 4}, {3}],
         [{3}, {1, 2}, {4}],
         [{4}, {1, 2}, {3}],
         [{3}, {1, 4}, {2}],
         [{4}, {1, 3}, {2}],
         [{2}, {3, 4}, {1}],
         [{3}, {2, 4}, {1}],
         [{4}, {2, 3}, {1}]]

    Since :trac:`14140`, we can create an ordered set partition directly by
    :class:`OrderedSetPartition` which creates the parent object by taking the
    union of the partitions passed in. However it is recommended and
    (marginally) faster to create the parent first and then create the ordered
    set partition from that. ::

        sage: s = OrderedSetPartition([[1,3],[2,4]]); s
        [{1, 3}, {2, 4}]
        sage: s.parent()
        Ordered set partitions of {1, 2, 3, 4}

    REFERENCES:

    :wikipedia:`Ordered_partition_of_a_set`
    """
    __metaclass__ = ClasscallMetaclass

    @staticmethod
    def __classcall_private__(cls, parts):
        """
        Create a set partition from ``parts`` with the appropriate parent.

        EXAMPLES::

            sage: s = OrderedSetPartition([[1,3],[2,4]]); s
            [{1, 3}, {2, 4}]
            sage: s.parent()
            Ordered set partitions of {1, 2, 3, 4}
            sage: t = OrderedSetPartition([[2,4],[1,3]]); t
            [{2, 4}, {1, 3}]
            sage: s != t
            True
            sage: OrderedSetPartition([])
            []
        """
        P = OrderedSetPartitions( reduce(lambda x,y: x.union(y), map(Set, parts), Set([])) )
        return P.element_class(P, parts)

    def __init__(self, parent, s):
        """
        Initialize ``self``.

        EXAMPLES::

            sage: OS = OrderedSetPartitions(4)
            sage: s = OS([[1, 3], [2, 4]])
            sage: TestSuite(s).run()
        """
        ClonableArray.__init__(self, parent, map(Set, s))

    def check(self):
        """
        Check that we are a valid ordered set partition.

        EXAMPLES::

            sage: OS = OrderedSetPartitions(4)
            sage: s = OS([[1, 3], [2, 4]])
            sage: s.check()
        """
        assert self in self.parent()

    @combinatorial_map(name='to composition')
    def to_composition(self):
        r"""
        Return the integer composition whose parts are the sizes of the sets
        in ``self``.

        EXAMPLES::

            sage: S = OrderedSetPartitions(5)
            sage: x = S([[3,5,4], [1, 2]])
            sage: x.to_composition()
            [3, 2]
            sage: y = S([[3,1], [2], [5,4]])
            sage: y.to_composition()
            [2, 1, 2]
        """
        return Composition(map(len, self))

    @combinatorial_map(name='to packed word')
    def to_packed_word(self):
        """
        Return the packed word `w_1\dots w_n` such that
        for any `i`, one has `i \in \pi_{w_i}` with
        `(\pi_j)` is ``self``.

        TESTS::

            sage: OrderedSetPartition([[1,3],[2]]).to_packed_word()
            [1, 2, 1]
        """
        from sage.combinat.packed_word import PackedWord
        dic = {}
        i = 1
        for set in self:
            for p in set:
                dic[p] = i
            i += 1
        return PackedWord(dic.values())

    def __mul__(self, right):
        """
        TESTS::

            sage: import itertools
            sage: OSP = OrderedSetPartition
            sage: OSP([[1,2,3],[4]]) * OSP([[2,3],[1,4]])
            [{2, 3}, {1}, {4}]
            sage: a = OSP([[1],[2,3],[4]])
            sage: b = OSP([[1,2,3],[4]])
            sage: c = OSP([[2,3],[1,4]])
            sage: a * (b * c) == (a * b) * c
            True
            sage: for (a,b,c) in itertools.product(*itertools.tee(OrderedSetPartitions(3), 3)):
            ....:     assert (a * (b * c) == (a * b) * c)
            sage: OSP([[1,2,3,4]]) * a == a
            True
            sage: a * OSP([[1,2,3,4]]) == a
            True

        """
        return OrderedSetPartition(filter(
                lambda set: set.cardinality() > 0,
                itertools.imap(
                    lambda (setL, setR): Set(setL).intersection(Set(setR)),
                    itertools.product(self, right)
        )))

    def transformation_bergeron_zabrocki_relation(self, i):
        """
        This method implement the function `f(\Phi, i)` defined by (for
        `1 \leqslant i < l(\Phi)`):

        MATH::

            f(\Phi, i) := (\Phi_1, \cdots , \Phi_i \cup \Phi_{i+1}, \Phi_{i+2}, \cdots, \Phi_{l(\Phi)})\,.

        (see :meth:`succ_bergeron_zabrocki_relation` and _[BerZab])

        (here define from *0* to *len(self) -2* to respect the python
         convention)

        EXAMPLES::

            sage: o = OrderedSetPartition([{1,2},{3,4,5},{6}]);o
            [{1, 2}, {3, 4, 5}, {6}]
            sage: o.transformation_bergeron_zabrocki_relation(0)
            [{1, 2, 3, 4, 5}, {6}]
            sage: o.transformation_bergeron_zabrocki_relation(1)
            [{1, 2}, {3, 4, 5, 6}]
            sage: o.transformation_bergeron_zabrocki_relation(2)
            Traceback (most recent call last):
            ...
            AttributeError: `i=2` must be smaller than the length -2 of `[{1, 2}, {3, 4, 5}, {6}]`
        """
        if i >= len(self) - 1:
            raise AttributeError("`i=%d` must be smaller than the "%i +
                                 "length -2 of `%s`"%repr(self))
        return self.parent()(self[:i] + [self[i] + self[i+1]] + self[i+2:])

    def succ_zabrocki_bergeron(self):
        """
        Compute successors for the order on defined with the covering relation:

        MATH::

            \Phi \lessdot (\Phi_1, \cdots , \Phi_i \cup \Phi_{i+1}, \Phi_{i+2}, \cdots, \Phi_{l(\Phi)})

        for each `1 \leqslant i < l(\Phi)` (see _[BerZab]).

        TESTS::

            sage: o = OrderedSetPartition([{1,2},{3,4,5},{6}]);o
            [{1, 2}, {3, 4, 5}, {6}]
            sage: list(o.succ_zabrocki_bergeron())
            [[{1, 2, 3, 4, 5}, {6}], [{1, 2}, {3, 4, 5, 6}]]

        """
        for i in range(len(self)-1):
            yield self.transformation_bergeron_zabrocki_relation(i)

    def greater_zabrocki_bergeron(self):
        """
        see: :meth:`succ_zabrocki_bergeron` and _[BerZab]

        TESTS::

            sage: o = OrderedSetPartition([{1,2},{3,4,5},{6}]);o
            [{1, 2}, {3, 4, 5}, {6}]
            sage: o.greater_zabrocki_bergeron()
            [[{1, 2, 3, 4, 5, 6}],
             [{1, 2}, {3, 4, 5, 6}],
             [{1, 2, 3, 4, 5}, {6}],
             [{1, 2}, {3, 4, 5}, {6}]]
            sage: OrderedSetPartition([{1},{2},{3}]).greater_zabrocki_bergeron()
            [[{1, 2, 3}], [{1}, {2, 3}], [{1, 2}, {3}], [{1}, {2}, {3}]]
        """
        return transitive_ideal(
            OrderedSetPartition.succ_zabrocki_bergeron,
            self
        )

    def shifted_shuffle(self, other):
        """
        TESTS::

            sage: O = OrderedSetPartition
            sage: o1, o2 = O([{1}, {2}]), O([{1,2}])
            sage: list(o1.shifted_shuffle(o2))
            [[{1}, {2}, {3, 4}], [{3, 4}, {1}, {2}], [{1}, {3, 4}, {2}]]
        """
        k = len(self.parent()._set)
        return iter(ShuffleProduct(
            self, [[i + k for i in set] for set in other],
            element_constructor=OrderedSetPartition))

    def shifted_quasi_shuffle(self, other):
        """
        The generalization of the *shifted_shuffle* (..see
        :meth:`sage.combinat.permutation.Permutation_class.shifted_shuffle`)
        for *ordered set partition*:

        MATH::

            p_1\dots p_k \Cup q_1\dots q_l[m] :=
                p_1 \dot (p_2 \dots p_k \Cup (q_1 \dots q_l)[m])
                + (q_1[m]) \dot (p_1 \dots p_k \Cup (q_2 \dots q_l)[m])
                + (p_1 \cup q_1[m]) \dot (p_2 \dots p_k \Cup (q_2 \dots q_l)[m])\,.

        with `m := \max(p_1 \dots p_k)` and `q_i[m]` the set `q_i` with its elements
        are shifted by `m`.

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: list(OSP([[1]]).shifted_quasi_shuffle(OSP([[1,2]])))
            [[{1}, {2, 3}], [{2, 3}, {1}], [{1, 2, 3}]]
            sage: list(OSP([[1], [2]]).shifted_quasi_shuffle(OSP([[3],[1,2]])))
            [[{1}, {2}, {5}, {3, 4}],
             [{1}, {5}, {2}, {3, 4}],
             [{1}, {5}, {3, 4}, {2}],
             [{1}, {5}, {2, 3, 4}],
             [{1}, {2, 5}, {3, 4}],
             [{5}, {1}, {2}, {3, 4}],
             [{5}, {1}, {3, 4}, {2}],
             [{5}, {1}, {2, 3, 4}],
             [{5}, {3, 4}, {1}, {2}],
             [{5}, {1, 3, 4}, {2}],
             [{1, 5}, {2}, {3, 4}],
             [{1, 5}, {3, 4}, {2}],
             [{1, 5}, {2, 3, 4}]]
            sage: list(OSP([{1, 5}, {2, 3, 4}]).shifted_quasi_shuffle([{1,3}, {2,4}]))
            [[{1, 5}, {2, 3, 4}, {8, 6}, {9, 7}],
             [{1, 5}, {8, 6}, {2, 3, 4}, {9, 7}],
             [{1, 5}, {8, 6}, {9, 7}, {2, 3, 4}],
             [{1, 5}, {8, 6}, {9, 2, 3, 4, 7}],
             [{1, 5}, {8, 2, 3, 4, 6}, {9, 7}],
             [{8, 6}, {1, 5}, {2, 3, 4}, {9, 7}],
             [{8, 6}, {1, 5}, {9, 7}, {2, 3, 4}],
             [{8, 6}, {1, 5}, {9, 2, 3, 4, 7}],
             [{8, 6}, {9, 7}, {1, 5}, {2, 3, 4}],
             [{8, 6}, {1, 7, 5, 9}, {2, 3, 4}],
             [{8, 1, 5, 6}, {2, 3, 4}, {9, 7}],
             [{8, 1, 5, 6}, {9, 7}, {2, 3, 4}],
             [{8, 1, 5, 6}, {9, 2, 3, 4, 7}]]
            sage: list(OSP([]).shifted_quasi_shuffle([{1,3}, {2,4}]))
            [[{1, 3}, {2, 4}]]
            sage: list(OSP([{1, 5}, {2, 3, 4}]).shifted_quasi_shuffle([]))
            [[{1, 5}, {2, 3, 4}]]
        """
        k = len(self.parent()._set)
        return iter(QuasiShuffleProduct(
            self, [[i + k for i in set] for set in other],
            elem_constructor=OrderedSetPartition,
            reducer=lambda l, r: [list(l) + r]))

    def pseudo_permutohedron_succ(self):
        r"""
        Iterate the successor of the ordered set partition ``self``.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: list(OSP([[1,2,3]]).pseudo_permutohedron_succ())
            [[{2, 3}, {1}],
             [{3}, {1, 2}]]
            sage: list(OSP([[1],[2],[3]]).pseudo_permutohedron_succ())
            [[{1, 2}, {3}],
             [{1}, {2, 3}]]
            sage: list(OSP([[1],[2,3]]).pseudo_permutohedron_succ())
            [[{1, 2, 3}],
             [{1}, {3}, {2}]]
            sage: list(OSP([[3],[2],[1]]).pseudo_permutohedron_succ())
            []
            sage: list(OSP([[1],[2,3,4],[5,7],[6]]).pseudo_permutohedron_succ())
            [[{1, 2, 3, 4}, {5, 7}, {6}],
             [{1}, {2, 3, 4, 5, 7}, {6}],
             [{1}, {3, 4}, {2}, {5, 7}, {6}],
             [{1}, {4}, {2, 3}, {5, 7}, {6}],
             [{1}, {2, 3, 4}, {7}, {5}, {6}]]

        """
        #####
        # first operation M_i
        # # "the operator m_i acts on the j-th and the j+1th
        # # parentheses of a 'quasi-permutation' as follows:
        # # if each element of the j-th parenthese is smaller
        # # than all the elements of the (j+1) th, then one
        # # can merge these two parentheses into one single
        # # parenthese which contains the union of the elements
        # # of these two parentheses."
        # ## [(1), (2,3,4), (5,7), (6)]
        # ##  -- > 1 < 2 == true for i = 0 :
        # ##      [(1,2,3,4), (5,7), (6)]
        # ##  -- > 4 < 5 == true for i = 1 :
        # ##      [(1), (2,3,4,5,7), (6)]
        # ##  -- > 7 < 6 == false for i = 2.
        for i_part in range(1, len(self)):
            if max (self[i_part - 1]) < min (self[i_part]):
                yield self.parent()._element_constructor(
                    self[:i_part - 1] + [self[i_part - 1].union(self[i_part])] + self[i_part + 1:])

        #####
        # second operation S_i,j
        # # "the operator S_i,j acts on the j-th parenthese of a
        # # 'quasi-permutation' as follows : it splits this parentheses
        # # into two parentheses, the second one containing the j
        # # smallest elements of the initial parenthese and the first
        # # one containing the others
        # ## [(1), (2,3,4), (5,7), (6)]
        # ## -- > for i = 0 : too short
        # ## -- > for i = 1 : len (2,3,4) = 3 OK :
        # ##      [(1), (3,4), (2), ...]
        # ##      [(1), (4), (2,3), ...]
        # ## -- > for i = 2 : len (5,7) = 2 OK :
        # ##      [..., (7), (5), (6)]
        # ## -- > for i = 3 : too short
        for i_part in range(len(self)):
            part = sorted(self[i_part])
            for j in range(1, len(part)):
                yield self.parent()._element_constructor(
                    self[:i_part] + [uniq(part[j:]), uniq(part[:j])] + self[i_part + 1:])

    def pseudo_permutohedron_pred(self):
        """
        Iterate the predecessor of the ordered set partition ``self``.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: list(OSP([[1,2,3]]).pseudo_permutohedron_pred())
            [[{1}, {2, 3}], [{1, 2}, {3}]]
            sage: list(_[0].pseudo_permutohedron_pred())
            [[{1}, {2}, {3}]]
            sage: list(_[0].pseudo_permutohedron_pred())
            []
            sage: list(OSP([[3],[2],[1]]).pseudo_permutohedron_pred())
            [[{2, 3}, {1}], [{3}, {1, 2}]]

        """
        #####
        # first operation 'M_i^{-1}'
        for i_part in range(len(self)):
            part = sorted(self[i_part])
            for j in range(1, len(part)):
                yield self.parent()._element_constructor(
                    self[:i_part] +
                    [uniq(part[:j])] + [uniq(part[j:])] +
                    self[i_part + 1:])
        #####
        # second operation "S_i,j^{-1}"
        for i_part in range(1, len(self)):
            if min(self[i_part - 1]) > max(self[i_part]):
                yield self.parent()._element_constructor(
                    self[:i_part - 1] +
                    [self[i_part - 1] + self[i_part]] +
                    self[i_part + 1:])

    def pseudo_permutohedron_greater(self):
        """
        Iterate through a list of ordered set partition greater than or equal to ``p``
        in the pseudo-permutohedron order.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: OSP([[3],[2],[1]]).pseudo_permutohedron_greater()
            [[{3}, {2}, {1}]]
            sage: OSP([[1],[2],[3]]).pseudo_permutohedron_greater()
            [[{3}, {1}, {2}],
             [{2}, {3}, {1}],
             [{3}, {2}, {1}],
             [{1, 3}, {2}],
             [{2}, {1, 3}],
             [{3}, {1, 2}],
             [{2, 3}, {1}],
             [{1}, {3}, {2}],
             [{2}, {1}, {3}],
             [{1, 2, 3}],
             [{1}, {2, 3}],
             [{1, 2}, {3}],
             [{1}, {2}, {3}]]
        """
        return transitive_ideal(lambda x: x.pseudo_permutohedron_succ(), self)

    def pseudo_permutohedron_smaller(self):
        """
        Iterate through a list of ordered set partition smaller than or equal to ``p``
        in the pseudo-permutohedron order.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: OSP([[3],[2],[1]]).pseudo_permutohedron_smaller()
            [[{1}, {3}, {2}],
             [{1}, {2}, {3}],
             [{2}, {1}, {3}],
             [{1, 3}, {2}],
             [{1, 2}, {3}],
             [{1}, {2, 3}],
             [{2}, {1, 3}],
             [{3}, {1}, {2}],
             [{1, 2, 3}],
             [{2}, {3}, {1}],
             [{3}, {1, 2}],
             [{2, 3}, {1}],
             [{3}, {2}, {1}]]
            sage: OSP([[1],[2],[3]]).pseudo_permutohedron_smaller()
            [[{1}, {2}, {3}]]

        """
        return transitive_ideal(lambda x: x.pseudo_permutohedron_pred(), self)

    def half_inversions(self):
        """
        Return a list of the half inversions of ``self``.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: OSP([[1,2,3]]).half_inversions()
            [(1, 2), (1, 2), (2, 3)]
            sage: OSP([[1,3], [2]]).half_inversions()
            [(1, 3)]
            sage: OSP([[4,5],[1,3],[2,6,7],[8]]).half_inversions()
            [(1, 3), (2, 6), (2, 6), (4, 5), (6, 7)]
        """
        half_inv = []
        for set in self:
            l = sorted(set.list())
            n = len(l)
            half_inv.extend((l[i], l[i+1])
                    for i in range(n - 1)
                    for j in range(i + 1, n)
            )
        return sorted(half_inv)

    def inversions(self):
        """
        Return a list of the inversions of ``self``.

        ..see: :meth:`sage.combinat.packed_word.PackedWord.inversions`.

        ..see _[NoTh06] §2.6 The pseudo-permutohedron

        TESTS::

            sage: OSP = OrderedSetPartition
            sage: OSP([[1,2,3]]).inversions()
            []
            sage: OSP([[3], [1,2]]).inversions()
            [(1, 3), (2, 3)]

        """
        return self.to_packed_word().inversions()

class OrderedSetPartitions(Parent, UniqueRepresentation):
    """
    Return the combinatorial class of ordered set partitions of ``s``.

    EXAMPLES::

        sage: OS = OrderedSetPartitions([1,2,3,4]); OS
        Ordered set partitions of {1, 2, 3, 4}
        sage: OS.cardinality()
        75
        sage: OS.first()
        [{1}, {2}, {3}, {4}]
        sage: OS.last()
        [{1, 2, 3, 4}]
        sage: OS.random_element()
        [{3}, {1}, {2}, {4}]

    ::

        sage: OS = OrderedSetPartitions([1,2,3,4], [2,2]); OS
        Ordered set partitions of {1, 2, 3, 4} into parts of size [2, 2]
        sage: OS.cardinality()
        6
        sage: OS.first()
        [{1, 2}, {3, 4}]
        sage: OS.last()
        [{3, 4}, {1, 2}]
        sage: OS.list()
        [[{1, 2}, {3, 4}],
         [{1, 3}, {2, 4}],
         [{1, 4}, {2, 3}],
         [{2, 3}, {1, 4}],
         [{2, 4}, {1, 3}],
         [{3, 4}, {1, 2}]]

    ::

        sage: OS = OrderedSetPartitions("cat"); OS
        Ordered set partitions of {'a', 'c', 't'}
        sage: OS.list()
        [[{'a'}, {'c'}, {'t'}],
         [{'a'}, {'t'}, {'c'}],
         [{'c'}, {'a'}, {'t'}],
         [{'t'}, {'a'}, {'c'}],
         [{'c'}, {'t'}, {'a'}],
         [{'t'}, {'c'}, {'a'}],
         [{'a'}, {'c', 't'}],
         [{'c'}, {'a', 't'}],
         [{'t'}, {'a', 'c'}],
         [{'a', 'c'}, {'t'}],
         [{'a', 't'}, {'c'}],
         [{'c', 't'}, {'a'}],
         [{'a', 'c', 't'}]]
    """
    @staticmethod
    def __classcall_private__(cls, s, c=None):
        """
        Choose the correct parent based upon input.

        EXAMPLES::

            sage: OrderedSetPartitions(4)
            Ordered set partitions of {1, 2, 3, 4}
            sage: OrderedSetPartitions(4, [1, 2, 1])
            Ordered set partitions of {1, 2, 3, 4} into parts of size [1, 2, 1]
        """
        if isinstance(s, (int, sage.rings.integer.Integer)):
            if s < 0:
                raise ValueError("s must be non-negative")
            s = frozenset(range(1, s+1))
        else:
            s = frozenset(s)

        if c is None:
            return OrderedSetPartitions_s(s)

        if isinstance(c, (int, sage.rings.integer.Integer)):
            return OrderedSetPartitions_sn(s, c)
        if c not in Compositions(len(s)):
            raise ValueError("c must be a composition of %s"%len(s))
        return OrderedSetPartitions_scomp(s, Composition(c))

    def __init__(self, s):
        """
        Initialize ``self``.

        EXAMPLES::

            sage: OS = OrderedSetPartitions(4)
            sage: TestSuite(OS).run()
        """
        self._set = s
        Parent.__init__(self, category=FiniteEnumeratedSets())

    def _element_constructor_(self, s):
        """
        Construct an element of ``self`` from ``s``.

        EXAMPLES::

            sage: OS = OrderedSetPartitions(4)
            sage: OS([[1,3],[2,4]])
            [{1, 3}, {2, 4}]
        """
        if isinstance(s, OrderedSetPartition):
            if s.parent() == self:
                return s
            raise ValueError("Cannot convert %s into an element of %s"%(s, self))
        return self.element_class(self, list(s))

    Element = OrderedSetPartition

    def __contains__(self, x):
        """
        TESTS::

            sage: OS = OrderedSetPartitions([1,2,3,4])
            sage: all([sp in OS for sp in OS])
            True
        """
        #x must be a list
        if not isinstance(x, (OrderedSetPartition, list, tuple)):
            return False

        #The total number of elements in the list
        #should be the same as the number is self._set
        if sum(map(len, x)) != len(self._set):
            return False

        #Check to make sure each element of the list
        #is a set
        u = Set([])
        for s in x:
            if not isinstance(s, (set, frozenset)) and not is_Set(s):
                return False
            u = u.union(s)

        #Make sure that the union of all the
        #sets is the original set
        if u != Set(self._set):
            return False

        return True

class OrderedSetPartitions_s(OrderedSetPartitions):
    """
    Class of ordered partitions of a set `S`.
    """
    def _repr_(self):
        """
        TESTS::

            sage: OrderedSetPartitions([1,2,3,4])
            Ordered set partitions of {1, 2, 3, 4}
        """
        return "Ordered set partitions of %s"%Set(self._set)

    def cardinality(self):
        """
        EXAMPLES::

            sage: OrderedSetPartitions(0).cardinality()
            1
            sage: OrderedSetPartitions(1).cardinality()
            1
            sage: OrderedSetPartitions(2).cardinality()
            3
            sage: OrderedSetPartitions(3).cardinality()
            13
            sage: OrderedSetPartitions([1,2,3]).cardinality()
            13
            sage: OrderedSetPartitions(4).cardinality()
            75
            sage: OrderedSetPartitions(5).cardinality()
            541
        """
        return sum([factorial(k)*stirling_number2(len(self._set),k) for k in range(len(self._set)+1)])

    def __iter__(self):
        """
        EXAMPLES::

            sage: [ p for p in OrderedSetPartitions([1,2,3]) ]
            [[{1}, {2}, {3}],
             [{1}, {3}, {2}],
             [{2}, {1}, {3}],
             [{3}, {1}, {2}],
             [{2}, {3}, {1}],
             [{3}, {2}, {1}],
             [{1}, {2, 3}],
             [{2}, {1, 3}],
             [{3}, {1, 2}],
             [{1, 2}, {3}],
             [{1, 3}, {2}],
             [{2, 3}, {1}],
             [{1, 2, 3}]]
        """
        for x in Compositions(len(self._set)):
            for z in OrderedSetPartitions(self._set, x):
                yield self.element_class(self, z)

class OrderedSetPartitions_sn(OrderedSetPartitions):
    def __init__(self, s, n):
        """
        TESTS::

            sage: OS = OrderedSetPartitions([1,2,3,4], 2)
            sage: OS == loads(dumps(OS))
            True
        """
        OrderedSetPartitions.__init__(self, s)
        self.n = n

    def __contains__(self, x):
        """
        TESTS::

            sage: OS = OrderedSetPartitions([1,2,3,4], 2)
            sage: all([sp in OS for sp in OS])
            True
            sage: OS.cardinality()
            14
            sage: len(filter(lambda x: x in OS, OrderedSetPartitions([1,2,3,4])))
            14
        """
        return OrderedSetPartitions.__contains__(self, x) and len(x) == self.n

    def __repr__(self):
        """
        TESTS::

            sage: OrderedSetPartitions([1,2,3,4], 2)
            Ordered set partitions of {1, 2, 3, 4} into 2 parts
        """
        return "Ordered set partitions of %s into %s parts"%(Set(self._set),self.n)

    def cardinality(self):
        """
        Return the cardinality of ``self``.

        The number of ordered partitions of a set of size `n` into `k`
        parts is equal to `k! S(n,k)` where `S(n,k)` denotes the Stirling
        number of the second kind.

        EXAMPLES::

            sage: OrderedSetPartitions(4,2).cardinality()
            14
            sage: OrderedSetPartitions(4,1).cardinality()
            1
        """
        return factorial(self.n)*stirling_number2(len(self._set), self.n)

    def __iter__(self):
        """
        EXAMPLES::

            sage: [ p for p in OrderedSetPartitions([1,2,3,4], 2) ]
            [[{1, 2, 3}, {4}],
             [{1, 2, 4}, {3}],
             [{1, 3, 4}, {2}],
             [{2, 3, 4}, {1}],
             [{1, 2}, {3, 4}],
             [{1, 3}, {2, 4}],
             [{1, 4}, {2, 3}],
             [{2, 3}, {1, 4}],
             [{2, 4}, {1, 3}],
             [{3, 4}, {1, 2}],
             [{1}, {2, 3, 4}],
             [{2}, {1, 3, 4}],
             [{3}, {1, 2, 4}],
             [{4}, {1, 2, 3}]]
        """
        for x in Compositions(len(self._set),length=self.n):
            for z in OrderedSetPartitions_scomp(self._set,x):
                yield self.element_class(self, z)

class OrderedSetPartitions_scomp(OrderedSetPartitions):
    def __init__(self, s, comp):
        """
        TESTS::

            sage: OS = OrderedSetPartitions([1,2,3,4], [2,1,1])
            sage: OS == loads(dumps(OS))
            True
        """
        OrderedSetPartitions.__init__(self, s)
        self.c = Composition(comp)

    def __repr__(self):
        """
        TESTS::

            sage: OrderedSetPartitions([1,2,3,4], [2,1,1])
            Ordered set partitions of {1, 2, 3, 4} into parts of size [2, 1, 1]
        """
        return "Ordered set partitions of %s into parts of size %s"%(Set(self._set), self.c)

    def __contains__(self, x):
        """
        TESTS::

            sage: OS = OrderedSetPartitions([1,2,3,4], [2,1,1])
            sage: all([ sp in OS for sp in OS])
            True
            sage: OS.cardinality()
            12
            sage: len(filter(lambda x: x in OS, OrderedSetPartitions([1,2,3,4])))
            12
        """
        return OrderedSetPartitions.__contains__(self, x) and map(len, x) == self.c

    def cardinality(self):
        r"""
        Return the cardinality of ``self``.

        The number of ordered set partitions of a set of length `k` with
        composition shape `\mu` is equal to

        .. MATH::

            \frac{k!}{\prod_{\mu_i \neq 0} \mu_i!}.

        EXAMPLES::

            sage: OrderedSetPartitions(5,[2,3]).cardinality()
            10
            sage: OrderedSetPartitions(0, []).cardinality()
            1
            sage: OrderedSetPartitions(0, [0]).cardinality()
            1
            sage: OrderedSetPartitions(0, [0,0]).cardinality()
            1
            sage: OrderedSetPartitions(5, [2,0,3]).cardinality()
            10
        """
        return factorial(len(self._set))/prod([factorial(i) for i in self.c])

    def __iter__(self):
        """
        TESTS::

            sage: [ p for p in OrderedSetPartitions([1,2,3,4], [2,1,1]) ]
            [[{1, 2}, {3}, {4}],
             [{1, 2}, {4}, {3}],
             [{1, 3}, {2}, {4}],
             [{1, 4}, {2}, {3}],
             [{1, 3}, {4}, {2}],
             [{1, 4}, {3}, {2}],
             [{2, 3}, {1}, {4}],
             [{2, 4}, {1}, {3}],
             [{3, 4}, {1}, {2}],
             [{2, 3}, {4}, {1}],
             [{2, 4}, {3}, {1}],
             [{3, 4}, {2}, {1}]]

            sage: len(OrderedSetPartitions([1,2,3,4], [1,1,1,1]))
            24

            sage: [ x for x in OrderedSetPartitions([1,4,7], [3]) ]
            [[{1, 4, 7}]]

            sage: [ x for x in OrderedSetPartitions([1,4,7], [1,2]) ]
            [[{1}, {4, 7}], [{4}, {1, 7}], [{7}, {1, 4}]]

            sage: [ p for p in OrderedSetPartitions([], []) ]
            [[]]

            sage: [ p for p in OrderedSetPartitions([1], [1]) ]
            [[{1}]]
        """
        comp = self.c
        lset = [x for x in self._set]
        l = len(self.c)
        dcomp = [-1] + comp.descents(final_descent=True)

        p = []
        for j in range(l):
            p += [j+1]*comp[j]

        for x in permutation.Permutations(p):
            res = to_standard(x).inverse()
            res = [lset[x-1] for x in res]
            yield self.element_class( self, [ Set( res[dcomp[i]+1:dcomp[i+1]+1] ) for i in range(l)] )

##########################################################
# Deprecations

class SplitNK(OrderedSetPartitions_scomp):
    def __setstate__(self, state):
        r"""
        For unpickling old ``SplitNK`` objects.

        TESTS::

            sage: loads("x\x9ck`J.NLO\xd5K\xce\xcfM\xca\xccK,\xd1+.\xc8\xc9,"
            ....:   "\x89\xcf\xcb\xe6\n\x061\xfc\xbcA\xccBF\xcd\xc6B\xa6\xda"
            ....:   "Bf\x8dP\xa6\xf8\xbcB\x16\x88\x96\xa2\xcc\xbc\xf4b\xbd\xcc"
            ....:   "\xbc\x92\xd4\xf4\xd4\"\xae\xdc\xc4\xec\xd4x\x18\xa7\x905"
            ....:   "\x94\xd1\xb45\xa8\x90\r\xa8>\xbb\x90=\x03\xc85\x02r9J\x93"
            ....:   "\xf4\x00\xb4\xc6%f")
            Ordered set partitions of {0, 1, 2, 3, 4} into parts of size [2, 3]
        """
        self.__class__ = OrderedSetPartitions_scomp
        n = state['_n']
        k = state['_k']
        OrderedSetPartitions_scomp.__init__(self, range(state['_n']), (k,n-k))

from sage.structure.sage_object import register_unpickle_override
register_unpickle_override("sage.combinat.split_nk", "SplitNK_nk", SplitNK)

