"""
Energy Partition Tuples

AUTHORS:

- Reimundo Heluani (06-09-2020): Initial implementation.

.. linkall
"""

#******************************************************************************
#       Copyright (C) 2019 Reimundo Heluani <heluani@potuz.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************
from sage.categories.infinite_enumerated_sets import InfiniteEnumeratedSets
from sage.combinat.partition_tuple import PartitionTuples,\
                                          PartitionTuple,\
                                          RegularPartitionTuples_level

from sage.geometry.polyhedron.constructor import Polyhedron
from sage.misc.flatten import flatten
from sage.rings.integer import Integer
from sage.rings.rational_field import QQ
from sage.rings.semirings.non_negative_integer_semiring import NN
from sage.sets.family import Family
from sage.categories.finite_enumerated_sets import FiniteEnumeratedSets
import itertools
from sage.arith.functions import lcm
from sage.combinat.integer_vector import IntegerVectors
from sage.combinat.partition import Partitions
from .energy_partitions import EnergyPartitions, EnergyPartition
from sage.structure.parent import Parent
from sage.combinat.combinat import CombinatorialElement
from sage.rings.integer_ring import ZZ
from sage.structure.richcmp import op_LT, op_LE, op_GT, op_GE, op_EQ, op_NE,\
                                   richcmp_method

@richcmp_method
class EnergyPartitionTuple(PartitionTuple):
    """
    Base class of EnergyPartitionTuples Element class
    """
    Element = EnergyPartition

    def __init__(self,parent,mu):
        """
        Initialize this Energy Partition Tuple.

        TESTS::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: type(EnergyPartitionTuples((1/2,1/2),2).an_element()[0])
            <class 'sage.algebras.vertex_algebras.energy_partitions.EnergyPartitions_all_with_category.element_class'>
        """
        #override PartitionTuples's init that explicitly set Partitions
        mu = [EnergyPartitions(w,regular=r)(x) if r else EnergyPartitions(
              w)(x) for w,r,x in zip(parent._weights,parent._regular,mu)]
        CombinatorialElement.__init__(self,parent,mu)

    def energy(self):
        """
        Return the total energy of this Energy Partition Tuple.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: v = EnergyPartitionTuples((1/2,3/2),2)([[5,1],[3,3]]); v.energy()
            12

        TESTS::

            sage: EnergyPartitionTuples((1/2,1/2),2)([[],[]]).energy()
            0
        """
        return sum(x.energy() for x in self)

    def __richcmp__(self,other,op):
        """
        A comparison key between Energy Partition Tuples.

        The order of the basis is used when constructing submodules and
        quotients. In order to optimize the computations in the
        classical limits it is important to have a basis ordering which
        is compatible with the Li (or PBW in this case) filtration.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: EPT = EnergyPartitionTuples((2,3/2), 2, 9, regular=(0,2))
            sage: EPT.list()
            [([2, 1, 1, 1], []),
             ([1, 1], [3, 1]),
             ([2, 1], [2, 1]),
             ([4, 1, 1], []),
             ([3, 2, 1], []),
             ([2, 2, 2], []),
             ([1], [5, 1]),
             ([1], [4, 2]),
             ([2], [4, 1]),
             ([4], [2, 1]),
             ([3], [3, 1]),
             ([2], [3, 2]),
             ([6, 1], []),
             ([5, 2], []),
             ([4, 3], []),
             ([], [7, 1]),
             ([], [6, 2]),
             ([], [5, 3]),
             ([8], [])]
        """
        if op == op_EQ:
            return self._list == other._list
        if op == op_NE:
            return self._list != other._list
        if op == op_LT:
            if self.energy() < other.energy():
                return True
            elif self.energy() > other.energy():
                return False
            #Li filtration:
            if self.size() - sum(len(x) for x in self) < other.size() - sum(
                len(x) for x in other):
                return True
            if self.size() - sum(len(x) for x in self) > other.size() - sum(
                len(x) for x in other):
                return False
            #reverse lexicographic
            ms = max(max(x, default=0) for x in self)
            mo = max(max(x, default=0) for x in other)
            if ms < mo:
                return True
            if ms > mo:
                return False
            exps = list(self.to_exp(ms))
            exps.reverse()
            for l in exps:
                l.reverse()
            expo = list(other.to_exp(ms))
            expo.reverse()
            for l in expo:
                l.reverse()
            for i in range(ms):
                for a,b in zip(exps, expo):
                    if a[i] < b[i]:
                        return True
                    if a[i] > b[i]:
                        return False
            raise ValueError("This should not happen s:{} o:{}".format(self,other))
        if op == op_LE:
            return self < other or self == other
        if op == op_GT:
            return not (self < other)
        if op == op_GE:
            return self > other or self == other



class EnergyPartitionTuples(PartitionTuples):
    r"""
    This class models the PBW basis of an H-graded vertex algebra.
    It encodes the bijection between the usual Fourier modes
    of the `n`-th products of vectors and the *shifted* modes with
    respect to conformal weight. Let `L` be a Lie conformal algebra
    finitely generated by vectors `\{a^i\}`. Its universal enveloping
    vertex algebra `V` has a PBW basis consisting on vectors of the form

    .. MATH::
        :label:modes

        a^{i_1}_{(-n_1)}\cdots a^{i_k}_{(-n_k)}|0\rangle,

    for some integer numbers `n_i >  0`.

    If `L` is H-graded, so is `V`. Let `\{w_i\}` be the *degree* or
    *conformal weight* of the generator `a^i`. We assume each `w_i`
    to be a positive rational number. Then one defines the *shifted*
    modes `a^i_{n} = a^i_{(n+w_i-1)}`. With respect to these modes
    a general vector in `V` is written as linear combination of
    vectors of the form

    .. MATH::
        :label:modes2

        a^{i_1}_{-n_1}\cdots a^{i_k}_{-n_k}|0\rangle,

    for some **rational** `n_i > 0`. The conformal weight of this
    element is given by `n=\sum_{j=1}^k n_j`. This class implements the
    bijection between :eq:`modes` and :eq:`modes2`.

    An ``EnergyPartitionTuple`` of energy ``n``, weights
    `(w_1,\ldots,w_n)` and length `k` parametrizes elements like
    :eq:`modes2`. Its implementation is as a tuple or list of `k`
    :class:`EnergyPartition`.

    INPUT:

    - ``weights`` -- a list or tuple of positive rational numbers;
      the weights `w_i` as above. The length of this tuple has to
      equal the parameter ``level``.

    - ``level`` -- a positive integer; the number of genertors,
      or length of this tuple.

    - ``energy`` -- a non-negative rational or ``None`` (default:
      ``None``); the total energy of the tuples in this class, or all
      tuples if ``None``.

    - ``regular`` -- a non-negative integer number or a list of
      non-negative integers of length ``level``; if the `i-th` entry
      of this list is `\ell`, then the corresponding `i-th`
      ``EnergyPartition`` of this tuple is `\ell` regular. The special
      value of `\ell=0` is used to allow all Energy Partitions. If
      ``regular`` is a positive integer, it is applied to all entries.

    OUTPUT: The class of Energy Partition Tuples with the prescribed
    conditions.

    EXAMPLES:

    The vectors of the :class:`NeveuSchwarzVertexAlgebra` of energy
    ``6`` are parametrized by::

        sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
        sage: L = EnergyPartitionTuples((2,3/2),2,6,regular=(0,2)); L
        (0, 2)-Regular Energy Partition Tuples of energy 6 with level 2 and weights (2, 3/2)
        sage: l = L.list(); l
        [([], [4, 1]),
         ([], [3, 2]),
         ([5], []),
         ([1], [2, 1]),
         ([3, 1], []),
         ([2, 2], []),
         ([1, 1, 1], [])]
        sage: V = NeveuSchwarzVertexAlgebra(QQ,1/2); V
        The Neveu-Schwarz super vertex algebra at central charge 1/2
        sage: [V(v) for v in l]
        [G_-9/2G_-3/2|0>,
         G_-7/2G_-5/2|0>,
         L_-6|0>,
         L_-2G_-5/2G_-3/2|0>,
         L_-4L_-2|0>,
         L_-3L_-3|0>,
         L_-2L_-2L_-2|0>]

    The Universal Affine Kac-Moody vertex algebra of type `A_1`, has
    the following basis::

        sage: L = EnergyPartitionTuples((1,1,1),3); L
        (0, 0, 0)-Regular Energy Partition Tuples of level 3 with weights (1, 1, 1)
        sage: l = L[0:13]; l
        [([], [], []),
         ([], [], [1]),
         ([], [1], []),
         ([1], [], []),
         ([], [], [2]),
         ([], [], [1, 1]),
         ([], [2], []),
         ([], [1], [1]),
         ([], [1, 1], []),
         ([2], [], []),
         ([1], [], [1]),
         ([1], [1], []),
         ([1, 1], [], [])]
        sage: V = AffineVertexAlgebra(QQ,'A1',1, names=('e','h','f')); V
        The universal affine vertex algebra of CartanType ['A', 1] at level 1
        sage: [V(v) for v in l]
        [|0>,
         f_-1|0>,
         h_-1|0>,
         e_-1|0>,
         f_-2|0>,
         f_-1f_-1|0>,
         h_-2|0>,
         h_-1f_-1|0>,
         h_-1h_-1|0>,
         e_-2|0>,
         e_-1f_-1|0>,
         e_-1h_-1|0>,
         e_-1e_-1|0>]

    The Free Fermions have a similar structure::

        sage: L = EnergyPartitionTuples(1/2,1,regular=2); L
        (2,)-Regular Energy Partition Tuples of level 1 with weights (1/2,)
        sage: l = L[0:10]; l
        [([]),
         ([1]),
         ([2]),
         ([2, 1]),
         ([3]),
         ([3, 1]),
         ([4]),
         ([3, 2]),
         ([4, 1]),
         ([3, 2])]
        sage: V = FreeFermionsVertexAlgebra(QQ); V
        The Free Fermions super vertex algebra with generators (psi_-1/2|0>,)
        sage: [V(v) for v in l]
        [|0>,
         psi_-1/2|0>,
         psi_-3/2|0>,
         psi_-3/2psi_-1/2|0>,
         psi_-5/2|0>,
         psi_-5/2psi_-1/2|0>,
         psi_-7/2|0>,
         psi_-5/2psi_-3/2|0>,
         psi_-7/2psi_-1/2|0>,
         psi_-5/2psi_-3/2|0>]

    TESTS::

        sage: ([],[]) in EnergyPartitionTuples((1,1),2,1)
        False
        sage: ([],[]) in EnergyPartitionTuples((1,1),2,0)
        True
        sage: ([],[]) in EnergyPartitionTuples((1,1),2,0,regular=1)
        True
    """
    @staticmethod
    def __classcall_private__(cls, weights=None, level=None, energy=None,
                              regular = 0):

        if not isinstance(level,(int,Integer)) or level<1:
            raise ValueError('the level must be a positive integer')

        if not isinstance(weights,(tuple, list)):
            if weights not in QQ or weights <= 0:
                raise ValueError("weights must be either a positive rational "\
                                 "or a list of positive rationals")
            weights = tuple([weights,]*level)
        elif len(weights) != level or any \
                                    (i not in QQ or i <= 0 for i in weights):
            raise ValueError("weights must be a list of positive rationals "\
                             "of the same length as level")
        weights = Family(weights)

        if not isinstance(regular,(tuple, list)):
            if regular not in NN or regular < 0:
                raise ValueError("regular must be either a nonnegative integer"\
                                 " or a list of non-negative integers")
            regular = tuple([regular,]*level)
        elif len(regular) != level or any \
                                    (i not in NN or i < 0 for i in regular):
            raise ValueError("regular must be a list of positive integers "\
                             "of the same length as level")
        regular = Family(regular)

        if energy is not None and (energy not in QQ or energy<0):
            raise ValueError('energy must be a non-negative rational')

        if energy is None:
            return EnergyPartitionTuples_all(weights, level, regular)
        return EnergyPartitionTuples_n(weights, level, regular, energy )

    _energy=None

    def __init__(self, weights, level, regular, is_infinite=False):
        self._weights = tuple(weights)
        self._level = level
        self._regular = tuple(regular)

        if is_infinite:
            category = InfiniteEnumeratedSets()
        else:
            category = FiniteEnumeratedSets()
        super(EnergyPartitionTuples,self).__init__(category=category)

    Element = EnergyPartitionTuple

    def _element_constructor_(self, mu):
        """
        Constructs elements of this class.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((1/2,1/2),2,regular=2); V
            (2, 2)-Regular Energy Partition Tuples of level 2 with weights (1/2, 1/2)
            sage: V(([2,1],[1]))
            ([2, 1], [1])
            sage: V(([2,1],[1,1]))
            Traceback (most recent call last):
            ...
            ValueError: ([2, 1], [1, 1]) is not a (2, 2)-Regular Energy Partition Tuples of level 2 with weights (1/2, 1/2)
        """
        #override partition_tuple's constructor to get EnergyPartitions

        try:
            l = len(mu)
        except TypeError:
                raise ValueError("Do not know how to convert {} to {}".format(
                                                                       mu,self))
        if l != self._level:
            raise ValueError("Do not know how to convert {} to {}".format(mu,
                                                                         self))
        try:
            mu = [EnergyPartitions(w,regular=r)(x) if r else EnergyPartitions(
                  w)(x) for w,r,x in zip(self._weights,self._regular,mu)]
        except ValueError:
            raise ValueError('{} is not a {}'.format(mu, self))

        if mu not in self:
            raise ValueError('{} is not a {}'.format(mu, self))

        return self.element_class(self, mu)


    def __contains__(self,x):
        """
        Whether this class of Energy Partition Tuples contains ``x``.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: [[2,1],[]] in EnergyPartitionTuples((1,1),2)
            True
            sage: [[2,1],[]] in EnergyPartitionTuples((1,),1)
            False
            sage: [[]] in EnergyPartitionTuples((1,),1)
            True
        """
        return len(x) == self._level and all(p in EnergyPartitions(w,regular=r)\
            if r else EnergyPartitions(w) for p,r,w in zip(x,self._regular,
            self._weights))


class EnergyPartitionTuples_all(EnergyPartitionTuples):
    def __init__(self,weights,level, regular):
        """
        Base class for all Energy Partition Tuples.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((2,3/2),2); V
            (0, 0)-Regular Energy Partition Tuples of level 2 with weights (2, 3/2)
            sage: V[0:8]
            [([], []),
             ([], [1]),
             ([1], []),
             ([], [2]),
             ([], [1, 1]),
             ([2], []),
             ([], [3]),
             ([1], [1])]
        """
        EnergyPartitionTuples.__init__(self,weights,level,regular,True)

    def _repr_(self):
        """
        The name of this class.

        EXAMPLES::
            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((2,3/2),2); V
            (0, 0)-Regular Energy Partition Tuples of level 2 with weights (2, 3/2)
        """
        return "{}-Regular Energy Partition Tuples of level {} with weights"\
               " {}".format(self._regular, self._level,self._weights)

    def __iter__(self):
        """
        Iterate over all Energy Partition Tuples in this class.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((2,3/2),2); V[0:8]
            [([], []),
             ([], [1]),
             ([1], []),
             ([], [2]),
             ([], [1, 1]),
             ([2], []),
             ([], [3]),
             ([1], [1])]
        """
        yield self.element_class(self,[[]]*self._level)
        l = lcm([w.denominator() for w in self._weights])
        n = 1
        while True:
            ieqs = [[0]*(i+1) + [1] + [0]*(self._level-i)\
                    for i in range(self._level+1)]
            eqns = [-n]+[[l*w] for w in self._weights] + [l]
            eqns = [flatten(eqns)]
            P = Polyhedron(ieqs=ieqs, eqns=eqns)
            nlist = []
            for p in P.integral_points():
                for x in IntegerVectors(p[-1],self._level):
                    for y in itertools.product(*[EnergyPartitions(
                             w,i+j*w,regular=r,length=j) if r else
                             EnergyPartitions(w,i+j*w,length=j)for w,r,i,j\
                             in zip(self._weights,self._regular,x,p[:-1])]):
                        nlist.append(self.element_class(self,list(y)))
            for z in sorted(nlist):
                yield z
            n += 1

    def an_element(self):
        """
        An element of this class.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: EnergyPartitionTuples((1/2,1/2),2).an_element()
            ([1, 1, 1, 1], [2, 1, 1])
            sage: EnergyPartitionTuples((1/2,1/2),2,regular=2).an_element()
            ([], [])
        """
        if self._regular[0]:
            return self[0]
        return self.element_class(self,PartitionTuples.an_element(self))

    def subset(self, energy=None):
        if energy is None:
            return self
        return EnergyPartitionTuples(self._weights, self._level, energy,
                                     self._regular)

class EnergyPartitionTuples_n(EnergyPartitionTuples):
    def __init__(self,weights,level,regular,energy):
        """
        Energy Partition Tuples with a fixed energy.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((1/2,1/2),2,4,regular=2); V
            (2, 2)-Regular Energy Partition Tuples of energy 4 with level 2 and weights (1/2, 1/2)
            sage: V.list()
            [([2, 1], [2, 1]),
             ([], [4, 1]),
             ([], [3, 2]),
             ([4], [1]),
             ([3, 2], [1]),
             ([3], [2]),
             ([2], [3]),
             ([1], [4]),
             ([1], [3, 2]),
             ([4, 1], []),
             ([3, 2], [])]
            sage: V.an_element()
            ([2, 1], [2, 1])
        """
        self._energy = energy
        EnergyPartitionTuples.__init__(self,weights,level,regular)

    def _repr_(self):
        """
        The name of this class of Energy Partition Tuples.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((1/2,1/2),2,4,regular=2); V
            (2, 2)-Regular Energy Partition Tuples of energy 4 with level 2 and weights (1/2, 1/2)
        """
        return "{}-Regular Energy Partition Tuples of energy {} with level {}"\
               " and weights {}".format(self._regular, self._energy,
                                        self._level, self._weights)

    def __contains__(self,x):
        """
        Whether this class contains ``x``.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: ([2,1],[2,1]) in EnergyPartitionTuples((1/2,1/2,1),3,4,regular=2)
            False
            sage: ([2,1],[],[2,1]) in EnergyPartitionTuples((1/2,1/2,1),3,4,regular=2)
            False
            sage: ([2,1],[2,1],[]) in EnergyPartitionTuples((1/2,1/2,1),3,4,regular=2)
            True

        TESTS::

            sage: ([],[],[]) in EnergyPartitionTuples((1/2,1/2,1),3,0,regular=2)
            True
        """
        return EnergyPartitionTuples.__contains__(self,x) and \
            sum(p.energy() for p in self.element_class(self,x)) == self._energy

    def __iter__(self):
        """
        Iterate over all Energy Partition Tuples in this class.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((1/2,1/2),2,4,regular=2); V.list()
            [([2, 1], [2, 1]),
             ([], [4, 1]),
             ([], [3, 2]),
             ([4], [1]),
             ([3, 2], [1]),
             ([3], [2]),
             ([2], [3]),
             ([1], [4]),
             ([1], [3, 2]),
             ([4, 1], []),
             ([3, 2], [])]
        """
        if self._energy == 0:
            yield self.element_class(self,[[]]*self._level)
            return

        ieqs = [[0]*(i+1) + [1] + [0]*(self._level-i)\
                for i in range(self._level+1)]
        eqns = [-self._energy]+[[w] for w in self._weights] + [1]
        eqns = [flatten(eqns)]
        P = Polyhedron(ieqs=ieqs, eqns=eqns)
        nlist = []
        for p in P.integral_points():
            for x in IntegerVectors(p[-1],self._level):
                for y in itertools.product(*[EnergyPartitions(
                         w,i+j*w,regular=r,length=j) if r else
                         EnergyPartitions(w,i+j*w,length=j)for w,r,i,j\
                         in zip(self._weights,self._regular,x,p[:-1])]):
                    nlist.append(self.element_class(self,list(y)))
        for z in sorted(nlist):
            yield z

    def an_element(self):
        """
        An element of this class of Energy Partition Tuples.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: V = EnergyPartitionTuples((1/2,1/2),2,4,regular=2);
            sage: V.an_element()
            ([2, 1], [2, 1])
        """
        return self[0]

    def cardinality(self):
        """
        The number of Energy Partition Tuples in this class.

        EXAMPLES::

            sage: from sage.algebras.vertex_algebras.energy_partition_tuples import EnergyPartitionTuples
            sage: EnergyPartitionTuples((2,3/2),2,28,regular=(0,2)).cardinality()
            9079
        """
        #TODO: implement a faster way
        return ZZ.sum(1 for x in self)

