# -*- coding: utf-8 -*-
r"""
Two-graphs

A two-graph on `n` points is a family `T \subset \binom {[n]}{3}`
of `3`-sets, such that any `4`-set `S\subset [n]` of size four
contains an even number of elements of `T`. Any graph `([n],E)`
gives rise to a two-graph
`T(E)=\{t \in \binom {[n]}{3} : \left| \binom {t}{2} \cap E \right|\ odd \}`,
and any two graphs with the same two-graph can be obtained one
from the other by :meth:`Seidel switching <Graph.seidel_switching>`.
This defines an equivalence relation on the graphs on `[n]`,
called Seidel switching equivalence.
Conversely, given a two-graph `T`, one can construct a graph
`\Gamma` in the corresponding Seidel switching class with an
isolated vertex `w`. The graph `\Gamma \setminus w` is called
the :meth:`descendant <TwoGraph.descendant>` of `T` w.r.t. `v`.

`T` is called regular if each two-subset of `[n]` is contained
in the same number alpha of triples of `T`.

This module implements a direct construction of a two-graph from a list of
triples, construction of descendant graphs, regularity checking, and other
things such as constructing the complement two-graph, cf. [BH12]_.

AUTHORS:

- Dima Pasechnik (Aug 2015)

Index
-----

This module's methods are the following:

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :meth:`~TwoGraph.is_regular_twograph` | tests if ``self`` is a regular two-graph, i.e. a 2-design
    :meth:`~TwoGraph.complement` | returns the complement of ``self``
    :meth:`~TwoGraph.descendant` | returns the descendant graph at `w`

This module's functions are the following:

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :func:`~taylor_twograph` | constructs Taylor's two-graph for `U_3(q)`
    :func:`~is_twograph`         | checks that the incidence system is a two-graph
    :func:`~twograph_descendant`  | returns the descendant graph w.r.t. a given vertex of the two-graph of a given graph

Methods
---------
"""
from __future__ import absolute_import

from sage.combinat.designs.incidence_structures import IncidenceStructure
from itertools import combinations

class TwoGraph(IncidenceStructure):
    r"""
    Two-graphs class.

    A two-graph on `n` points is a 3-uniform hypergraph, i.e.  a family `T
    \subset \binom {[n]}{3}` of `3`-sets, such that any `4`-set `S\subset [n]`
    of size four contains an even number of elements of `T`. For more
    information, see the documentation of the
    :mod:`~sage.combinat.designs.twographs` module.

    """
    def __init__(self, points=None, blocks=None, incidence_matrix=None,
            name=None, check=False, copy=True):
        r"""
        Constructor of the class

        TESTS::

            sage: from sage.combinat.designs.twographs import TwoGraph
            sage: TwoGraph([[1,2]])
            Incidence structure with 2 points and 1 blocks
            sage: TwoGraph([[1,2]], check=True)
            Traceback (most recent call last):
            ...
            AssertionError: the structure is not a 2-graph!
            sage: p=graphs.PetersenGraph().twograph()
            sage: TwoGraph(p, check=True)
            Incidence structure with 10 points and 60 blocks
        """
        IncidenceStructure.__init__(self, points=points, blocks=blocks,
                                    incidence_matrix=incidence_matrix,
                                    name=name, check=False, copy=copy)
        if check:  # it is a very slow, O(|points|^4), test...
           assert is_twograph(self), "the structure is not a 2-graph!"

    def is_regular_twograph(self, alpha=False):
        r"""
        Test if the :class:`TwoGraph` is regular, i.e. is a 2-design.

        Namely, each pair of elements of :meth:`ground_set` is contained in
        exactly ``alpha`` triples.

        INPUT:

        - ``alpha`` -- (optional, default is ``False``) return the value of
          ``alpha``, if possible.

        EXAMPLES::

            sage: p=graphs.PetersenGraph().twograph()
            sage: p.is_regular_twograph(alpha=True)
            4
            sage: p.is_regular_twograph()
            True
            sage: p=graphs.PathGraph(5).twograph()
            sage: p.is_regular_twograph(alpha=True)
            False
            sage: p.is_regular_twograph()
            False
        """
        r, (_,_,_,a) = self.is_t_design(t=2, k=3, return_parameters=True)
        if r and alpha:
            return a
        return r

    def descendant(self, v):
        """
        The descendant :class:`graph <sage.graphs.graph.Graph>` at ``v``

        The :mod:`switching class of graphs <sage.combinat.designs.twographs>`
        corresponding to ``self`` contains a graph ``D`` with ``v`` its own connected
        component; removing ``v`` from ``D``, one obtains the descendant graph of
        ``self`` at ``v``, which is constructed by this method.

        INPUT:

        - ``v`` -- an element of :meth:`ground_set`

        EXAMPLES::

            sage: p = graphs.PetersenGraph().twograph().descendant(0)
            sage: p.is_strongly_regular(parameters=True)
            (9, 4, 1, 2)
        """
        from sage.graphs.graph import Graph
        return Graph([[z for z in x if z != v]
                      for x in self.blocks() if v in x])

    def complement(self):
        """
        The two-graph which is the complement of ``self``

        That is, the two-graph consisting exactly of triples not in ``self``.
        Note that this is different from :meth:`complement
        <sage.combinat.designs.incidence_structures.IncidenceStructure.complement>`
        of the :class:`parent class
        <sage.combinat.designs.incidence_structures.IncidenceStructure>`.

        EXAMPLES::

            sage: p = graphs.CompleteGraph(8).line_graph().twograph()
            sage: pc = p.complement(); pc
            Incidence structure with 28 points and 1260 blocks

        TESTS::

            sage: from sage.combinat.designs.twographs import is_twograph
            sage: is_twograph(pc)
            True
        """
        return super(TwoGraph, self).complement(uniform=True)

def taylor_twograph(q):
    r"""
    constructing Taylor's two-graph for `U_3(q)`, `q` odd prime power

    The Taylor's two-graph `T` has the `q^3+1` points of the projective plane over `F_{q^2}`
    singular w.r.t. the non-degenerate Hermitean form `S` preserved by `U_3(q)` as its ground set;
    the triples are `\{x,y,z\}` satisfying the condition that `S(x,y)S(y,z)S(z,x)` is square
    (respectively non-square) if `q \cong 1 \mod 4` (respectively if `q \cong 3 \mod 4`).
    See §7E of [BL1984]_.

    There is also a `2-(q^3+1,q+1,1)`-design on these `q^3+1` points, known as the unital of
    order `q`, also invariant under `U_3(q)`.

    INPUT:

    - ``q`` -- a power of an odd prime

    EXAMPLES::

        sage: from sage.combinat.designs.twographs import taylor_twograph
        sage: T=taylor_twograph(3); T
        Incidence structure with 28 points and 1260 blocks
    """
    from sage.graphs.generators.classical_geometries import TaylorTwographSRG
    return TaylorTwographSRG(q).twograph()

def is_twograph(T):
    r"""
    Checks that the incidence system `T` is a two-graph

    INPUT:

    - ``T`` -- an :class:`incidence structure <sage.combinat.designs.IncidenceStructure>`

    EXAMPLES:

    a two-graph from a graph::

        sage: from sage.combinat.designs.twographs import (is_twograph, TwoGraph)
        sage: p=graphs.PetersenGraph().twograph()
        sage: is_twograph(p)
        True

    a non-regular 2-uniform hypergraph which is a two-graph::

        sage: is_twograph(TwoGraph([[1,2,3],[1,2,4]]))
        True

    TESTS:

    wrong size of blocks::

        sage: is_twograph(designs.projective_plane(3))
        False

    a triple system which is not a two-graph::

        sage: is_twograph(designs.projective_plane(2))
        False
    """
    if not T.is_uniform(3):
        return False

    # A structure for a fast triple existence check
    v_to_blocks = {v:set() for v in range(T.num_points())}
    for B in T._blocks:
        B = frozenset(B)
        for x in B:
            v_to_blocks[x].add(B)

    def has_triple(x_y_z):
        x, y, z = x_y_z
        return bool(v_to_blocks[x] & v_to_blocks[y] & v_to_blocks[z])

    # Check that every quadruple contains an even number of triples
    for quad in combinations(range(T.num_points()),4):
        if sum(map(has_triple,combinations(quad,3))) % 2 == 1:
            return False

    return True

def twograph_descendant(G, v, name=None):
    r"""
    Returns the descendant graph w.r.t. vertex `v` of the two-graph of `G`

    In the :mod:`switching class <sage.combinat.designs.twographs>` of `G`,
    construct a graph `\Delta` with `v` an isolated vertex, and return the subgraph
    `\Delta \setminus v`. It is equivalent to, although much faster than, computing the
    :meth:`TwoGraph.descendant` of :meth:`two-graph of G <sage.graphs.graph.Graph.twograph>`, as the
    intermediate two-graph is not constructed.

    INPUT:

    - ``G`` -- a :class:`graph <sage.graphs.graph.Graph>`

    - ``v`` -- a vertex of ``G``

    - ``name`` -- (optional) ``None`` - no name, otherwise derive from the construction

    EXAMPLES:

    one of s.r.g.'s from the :mod:`database <sage.graphs.strongly_regular_db>`::

        sage: from sage.combinat.designs.twographs import twograph_descendant
        sage: A=graphs.strongly_regular_graph(280,135,70)                    # optional - gap_packages internet
        sage: twograph_descendant(A, 0).is_strongly_regular(parameters=True) # optional - gap_packages internet
        (279, 150, 85, 75)

    TESTS::

        sage: T8 = graphs.CompleteGraph(8).line_graph()
        sage: v = T8.vertices()[0]
        sage: twograph_descendant(T8, v)==T8.twograph().descendant(v)
        True
        sage: twograph_descendant(T8, v).is_strongly_regular(parameters=True)
        (27, 16, 10, 8)
        sage: p = graphs.PetersenGraph()
        sage: twograph_descendant(p,5)
        Graph on 9 vertices
        sage: twograph_descendant(p,5,name=True)
        descendant of Petersen graph at 5: Graph on 9 vertices
    """
    G = G.seidel_switching(G.neighbors(v),inplace=False)
    G.delete_vertex(v)
    if name:
        G.name('descendant of '+G.name()+' at '+str(v))
    else:
        G.name('')
    return G

def two_graph(n, l=None, regular=True, existence=False, check=True):
    r"""
    Return a two-graph on `n` points.

    .. NOTE:
        If `regular` is ``True``, then ``l`` must be specified
        and represents the `\lambda` value of the two-graph seen as a 2-design.

    INPUT:

    - ``n`` -- number of points of the two-grap

    - ``l`` -- (optional) `\lambda` value for the two-graph if ``regular`` is ``True``

    - ``regular`` -- if ``True`` Sage tries to build a regular two-graph; default: True

    - ``existence`` -- instead of building the design, return:

        * ``True`` -- meaning that Sage knows how to build the design

        * ``Unknown`` -- meaning that Sage does not know how to build the
          design, but that the design may exist (see :mod:`sage.misc.unknown`).

        * ``False`` -- meaning that the design does not exist.

    - ``check`` -- if ``True`` it checks that the two-graph built is a two-graph. If ``regular`` is also ``True``,
      then it also check that the two-graph is regular.

    EXAMPLES:

        sage: D = designs.two_graph(28, l=10, regular=True)
        sage: D.is_t_design(t=2, v=28, k=3, l=10)
        True

    TESTS::

        sage: D = designs.two_graph(126, 52)  # long
        sage: D.is_t_design(t=2, v=126, k=3, l=52)
        True
        sage: F = designs.two_graph(126, 72)
        sage: F.is_t_design(t=2, v=126, k=3, l=72)
        True
        sage: D.complement() == F
        True

        sage: D = designs.two_graph(344, 150)
        sage: D.is_regular_twograph()
        True
    """
    from sage.arith.misc import is_prime_power

    def do_check(D):
        if not check: return
        if not is_twograph(D):
            raise RuntimeError("Sage built an incorrect two-graph with {} points".format(n))
        if regular and not D.is_regular_twograph():
            raise RuntimeError("Sage built an incorrect regular two-graph with lambda = {}".format(l))

    if regular and l is None:
        raise ValueError("If two-graph is regular, then specify an l value")

    if l is not None and not regular:
        raise ValueError("If you want the two-graph to satisfy the l parameter, then pass regular=True")

    if is_prime_power(n-1) and n%2 == 0:
        # possible taylor U_3(q) or its complement
        (p,k) = is_prime_power(n-1,get_data=True)
        if k % 3 == 0:
            q = p**(k//3)
            if regular and l == (q-1)*(q*q+1)//2:  # taylor U_3(q)
                if existence: return True
                D = taylor_twograph(q)
                do_check(D)
                return D
            if regular and l == n - 2 - (q-1)*(q*q+1)//2:  # complement of taylor U_3(q)
                if existence: return True
                D = taylor_twograph(q)
                D = D.complement()
                do_check(D)
                return D

    if existence:
        return Unknown
    if regular:
        raise RuntimeError("Sage doesn't know how to build a regular two-graph with parameters ({},{})".format(n,l))
    raise RuntimeError("Sage doesn't know how to build a two-graph with {} points".format(n))
