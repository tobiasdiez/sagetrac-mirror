r"""
Interface with TdLib (algorithms for tree decompositions)

This module defines functions based on TdLib, a
library that implements algorithms for tree
decompositions written by Lukas Larisch. 

**Definition** :

A `tree decomposition` of a graph `G` is a pair `(T, \beta)` consisting of a 
tree T and a function `\beta: V(T) \rightarrow 2^{V(G)}` associating with each node `t \in V(T)` a set
of vertices `\beta (t) \subseteq V(G)` such that 

(T1) for every edge `e \in E(G)` there is a node `t \in V(T)` with `e \subseteq \beta (t)`, and

(T2) for all `v \in V(G)` the set `\beta^{-1} := \{t \in V(T): v \in \beta (t)\}` is non-empty and connected in T.

The width of `(T, \beta)` is defined as `max\{|\beta (t)|-1: t \in V(T) \}`.
The treewidth of G is defined as the minimum width over all tree decompositions of G.

**Some known results** :

    - Trees have treewidth 1
    
    - Cycles have treewidth 2
    
    - Series-parallel graphs have treewidth at most 2
    
    - Cliques must be contained in some bag of a tree decomposition


Computing the treewidth or a tree decomposition of a given graph is NP-hard in general.

**This module containes the following functions** :

.. csv-table::
    :class: contentstable
    :widths: 30, 70
    :delim: |

    :meth:`treedecomposition_exact` | Computes a tree decomposition of exact width

    :meth:`get_width` | Returns the width of a given tree decomposition


AUTHOR: Lukas Larisch (10-25-2015): Initial version

REFERENCE:

.. [1] P. D. Seymour and Robin Thomas. 1993. Graph searching and a min-max theorem for tree-width. J. Comb. Theory Ser. B 58, 1 (May 1993), 22-33.
.. [2] S. Arnborg, A. Proskurowski, Characterization and Recognition of Partial 3-Trees. SIAM Journal of Alg. and Discrete Methods, Vol. 7, pp. 305-314
.. [3] H. L. Bodlaender, A Tourist Guide through Treewidth, Acta Cybern. 1993


Methods
-------
"""

from libcpp.vector cimport vector

from tdlib cimport sage_exact_decomposition

from sage.sets.set import Set
from sage.graphs.graph import Graph

include "sage/ext/interrupt.pxi"
include 'sage/ext/stdsage.pxi'


#!!!!!!   NOTICE   !!!!!!!!
#Sage vertices have to be named by unsigned integers
#Sage bags of decompositions have to be lists of unsigned integers
#!!!!!!!!!!!!!!!!!!!!!!!!!!

##############################################################
############ GRAPH/DECOMPOSITION ENCODING/DECODING ###########
#the following will be used implicitly do the translation
#between Sage graph encoding and TdLib graph encoding,
#which is based on BGL

class TreeDecomposition(Graph):
    #This is just for the repr-message.
 
    def __repr__(self):
        r"""
        Returns a short string representation of self.

        EXAMPLE::

            sage: T = TreeDecomposition()
            sage: T
            Treedecomposition of width -1 on 0 vertices 
        """
        return "Treedecomposition of width " + str(get_width(self)) + " on " + str(self.order()) + " vertices"

cdef cython_make_tdlib_graph(G, vector[unsigned int] &V, vector[unsigned int] &E):
    V_python = G.vertices()
    for i in range(0, len(V_python)):
        V.push_back(V_python[i])

    E_python = G.edges()
    for i in range(0, len(E_python)):
        v,w,l = E_python[i]
        E.push_back(v)
        E.push_back(w)

cdef cython_make_tdlib_decomp(T, vector[vector[int]] &V, vector[unsigned int] &E):
    V_python = T.vertices()
    for i in range(0, len(V_python)):
        V.push_back(V_python[i])

    E_python = T.edges()
    for i in range(0, len(E_python)):
        v,w,l = E_python[i]
        E.push_back(V_python.index(v))
        E.push_back(V_python.index(w))


cdef cython_make_sage_graph(G, vector[unsigned int] &V, vector[unsigned int] &E):
    for i in range(0, len(V)):
        G.add_vertex(V[i])

    for i in range(0, len(E), 2):
        G.add_edge(V[E[i]], V[E[i+1]])


cdef cython_make_sage_decomp(G, vector[vector[int]] &V, vector[unsigned int] &E):
    for i in range(0, len(V)):
        G.add_vertex(Set(V[i]))

    for i in range(0, len(E), 2):
        G.add_edge(Set(V[E[i]]), Set(V[E[i+1]]))


##############################################################
############ EXACT ALGORITHMS ################################

def treedecomposition_exact(G, lb=-1):
    """
    Computes a tree decomposition of exact width, iff the given lower bound 
    is not greater than the treewidth of the input graph. Otherwise
    a tree decomposition of a width than matches the given lower bound
    will be computed.

    INPUTS:

    - ``G`` -- a generic graph

    - ``lb`` -- a lower bound to the treewidth of G, e.g. computed by lower_bound (default: ``'-1'``)

    OUTPUT:

    - A tree decomposition of G of tw(G), if the lower bound was not greater than tw(G), otherwise a tree decomposition of width = lb. 

..  WARNING::

    The computation can take a lot of time for a graph G on more than about 30 vertices and tw(G) > 3

EXAMPLES::

        sage: g = graphs.HouseGraph()
        sage: t = sage.graphs.tdlib.treedecomposition_exact(g)
        tree decomposition of width 2 computed
        sage: t.show(vertex_size=2000)

TEST::

        sage: g = graphs.HouseGraph()
        sage: t = sage.graphs.tdlib.treedecomposition_exact(g)
        tree decomposition of width 2 computed
    """
    cdef vector[unsigned int] V_G, E_G, E_T
    cdef vector[vector[int]] V_T

    cython_make_tdlib_graph(G, V_G, E_G)

    cdef int c_lb = lb 

    sig_on()

    sage_exact_decomposition(V_G, E_G, V_T, E_T, c_lb)

    sig_off()

    T = TreeDecomposition()
    cython_make_sage_decomp(T, V_T, E_T)

    print("tree decomposition of width " + str(get_width(T)) + " computed")

    return T

def get_width(T):
    """
    Returns the width (maximal size of a bag minus one) of a given tree decomposition.

    INPUT:

    - ``T`` -- a tree decomposition

    OUTPUT:

    - The width of ``T``

EXAMPLES::

        sage: g = graphs.RandomGNP(10, 0.05)
        sage: t = sage.graphs.tdlib.seperator_algorithm(g)
        sage: sage.graphs.tdlib.get_width(t)
        9
    """

    max = -1

    for v in T.vertices():
        if(len(v)-1 > max):
            max = len(v)-1

    return max

