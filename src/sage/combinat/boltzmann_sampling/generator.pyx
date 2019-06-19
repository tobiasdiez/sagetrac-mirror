# coding: utf-8
r"""Boltzmann generator for Context-free grammars.

This module provides functions for generating combinatorial objects (i.e.
objects described by a combinatorial specification, see
:ref:`sage.combinat.bolzmann_sampling.grammar`) according to the Boltzmann
distribution.

Given an unlabelled combinatorial class A, the Boltzmann distribution of
parameter x is such that an object of size n is drawn with the probability ``x^n
/ A(x)`` where ``A(x)`` denotes the ordinary generating function of A. For
labelled classes, this probability is set to ``x^n / (n! * A(x))`` where A(x)
denotes the exponential generating function of A. See [DuFlLoSc04] for details.

By default, the objects produced by the generator are nested tuples of strings
(the atoms). For instance ``('z', ('z', 'e', 'e'), ('z', 'e', 'e'))`` is a
balanced binary tree with 3 internal nodes (z) and 4 leaves (e). To alter this
behaviour and generate other types of objects, you can specify a builder
function for each type of symbol in the grammar. The behaviour of the builders
is that they are applied in bottom up order to the structure "on the fly" during
generation. For instance, in order to generate Dyck words using the grammar for
binary trees, one can use a builder that return ``""`` for each leaf ``"(" +
left child + ")" + right child`` for each node. The builders will receive a
tuple for each product, a string for each atom and builders for unions should be
computed using the ``UnionBuilder`` helper. See the example below for the case
of Dyck words.

EXAMPLES::

    sage: from sage.combinat.boltzmann_sampling.grammar import *
    sage: from sage.combinat.boltzmann_sampling.generator import *
    sage: leaf = Atom("e", size=0)
    sage: z = Atom("z")
    sage: grammar = Grammar(rules={"B": Union(leaf, Product(z, "B", "B"))})
    sage: generator = Generator(grammar)
    sage: def leaf_builder(_):
    ....:     return ""
    sage: def node_builder(tuple):
    ....:     _, left, right = tuple
    ....:     return "(" + left + ")" + right
    sage: generator.set_builder("B", UnionBuilder(leaf_builder, node_builder))
    sage: dyck_word, _ = generator.gen("B", (10, 20))
    sage: dyck_word  # random
    "(()((()())))((())(()((()))))"

Note that the builders' mechanism can also be used to compute parameters on the
structure on the fly without building the whole structure such as the height of
the tree.

EXAMPLES::

    sage: from sage.combinat.boltzmann_sampling.grammar import *
    sage: from sage.combinat.boltzmann_sampling.generator import *
    sage: leaf = Atom("e", size=0)
    sage: z = Atom("z")
    sage: grammar = Grammar(rules={"B": Union(leaf, Product(z, "B", "B"))})
    sage: generator = Generator(grammar)
    sage: def leaf_height(_):
    ....:     return 0
    sage: def node_height(tuple):
    ....:     _, left, right = tuple
    ....:     return 1 + max(left, right)
    sage: generator.set_builder("B", UnionBuilder(leaf_height, node_height))
    sage: height, _ = generator.gen("B", (10, 20))
    sage: height  # random
    6

REFERENCES::

.. [DuFlLoSc04] Philippe Duchon, Philippe Flajolet, Guy Louchard, and Gilles
   Schaeffer. 2004. Boltzmann Samplers for the Random Generation of
   Combinatorial Structures. Comb. Probab. Comput. 13, 4-5 (July 2004), 577-625.
   DOI=http://dx.doi.org/10.1017/S0963548304006315
"""

from sage.libs.gmp.random cimport gmp_randinit_set, gmp_randinit_default
from sage.libs.gmp.types cimport gmp_randstate_t
from sage.misc.randstate cimport randstate, current_randstate
from .grammar import Atom, Product, Ref, Union
from .oracle import SimpleOracle

ctypedef enum options:
    REF,
    ATOM,
    UNION,
    PRODUCT,
    TUPLE,
    FUNCTION,
    WRAP_CHOICE

# ---
# Preprocessing
# ---

# For performance reasons, we use integers to identify symbols rather that
# strings during generation. These two functions do the substitutions and
# compute tables for mapping the names to their integer id and the ids to their
# original names.
# All of this is hidden from the user.

cdef _map_all_names_to_ids_expr(name_to_id, expr):
    """Recursively transform an expression into a triple of the form
    (RULE_TYPE, weight, args) where:
    - RULE_TYPE is of the values of the options enum (see above)
    - weight is the value of the generating function of the expression
    - args is auxilliary information (the name of an atom, the terms of an
      union, ...)
    """
    if isinstance(expr, Ref):
        return (REF, expr.weight, name_to_id[expr.name])
    elif isinstance(expr, Atom):
        return (ATOM, expr.weight, (expr.name, expr.size))
    elif isinstance(expr, Union):
        args = tuple((_map_all_names_to_ids_expr(name_to_id, arg) for arg in expr.args))
        return (UNION, expr.weight, args)
    elif isinstance(expr, Product):
        args = tuple((_map_all_names_to_ids_expr(name_to_id, arg) for arg in expr.args))
        return (PRODUCT, expr.weight, args)

cdef _map_all_names_to_ids(rules):
    """Assign an integer (identifier) to each symbol in the grammar and compute
    two dictionnaries:
    - one that maps the names to their identifiers
    - one that maps the identifier to the original names
    """
    name_to_id = {}
    id_to_name = {}
    for i, name in enumerate(rules.keys()):
        name_to_id[name] = i
        id_to_name[i] = name
    rules = [
        _map_all_names_to_ids_expr(name_to_id, rules[id_to_name[i]])
        for i in range(len(name_to_id))
    ]
    return name_to_id, id_to_name, rules

# ---
# Simulation phase
# ---

cdef int c_simulate(int id, float weight, int size_max, flat_rules, randstate rstate):
    cdef int size = 0
    cdef list todo = [(REF, weight, id)]
    cdef float r = 0.

    while todo:
        type, weight, args = todo.pop()
        if type == REF:
            symbol = args
            todo.append(flat_rules[symbol])
        elif type == ATOM:
            __, atom_size = args
            size += atom_size
            if size > size_max:
                return size
        elif type == UNION:
            r = rstate.c_rand_double() * weight
            for arg in args:
                __, arg_weight, __ = arg
                r -= arg_weight
                if r <= 0:
                    todo.append(arg)
                    break
        elif type == PRODUCT:
            todo += args[::-1]

    return size

# ---
# Actual generation
# ---

cdef inline wrap_choice(float weight, int id):
    return (WRAP_CHOICE, weight, id)

cdef c_generate(int id, float weight, rules, builders, randstate rstate):
    generated = []
    cdef list todo = [(REF, weight, id)]
    cdef float r = 0.
    while todo:
        type, weight, args = todo.pop()
        if type == REF:
            symbol = args
            todo.append((FUNCTION, weight, symbol))
            todo.append(rules[symbol])
        elif type == ATOM:
            atom_name, __ = args
            generated.append(atom_name)
        elif type == UNION:
            r = rstate.c_rand_double() * weight
            for i in range(len(args)):
                arg = args[i]
                __, arg_weight, __ = arg
                r -= arg_weight
                if r <= 0:
                    todo.append(wrap_choice(arg_weight, i))
                    todo.append(arg)
                    break
        elif type == PRODUCT:
            nargs = len(args)
            todo.append((TUPLE, weight, nargs))
            todo += args[::-1]
        elif type == TUPLE:
            nargs = args
            t = tuple(generated[-nargs:])
            generated = generated[:-nargs]
            generated.append(t)
        elif type == FUNCTION:
            func = builders[args]
            x = generated.pop()
            generated.append(func(x))
        elif type == WRAP_CHOICE:
            choice = generated.pop()
            choice_number = args
            generated.append((choice_number, choice))

    obj, = generated
    return obj


cdef c_gen(int id, float weight, rules, int size_min, int size_max, int max_retry, builders):
    cdef int nb_rejections = 0
    cdef int cumulative_rejected_size = 0
    cdef int size = -1
    # A handle on the random generator
    cdef randstate rstate = current_randstate()
    # Allocate a gmp_randstate_t
    cdef gmp_randstate_t gmp_state
    gmp_randinit_default(gmp_state)

    while nb_rejections < max_retry:
        # save the random generator's state
        gmp_randinit_set(gmp_state, rstate.gmp_state)
        size = c_simulate(id, weight, size_max, rules, rstate)
        if size <= size_max and size >= size_min:
            break
        else:
            cumulative_rejected_size += size
            nb_rejections += 1

    if not(size <= size_max and size >= size_min):
        return None

    # Reset the random generator to the state it was just before the simulation
    gmp_randinit_set(rstate.gmp_state, gmp_state)
    obj = c_generate(id, weight, rules, builders, rstate)
    statistics = {
        "size": size,
        "nb_rejections": nb_rejections,
        "cumulative_rejected_size": cumulative_rejected_size,
    }
    return statistics, obj

# ---
# Builders
# ---

cdef inline identity(x):
    return x

def UnionBuilder(*builders):
    """"Helper for computing the builder of a union out of the builders of
    its components. For instance, in order to compute the height of a binary
    tree on the fly:

    EXAMPLES::

        sage: # Grammar: B = Union(leaf, Product(z, B, B))
        sage: from sage.combinat.boltzmann_sampling.generator import UnionBuilder
        sage: def leaf_builder(_):
        ....:     return 0
        sage: def node_builder(tuple):
        ....:     _, left, right = tuple
        ....:     return 1 + max(left, right)
        sage: builder = UnionBuilder(leaf_builder, node_builder)
        sage: choice_number = 0
        sage: builder((choice_number, "leaf"))
        0
        sage: choice_number = 1
        sage: builder((choice_number, ('z', 37, 23)))
        38
    """
    def build(obj):
        index, content = obj
        builder = builders[index]
        return builder(content)
    return build

cdef inline ProductBuilder(builders):
    """Default builder for product: return a tuple."""
    def build(terms):
        return tuple(builders[i](terms[i]) for i in range(len(terms)))
    return build

cdef make_default_builder(rule):
    """Generate the default builders for a rule"""
    type, __, args = rule
    if type == REF:
        return identity
    elif type == ATOM:
        return identity
    elif type == UNION:
        subbuilders = [make_default_builder(component) for component in args]
        return UnionBuilder(*subbuilders)
    elif type == PRODUCT:
        subbuilders = [make_default_builder(component) for component in args]
        return ProductBuilder(subbuilders)

class Generator:
    """High level interface for Boltzmann samplers."""

    def __init__(self, grammar, oracle=None):
        """Make a Generator out of a grammar.

        INPUT::

        - ``grammar`` -- a combinatorial grammar
        - ``oracle`` (default: None) -- an oracle for the grammar. If not
          supplied, a default generic oracle is automatically generated.
        """
        # Load the default oracle if none is supplied
        if oracle is None:
            oracle = SimpleOracle(grammar, e1=1e-6, e2=1e-6)
        self.oracle = oracle
        # flatten the grammar for faster access to rules
        self.grammar = grammar
        self.grammar.annotate(oracle)
        name_to_id, id_to_name, flat_rules = _map_all_names_to_ids(grammar.rules)
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self.flat_rules = flat_rules
        # init builders
        self.builders = [
            make_default_builder(rule)
            for rule in self.flat_rules
        ]

    def set_builder(self, non_terminal, func):
        """Set the builder for a non-terminal symbol.

        INPUT::

        - ``non_terminal`` -- string, the name of the non-terminal symbol
        - ``func`` -- function, the builder
        """
        symbol_id = self.name_to_id[non_terminal]
        self.builders[symbol_id] = func

    def get_builder(self, non_terminal):
        """Retrieve the current builder for a non-terminal symbol.

        INPUT::

        - ``non_terminal`` -- string, the name of the non-terminal symbol
        """
        symbol_id = self.name_to_id[non_terminal]
        return self.builders[symbol_id]

    def gen(self, name, window, max_retry=2000):
        """Generate a term of the grammar in a given size window.

        INPUT::

        - ``name`` -- string, the name of the symbol of the grammar you want to
          generate
        - ``window`` -- pair of integers, the size of the generated object will
          be greater than the first component of the window and lower than the
          second component
        - ``max_retry`` (default: 2000) -- integer, maximum number of attempts.
          If no object in the size window is found in less that ``max_retry``
          attempts, the generator returns None
        """
        id = self.name_to_id[name]
        weight = self.grammar.rules[name].weight
        size_min, size_max = window
        statistics, obj = c_gen(
            id,
            weight,
            self.flat_rules,
            size_min,
            size_max,
            max_retry,
            self.builders,
        )
        return obj, statistics
