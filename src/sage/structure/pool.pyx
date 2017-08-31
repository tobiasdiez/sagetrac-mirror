from libc.stdint cimport uintptr_t

"""
This module implements pools for Sage elements.

The purpose of a pool is to speed up creation and deletion of 
elements. This works as follows. When an element is collected
by the garbage collected, it is not deleted from memory but
pushed to the pool (except if the pool is full). On the other
hand, when a new element (with the same parent) is requested
for creation, the system checks if there are some available
elements in the pool. If there is one, it is returned and,
of course, removed from the pool. Otherwise, a new element
is created.

When elements are quite simple (e.g. small integers, p-adic 
numbers), the bottleneck is often the creation/deletion of
instances of the corresponding classes. Therefore, having a
pool can improve drastically the performances.

Parents that wants to benefit of pool features should derive
from the class :class:`ParentWithPool`.

EXAMPLES::

    sage: R = Zp(3)
    sage: x = R.random_element()
    sage: y = R.random_element()

    sage: R.pool_disable()
    sage: timeit("z = x + y")   # somewhat random
    625 loops, best of 3: 390 ns per loop

    sage: R.pool_enable()
    sage: timeit("z = x + y")   # somewhat random
    625 loops, best of 3: 193 ns per loop

"""

import _weakref

from cpython.object cimport PyObject, PyTypeObject, destructor
from cysignals.memory cimport sig_malloc, sig_realloc, sig_free

from sage.structure.parent cimport Parent
from sage.structure.element cimport Element

cdef extern from "Python.h":
    int unlikely(int) nogil  # Defined by Cython


cdef long DEFAULT_POOL_LENGTH = 100

cdef dict pools_disabled = {}
cdef dict pools_enabled = {}

cdef dict tp_deallocs = {}
cdef class PythonDestructor:
    cdef destructor d
    cdef set(self, destructor d):
        self.d = d


cdef class Pool:
    """
    A class for handling pools in Sage.
    """
    def init_disabled(self, t):
        """
        Initialize this pool.

        By default the pool is empty and disabled.

        This method must not be called directly.
        Instead use the functions :func:`pool_enabled` and
        :func:`pool_disabled` which serve as factory. They
        in particular prevent the creation of many disabled
        pools for the same type.
        """
        cdef PythonDestructor Py_destructor
        self.type = <PyTypeObject*>t
        if tp_deallocs.has_key(t):
            self.save_tp_dealloc = (<PythonDestructor>tp_deallocs[t]).d
        else:
            self.save_tp_dealloc = self.type.tp_dealloc
            Py_destructor = PythonDestructor()
            Py_destructor.set(self.type.tp_dealloc)
            tp_deallocs[t] = Py_destructor
        self.type.tp_dealloc = &tp_dealloc

    def __repr__(self):
        """
        Return a string representation of this pool

            sage: R = Zp(3)
            sage: R.pool_enable()
            sage: R.pool()   # indirect doctest
            Pool of ... elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement
        """
        if self.disabled is None:
            s = "Disabled pool of type "
        else:
            s = "Pool of %s elements of type " % self.allocated
        s += str(<type>self.type)[7:-2]
        return s

    cdef Pool new_enabled(self):
        """
        Return a new enabled pool with the same type

        This method should not be called directly but through
        the function :func:`pool_enabled`.
        """
        cdef Pool pool = Pool()
        pool.type = self.type
        pool.save_tp_dealloc = self.save_tp_dealloc
        if self.disabled is None:
            pool.disabled = self
        else:
            pool.disabled = self.disabled
        return pool

    def automatic_resize(self, enable=True):
        """
        Enable or disable automatic resizing for this pool

        When enabled, the length of the pool doubles when the
        pool is full and a new element needs to be added to
        the pool. In this case, deallocations never occur.

        INPUT:

        - `enable` -- boolean (default: True)
          whether we should enable or disable automatic
          resizing for this pool.

        NOTE::

        A disabled pool cannot be resized

        EXAMPLES::

            sage: R = Zp(3)
            sage: R.pool_enable()
            sage: pool = R.pool()
            sage: pool.length()
            100

            sage: pool.automatic_resize()
            sage: M = identity_matrix(R, 50)
            sage: M.determinant()
            1 + O(3^20)
            sage: pool.length()
            3200
            sage: pool.usage()
            2452

            sage: pool.automatic_resize(False)
            sage: pool.clear(); pool.resize(100)
            sage: M = identity_matrix(R, 50)
            sage: M.determinant()
            1 + O(3^20)
            sage: pool.length()
            100
            sage: pool.usage()
            99


        TESTS::

            sage: R = Zp(5)
            sage: R.pool_disable()
            sage: R.pool()
            Disabled pool of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

            sage: R.pool().automatic_resize()
            Traceback (most recent call last):
            ...
            ValueError: this pool is disabled
        """
        if enable:
            if self.disabled is None:
                raise ValueError("this pool is disabled")
            else:
                self.type.tp_dealloc = &tp_dealloc_with_resize
        else:
            self.type.tp_dealloc = &tp_dealloc

    def length(self):
        """
        Return the length of this pool (that is the total
        number of elements that can be put in this pool)

        EXAMPLES::

            sage: R = Zp(3)
            sage: R.pool_enable()
            sage: R.pool().length()
            100
        """
        from sage.rings.integer_ring import ZZ
        return ZZ(self.size)

    def usage(self):
        """
        Return the number of elements stored in this pool

        EXAMPLES::

            sage: R = Zp(3)
            sage: R.pool_enable()

            sage: R.pool().clear()
            sage: R.pool().usage()
            0

            sage: x = R.random_element()
            sage: del x   # here, x is added to the pool
            sage: R.pool().usage()
            1
        """
        from sage.rings.integer_ring import ZZ
        return ZZ(self.allocated)

    def is_local(self):
        """
        Return whether this pool is local (i.e. only used for
        one parent) or global (i.e. shared with all parents
        having the same element class).

        EXAMPLES::

            sage: R = Zp(5)
            sage: R.pool_enable()

        By default, the pool is global::

            sage: R.pool().is_local()
            False

        We can create a local pool as follows::

            sage: R.pool_enable(local=True)
            sage: R.pool()   # we get here a newly created empty pool
            Pool of 0 elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement
            sage: R.pool().is_local()
            True

        and we reactive a global pool as follows::

            sage: R.pool_enable()
            sage: R.pool().is_local()
            False

        Note that the latter global pool might be different from the 
        former if the former has been garbage collected in the meanwhile.
        """
        cdef type t = <type>self.type
        cdef dict d
        if self.disabled is None:
            d = pools_disabled
        else:
            d = pools_enabled
        if not d.has_key(t):
            return True
        return not(_weakref.ref(self) == d[t])

    def resize(self, length=None):
        """
        Resize this pool

        INPUT:

        - `length` -- an integer or `None` (default: `None`)
          The new length of the pool. If none, the length is doubled.

        EXAMPLES::

            sage: R = Zp(3)
            sage: R.pool_enable()

            sage: pool = R.pool()
            sage: pool.length()
            100

            sage: pool.resize(500)
            sage: pool.length()
            500

        Without argument, the method :meth:`resize` doubles the length::

            sage: pool.resize()
            sage: pool.length()
            1000

        Of course, it is possible to lower the length as well.
        In this case, if there are elements above the new length, they
        are deallocated::

            sage: pool.resize(100)
            sage: pool.length()
            100

        TESTS::

            sage: R = Zp(5)
            sage: R.pool_disable()
            sage: R.pool()
            Disabled pool of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

            sage: R.pool().resize(1000)
            Traceback (most recent call last):
            ...
            ValueError: this pool is disabled
        """
        cdef long size
        if self.disabled is None:
            raise ValueError("this pool is disabled")
        if length is None:
            size = 2 * self.size 
        else:
            size = <long?>length
        self.clear(size)
        self.elements = <PyObject**> sig_realloc(self.elements, size*sizeof(PyObject*))
        # should we test self.elements == NULL?
        self.size = size

    def clear(self, start=0):
        """
        Deallocated elements of the pool

        INPUT:

        - `start` -- an integer (default: 0)
          the position from which elements are deallocated

        EXAMPLES::

            sage: R = Zp(2)
            sage: R.pool_enable()

            sage: pool = R.pool();
            sage: pool.clear()
            sage: pool
            Pool of 0 elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

            sage: L = [ R.random_element() for _ in range(30) ]
            sage: del L
            sage: pool
            Pool of 30 elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

            sage: pool.clear(10)
            sage: pool
            Pool of 10 elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

            sage: pool.clear()
            sage: pool
            Pool of 0 elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement
        """
        cdef long s = <long?>start
        cdef long i
        if s < self.allocated:
            for i from s <= i < self.allocated:
                self.save_tp_dealloc(self.elements[i])
            self.allocated = s

    def __dealloc__(self):
        """
        Deallocate this pool
        """
        self.clear()
        if self.elements != NULL:
            sig_free(self.elements)
        if self.disabled is None:
            # We reintroduce the standard deallocating function
            # if there are still alive objects of this type
            self.type.tp_dealloc = self.save_tp_dealloc


cdef inline PY_NEW_FROM_POOL(Pool pool):
    """
    Return a new object using the Pool `pool`

    INPUT:

    - `pool` -- a pool

    NOTE:

    If `pool` is not empty, an element is pulled for it.
    Otherwise, a new element with the type of the pool is
    created and returned.
    """
    cdef PyObject* o
    cdef PyTypeObject* t
    if pool.allocated > 0:
        #print("reuse")
        pool.allocated -= 1
        o = pool.elements[pool.allocated]
        o.ob_refcnt = 0
        return <object>o
    else:
        t = pool.type
        return t.tp_new(<type>t, <object>NULL, <object>NULL)


cdef void tp_dealloc_fallback(PyObject* o):
    cdef type t = type(<object>o)
    cdef destructor dealloc = (<PythonDestructor>tp_deallocs[t]).d
    dealloc(o)


cdef void tp_dealloc(PyObject* o):
    """
    Add an element to a pool

    INPUT:

    - `o` -- an object

    NOTES:

    The pool is infered from the element using `o.parent().pool()` 
    (or more precisely, a Cython fast equivalent of this).

    If the pool is full, the element is deallocated.

    This function must not be called manually (even in Cython code).
    It is called automatically by Python when the object `o` is collected.
    """
    cdef Parent parent = (<Element>o)._parent
    if unlikely(parent is None):
        tp_dealloc_fallback(o)
        return
    cdef Pool pool = parent._pool
    if unlikely(pool is None):
        tp_dealloc_fallback(o)
        return
    if pool.allocated < pool.size:
        #print("add to pool")
        o.ob_refcnt = 1
        (<Element>o)._parent = None
        pool.elements[pool.allocated] = o
        pool.allocated += 1
    else:
        #print("dealloc")
        pool.save_tp_dealloc(o)


cdef void tp_dealloc_with_resize(PyObject* o):
    """
    Add an element to a pool

    INPUT:

    - `o` -- an object

    NOTES:

    The pool is infered from the element using `o.parent().pool()` 
    (or more precisely, a Cython fast equivalent of this).

    If the pool is full, it is resized.

    This function must not be called manually. 
    It is called automatically by Python when the object `o` is collected.
    """
    cdef Parent parent = (<Element>o)._parent
    if unlikely(parent is None):
        tp_dealloc_fallback(o)
        return
    cdef Pool pool = parent._pool
    if unlikely(pool is None):
        tp_dealloc_fallback(o)
        return
    if unlikely(pool.allocated >= pool.size):
        pool.resize()
    #print("add to pool")
    o.ob_refcnt = 1
    pool.elements[pool.allocated] = o
    pool.allocated += 1


cdef pool_disabled(type t):
    """
    Return a disabled pool of requested type

    The result is cached. Therefore, two different calls
    to this function with the same argument return the same
    pool

    INPUT:

    - `t` -- a Python type

    EXAMPLES::

        sage: R = Zp(3)
        sage: R.pool_disable()
        sage: R.pool()   # indirect doctest
        Disabled pool of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

        sage: S = Zp(5)
        sage: S.pool_disable()
        sage: S.pool()   # indirect doctest
        Disabled pool of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

        sage: R.pool() is S.pool()
        True
    """
    cdef Pool pool = None
    if pools_disabled.has_key(t):
        wr_pool = pools_disabled[t]
        pool = wr_pool()
    if pool is None:
        pool = Pool()
        pool.init_disabled(t)
    pools_disabled[t] = _weakref.ref(pool)
    return pool


cdef pool_enabled(Pool pool_dis, length, bint local):
    """
    Return an enabled pool

    INPUT:

    - `pool_dis` -- a pool
      the disabled pool from which the returned pool is
      constructed (in particular, the associated Python
      type is infered from here)

    - `length` -- an integer or `None`
      the length of the pool

    - `local` -- a boolean
      whether this pool should be local or global 
      (a global enabled pool is unique for a given type)

    NOTE:

    If `length` is specified and `local` is false, the
    global corresponding pool, if it already exists, is resized.

    If `length` is not specified and a new pool needs to be
    created, it is sized to the default length stored in 
    the global variable `DEFAULT_POOL_LENGTH`.

    EXAMPLES::

        sage: R = Zp(3)
        sage: R.pool_enable()

        sage: R.pool()   # indirect doctest
        Pool of ... elements of type sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement

    In this particular example, the pool is global. Indeed::

        sage: S = Zp(5)
        sage: S.pool_enable()
        sage: R.pool() is S.pool()
        True
    """
    cdef type t = <type>pool_dis.type
    cdef Pool pool = None
    if local:
        pool = pool_dis.new_enabled()
        if length is None:
            length = DEFAULT_POOL_LENGTH
    else:
        if pools_enabled.has_key(t):
            wr_pool = pools_enabled[t]
            pool = wr_pool()
        if pool is None:
            pool = pool_dis.new_enabled()
            if length is None:
                length = DEFAULT_POOL_LENGTH
        pools_enabled[t] = _weakref.ref(pool)
    if length is not None:
        pool.resize(length)
    return pool
