r"""
Lazy lists

A lazy list is an iterator that behaves like a list and possesses a cache
mechanism. A lazy list is potentially infinite and speed performances of the
cache is comparable with Python lists. One major difference with original
Python list is that lazy list are immutable. The advantage is that slices
share memory.

EXAMPLES::

    sage: from sage.misc.lazy_list import lazy_list
    sage: P = lazy_list(Primes())
    sage: P[100]
    547
    sage: P[10:34]
    lazy list [31, 37, 41, ...]
    sage: P[12:23].list()
    [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]

    sage: f = lazy_list((i**2-3*i for i in xrange(10)))
    sage: for i in f: print i,
    0 -2 -2 0 4 10 18 28 40 54
    sage: i1 = iter(f)
    sage: i2 = iter(f)
    sage: print i1.next(), i1.next()
    0 -2
    sage: print i2.next(), i2.next()
    0 -2
    sage: print i1.next(), i2.next()
    -2 -2

It is possible to prepend a list to a lazy list::

    sage: from itertools import count
    sage: l = [3,7] + lazy_list(i**2 for i in count())
    sage: l
    lazy list [3, 7, 0, ...]

But, naturally, not the other way around::

    sage: lazy_list(i-1 for i in count()) + [3,2,5]
    Traceback (most recent call last):
    ...
    TypeError: can only add list to lazy_list
"""

# in types.pxd
#    bint PyType_Check(object o)
#    bint PyType_CheckExact(object o)
# include "sage/ext/python_iterator.pxi"

include "sage/ext/stdsage.pxi"

cdef extern from "Python.h":
    Py_ssize_t PY_SSIZE_T_MAX
    void Py_INCREF(object)

from cpython.list cimport *

from libc cimport limits

cdef class lazy_list(object):

    def __init__(self, start=None, stop=None, step=None):
        if start is None:
            start = 0
        if stop is None:
            stop = PY_SSIZE_T_MAX
        if step is None:
            step = 1

        if not isinstance(start, (int,long)):
            try:
                start = start.__index__()
            except Exception:
                raise TypeError("start must be None or integer")
        if start < 0:
            raise ValueError("start must be non negative")
        if not isinstance(stop, (int,long)):
            try:
                stop = stop.__index__()
            except Exception:
                raise TypeError("stop must be None or integer")
        if stop < 0:
            raise ValueError("stop must be larger than -2")
        if not isinstance(step, (int,long)):
            try:
                step = step.__index__()
            except Exception:
                raise TypeError("step must be None or integer")
        if step <= 0:
            raise ValueError("step must be positive")

        self.start = <Py_ssize_t> start
        self.stop  = <Py_ssize_t> stop
        self.step  = <Py_ssize_t> step

    def __getitem__(self, key):
        r"""
        Returns a lazy list which shares the same attributes.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: f = lazy_list_from_iterator(iter([1,2,3]))
            sage: f0 = f[0:]
            sage: print f.get(0), f.get(1), f.get(2)
            1 2 3
            sage: f1 = f[1:]
            sage: print f1.get(0), f1.get(1)
            2 3
            sage: f2 = f[2:]
            sage: print f2.get(0)
            3
            sage: f3 = f[3:]
            sage: print f3.get(0)
            Traceback (most recent call last):
            ...
            IndexError: lazy list index out of range

            sage: l = lazy_list_from_iterator([0]*12)[1::2]
            sage: l[2::3]
            lazy list [0, 0]
            sage: l[3::2]
            lazy list [0, 0]

        A lazy list automatically adjusts the indices in order that start and
        stop are congruent modulo step::

            sage: P = lazy_list_from_iterator(iter(Primes()))
            sage: P[1:12:4].start_stop_step()
            (1, 13, 4)
            sage: P[1:13:4].start_stop_step()
            (1, 13, 4)
            sage: P[1:14:4].start_stop_step()
            (1, 17, 4)

        We check commutation::

            sage: l = lazy_list_from_iterator(iter(xrange(10000)))
            sage: l1 = l[::2][:3001]
            sage: l2 = l[:6002][::2]
            sage: l1.start_stop_step() == l2.start_stop_step()
            True
            sage: l3 = l1[13::2][:50:2]
            sage: l4 = l1[:200][13:113:4]
            sage: l3.start_stop_step() == l4.start_stop_step()
            True

        Further tests::

            sage: l = lazy_list_from_iterator(iter([0]*25))
            sage: l[2::3][2::3][4::5]
            lazy list []
            sage: l[2::5][3::][1::]
            lazy list [0]
            sage: l[3:24:2][1::][1:7:3]
            lazy list [0, 0]
            sage: l[::2][2::][2::3]
            lazy list [0, 0, 0]
        """

        
        if not PY_TYPE_CHECK(key, slice):
            key = key.__index__()
        if PY_TYPE_CHECK(key, int) or PY_TYPE_CHECK(key, long):
            return self.get(key)

        # the following make all terms > 0
        if key.start is None:
            start = 0
        elif PY_TYPE_CHECK(key.start, int) or PY_TYPE_CHECK(key.start, long):
            start = key.start
        else:
            try:
                start = key.start.__index__()
            except Exception:
                raise TypeError("slice indices must be integers or None or have an __index__ method")

        if key.stop is None:
            stop = PY_SSIZE_T_MAX
        elif PY_TYPE_CHECK(key.stop, int) or PY_TYPE_CHECK(key.stop, long):
            stop = key.stop
        else:
            try:
                stop = key.stop.__index__()
            except Exception:
                raise TypeError("slice indices must be integers or None or have an __index__ method")

        if key.step is None:
            step = 1
        elif PY_TYPE_CHECK(key.step, int) or PY_TYPE_CHECK(key.step, long):
            step = key.step
        else:
            try:
                step = key.step.__index__()
            except Exception:
                raise TypeError("slice indices must be integers or None or have an __index__ method")

        if step == 0:
            raise TypeError("step may not be 0")
        if step < 0 or start < 0 or stop < 0:
            raise ValueError("slice indices must be non negative")

        step = step * self.step
        start = self.start + start * self.step
        if stop != PY_SSIZE_T_MAX:
            stop = self.start + stop * self.step
        if stop > self.stop:
            stop = self.stop
        if stop != PY_SSIZE_T_MAX and stop%step != start%step:
            stop = stop - (stop-start)%step + step

        if start >= stop:
            l = []
            return lazy_list_from_iterator(iter(l), l, 0, 0, 1)

        clss, args = self.__reduce__()
        args = args[:-3] + (start, stop, step) #because all subclasses need some arguments + (start, stop, step)

        return clss(*args)

    def __iter__(self):
        cdef Py_ssize_t i

        for i in range(self.start, self.stop, self.step):
            yield <object> self.get(i)
    
cdef class lazy_list_with_cache(lazy_list):
    r"""
    Abstract class for Lazy list with cache.
    Should not be instantiated

    INPUT:
    - ``cache`` -- ``None`` (default) or a list - used to initialize the cache.

    - ``start``, ``stop``, ``step`` -- ``None`` (default) or a non-negative
      integer - parameters for slices

    EXAMPLES::

        sage: from sage.misc.lazy_list import lazy_list_from_iterator
        sage: from itertools import count
        sage: m = lazy_list_from_iterator(count()); m
        lazy list [0, 1, 2, ...]

        sage: m2 = lazy_list_from_iterator(count(), start=8, stop=20551, step=2)
        sage: m2
        lazy list [8, 10, 12, ...]

        sage: x = iter(m)
        sage: print x.next(), x.next(), x.next()
        0 1 2
        sage: y = iter(m)
        sage: print y.next(), y.next(), y.next()
        0 1 2
        sage: print x.next(), y.next()
        3 3
        sage: loads(dumps(m))
        lazy list [0, 1, 2, ...]

    .. NOTE::

        - :class:`lazy_list` interprets the constant ``(size_t)-1`` as infinity
        - all entry indices are stictly less than ``stop`` so that
          :class:`lazy_list` agrees with ``range(start, stop)``
    """
    def __init__(self, cache, start=None, stop=None, step=None):
        r"""
        Initialize ``self``.

        TESTS::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: from itertools import count
            sage: f = lazy_list_from_iterator(count())
            sage: loads(dumps(f))
            lazy list [0, 1, 2, ...]
        """

        super(lazy_list_with_cache, self).__init__(start, stop, step)

        if cache is None or self.stop <= self.start:
            cache = []
        elif not isinstance(cache, list):
            raise TypeError("cache must be a list, not %s"%type(cache).__name__)

        self.cache = cache

    def start_stop_step(self):
        r"""
        Return the triple ``(start, stop, step)`` of reference points of the
        original lazy list.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list
            sage: p = lazy_list_from_iterator(iter(Primes()))[:2147483647]
            sage: p.start_stop_step()
            (0, 2147483647, 1)
            sage: q = p[100:1042233:12]
            sage: q.start_stop_step()
            (100, 1042240, 12)
            sage: r = q[233::3]
            sage: r.start_stop_step()
            (2896, 1042252, 36)
            sage: 1042241%3 == 233%3
            True
        """
        return (self.start, self.stop, self.step)

    def list(self):
        r"""
        Return the list made of the elements of ``self``.

        .. NOTE::

            If the iterator is sufficiently large, this will build a list
            of length ``(size_t)-1`` which should be beyond the capacity of
            your RAM!

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: P = lazy_list_from_iterator(iter(Primes()))
            sage: P[2:143:5].list()
            [5, 19, 41, 61, 83, 107, 137, 163, 191, 223, 241, 271, 307, 337, 367, 397, 431, 457, 487, 521, 563, 593, 617, 647, 677, 719, 751, 787, 823]
            sage: P = lazy_list_from_iterator(iter([1,2,3]))
            sage: P.list()
            [1, 2, 3]
            sage: P[:100000].list()
            [1, 2, 3]
            sage: P[1:7:2].list()
            [2]

        TESTS:

        Check that the cache is immutable::

            sage: lazy = lazy_list_from_iterator(iter(Primes()))[:5]
            sage: l = lazy.list(); l
            [2, 3, 5, 7, 11]
            sage: l[0] = -1; l
            [-1, 3, 5, 7, 11]
            sage: lazy.list()
            [2, 3, 5, 7, 11]
        """
        try:
            self.update_cache_up_to(self.stop-1)
        except StopIteration:
            pass
        return self.cache[self.start:self.stop:self.step]

    def info(self):
        r"""
        Print information about ``self`` on standard output.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list
            sage: P = lazy_list_from_iterator(iter(Primes()))[10:21474838:4]
            sage: P.info()
            cache length 0
            start        10
            stop         21474838
            step         4
            sage: P[0]
            31
            sage: P.info()
            cache length 11
            start        10
            stop         21474838
            step         4
        """
        print "cache length", len(self.cache)
        print "start       ", self.start
        print "stop        ", self.stop
        print "step        ", self.step

    def __repr__(self):
        r"""
        Return a string representation.

        TESTS::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: from itertools import count
            sage: r = lazy_list_from_iterator(count()); r  # indirect doctest
            lazy list [0, 1, 2, ...]
            sage: r[:0]
            lazy list []
            sage: r[:1]
            lazy list [0]
            sage: r[:2]
            lazy list [0, 1]
            sage: r[:3]
            lazy list [0, 1, 2]
            sage: r[:4]
            lazy list [0, 1, 2, ...]
            sage: lazy_list_from_iterator([0,1])
            lazy list [0, 1]
            sage: lazy_list_from_iterator([0,1,2,3])
            lazy list [0, 1, 2, ...]
        """
        cdef Py_ssize_t num_elts = 1 + (self.stop-self.start-1) / self.step
        cdef Py_ssize_t length = PyList_GET_SIZE(self.cache)

        if (length <= self.start + 3*self.step and
            num_elts != length / self.step):
            self._fit(self.start + 3*self.step)
            num_elts = 1 + (self.stop-self.start-1) / self.step

        if num_elts == 0:
            return "lazy list []"

        if num_elts == 1:
            return "lazy list [%s]"%(repr(self.get(0)))

        if num_elts == 2:
            return "lazy list [%s, %s]"%(
                    repr(self.get(0)),
                    repr(self.get(1)))

        if num_elts == 3:
            return "lazy list [%s, %s, %s]"%(
                repr(self.get(0)),
                repr(self.get(1)),
                repr(self.get(2)))

        return "lazy list [%s, %s, %s, ...]"%(
                repr(self.get(0)),
                repr(self.get(1)),
                repr(self.get(2)))

    cdef int update_cache_up_to(self, Py_ssize_t i) except *:
        r"""
        Update the cache up to ``i``.

        If the iterator stops, the function silently return 0 (no error are
        raised). Otherwise it returns 1.

        .. TODO::

            This method should be implemented in such way that it does not raise
            StopIteration (using PyIter_Next from sage/ext/python_iterator.pxi).
            The fact that the iterator stops before than expected may be encoded
            in the returned value of the function:

            * 0 : function succeded
            * 1 : iterator stopped before stop was reached
            * -1 : an error occurred
        """
        raise NotImplementedError("abstract class")
        
        
    def _fit(self, n):
        r"""
        Re-adjust ``self.stop`` if the iterator stops before ``n``.

        After a call to ``self._fit(n)`` and if *after the call* ``n`` is less
        than ``self.stop`` then you may safely call ``self.cache[n]``. In other
        words, ``self._fit(n)` ensure that either the lazy list is completely
        expanded in memory or that you may have access to the ``n``-th item.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: l = lazy_list_from_iterator([0,1,2,-34,3,2,-5,12,1,4,-18,5,-12])[2::3]
            sage: l.info()
            cache length 0
            start        2
            stop         14
            step         3
            sage: l
            lazy list [2, 2, 1, ...]
            sage: l._fit(13)
            sage: l.info()
            cache length 13
            start        2
            stop         14
            step         3
        """
        if n >= self.stop:
            n = self.stop
        try:
            self.update_cache_up_to(n)
        except StopIteration:
            self.stop = PyList_GET_SIZE(self.cache)
            if self.stop <= self.start:
                self.start = self.stop = 0
                self.step = 1
            if (self.start - self.stop) % self.step:
                self.stop += self.step + (self.start - self.stop) % self.step

    def __call__(self, i):
        r"""
        Return the element at position ``i``.

        If the index is not an integer, then raise a ``TypeError``.  If the
        argument is negative then raise a ``ValueError``.  Finally, if the
        argument is beyond the size of that lazy list it raises a
        ``IndexError``.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: from itertools import chain, repeat
            sage: f = lazy_list_from_iterator(chain(iter([1,2,3]), repeat('a')))
            sage: f.get(0)
            1
            sage: f.get(3)
            'a'
            sage: f.get(0)
            1
            sage: f.get(4)
            'a'

            sage: g = f[:10]
            sage: g.get(5)
            'a'
            sage: g.get(10)
            Traceback (most recent call last):
            ...
            IndexError: lazy list index out of range
            sage: g.get(1/2)
            Traceback (most recent call last):
            ...
            TypeError: rational is not an integer
        """
        cdef Py_ssize_t j

        if not hasattr(i, '__index__'):
            raise TypeError("indices must be integers, not %s"%type(i).__name__)

        j = <Py_ssize_t> i.__index__()
        if j < 0:
            raise ValueError("indices must be non negative")

        j = self.start + j * self.step
        if j >= self.stop:
            raise IndexError("lazy list index out of range")
        self._fit(j)
        if j >= self.stop:
            raise IndexError("lazy list index out of range")
        return <object> PyList_GET_ITEM(self.cache, j)

    get = __call__

    def __add__(self, other):
        r"""
        """
        if isinstance(self, list):
            return lazy_list_from_iterator(iter(other), cache=self[:]) # args[-4] = self.cache
        raise TypeError("can only add list to lazy_list")

    
cdef class lazy_list_from_iterator(lazy_list_with_cache):
    r"""
    Lazy list.

    INPUT:

    - ``iterator`` -- an iterable or an iterator

    - ``cache`` -- ``None`` (default) or a list - used to initialize the cache.

    - ``start``, ``stop``, ``step`` -- ``None`` (default) or a non-negative
      integer - parameters for slices

    EXAMPLES::

        sage: from sage.misc.lazy_list import lazy_list_from_iterator
        sage: from itertools import count
        sage: m = lazy_list_from_iterator(count()); m
        lazy list [0, 1, 2, ...]

        sage: m2 = lazy_list_from_iterator(count(), start=8, stop=20551, step=2)
        sage: m2
        lazy list [8, 10, 12, ...]

        sage: x = iter(m)
        sage: print x.next(), x.next(), x.next()
        0 1 2
        sage: y = iter(m)
        sage: print y.next(), y.next(), y.next()
        0 1 2
        sage: print x.next(), y.next()
        3 3
        sage: loads(dumps(m))
        lazy list [0, 1, 2, ...]

    .. NOTE::

        - :class:`lazy_list` interprets the constant ``(size_t)-1`` as infinity
        - all entry indices are stictly less than ``stop`` so that
          :class:`lazy_list` agrees with ``range(start, stop)``
    """
    def __init__(self, iterator, cache=None, start=None, stop=None, step=None):
        r"""
        Initialize ``self``.

        TESTS::

            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: from itertools import count
            sage: f = lazy_list_from_iterator(count())
            sage: loads(dumps(f))
            lazy list [0, 1, 2, ...]
        """
        
        super(lazy_list_from_iterator, self).__init__(cache, start, stop, step)
        
        from sage.misc.misc import is_iterator
        if not is_iterator(iterator):
            if isinstance(iterator, (list, tuple)):
                stop = min(stop,len(iterator))
            iterator = iter(iterator)
        self.iterator = iterator


    cdef int update_cache_up_to(self, Py_ssize_t i) except -1:
        r"""
        Update the cache up to ``i``.

        If the iterator stops, the function silently return 0 (no error are
        raised). Otherwise it returns 1.

        .. TODO::

            This method should be implemented in such way that it does not raise
            StopIteration (using PyIter_Next from sage/ext/python_iterator.pxi).
            The fact that the iterator stops before than expected may be encoded
            in the returned value of the function:

            * 0 : function succeded
            * 1 : iterator stopped before stop was reached
            * -1 : an error occurred
        """
        while PyList_GET_SIZE(self.cache) <= i:
            PyList_Append(self.cache, self.iterator.next())
        return 0

    def __reduce__(self):
        r"""
        Pickling support

        EXAMPLES::

            sage: from itertools import count
            sage: from sage.misc.lazy_list import lazy_list_from_iterator
            sage: m = lazy_list_from_iterator(count())
            sage: x = loads(dumps(m))
            sage: y = iter(x)
            sage: print y.next(), y.next(), y.next()
            0 1 2
        """
        return self.__class__, (self.iterator, self.cache, self.start, self.stop, self.step)

cdef class lazy_list_from_function(lazy_list_with_cache):
    r"""
    Lazy list.

    INPUT:

    - ``function`` -- a callable object

    - ``cache`` -- ``None`` (default) or a list - used to initialize the cache.

    - ``start``, ``stop``, ``step`` -- ``None`` (default) or a non-negative
      integer - parameters for slices

    EXAMPLES::

        sage: from sage.misc.lazy_list import lazy_list_from_function
        sage: from itertools import count
        sage: m = lazy_list_from_function(lambda a : [a[-1]+1], [0]); m
        lazy list [0, 1, 2, ...]

        sage: m2 = lazy_list_from_function(lambda a: [a[-1]+1], [0], start=8, stop=20551, step=2)
        sage: m2
        lazy list [8, 10, 12, ...]

        sage: x = iter(m)
        sage: print x.next(), x.next(), x.next()
        0 1 2
        sage: y = iter(m)
        sage: print y.next(), y.next(), y.next()
        0 1 2
        sage: print x.next(), y.next()
        3 3

    .. NOTE::

        - :class:`lazy_list` interprets the constant ``(size_t)-1`` as infinity
        - all entry indices are stictly less than ``stop`` so that
          :class:`lazy_list` agrees with ``range(start, stop)``
    """
    def __init__(self, function, cache=None, start=None, stop=None, step=None):
        r"""
        Initialize ``self``.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_function
            
            We can provide initial values:

            sage: m = lazy_list_from_function(lambda a: [a[-1]+1], [0]); m
            lazy list [0, 1, 2, ...]

            Beware of strange constructs!  The following makes each
            element of the lazy list the lazy list itself...

            sage: f = lazy_list_from_function(lambda a: a);
            sage: f[0]
            [[...]] # never stop
        """
        
        super(lazy_list_from_function, self).__init__(cache, start, stop, step)

        if not hasattr(function, '__call__'):
            raise TypeError("%s object is not callable"%(type(function)))
        self.function = function

    cdef int update_cache_up_to(self, Py_ssize_t i) except -1:
        r"""
        Update the cache up to ``i``.

        If the iterator stops, the function silently return 0 (no error are
        raised). Otherwise it returns 1.

        .. TODO::

            This method should be implemented in such way that it does not raise
            StopIteration (using PyIter_Next from sage/ext/python_iterator.pxi).
            The fact that the iterator stops before than expected may be encoded
            in the returned value of the function:

            * 0 : function succeded
            * 1 : iterator stopped before stop was reached
            * -1 : an error occurred
        """
        cdef Py_ssize_t j, n, s, j_max

        
        if i > PY_SSIZE_T_MAX:
            raise StopIteration

        s = PyList_GET_SIZE(self.cache)
        
        while s <= i:
            l = <list> self.function(self.cache)
            n = PyList_GET_SIZE(l)

            j_max = min(PY_SSIZE_T_MAX - s, n)
            
            for j in range(j_max):
                PyList_Append(self.cache, <object> PyList_GET_ITEM(l,j))

            s = s + j_max
                
        return 0


    def __reduce__(self):
        r"""
        Pickling support

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_from_function
            sage: m = lazy_list_from_function(lambda l : [l[-1]+1])
            sage: loads(dumps(m))
            ...
            PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed

        """

        return self.__class__, (self.function, self.cache, self.start, self.stop, self.step)
    
cdef class lazy_list_explicit(lazy_list_with_cache):
    r"""
    Lazy list.

    INPUT:

    - ``function`` -- a callable object

    - ``cache`` -- ``None`` (default) or a list - used to initialize the cache.

    - ``start``, ``stop``, ``step`` -- ``None`` (default) or a non-negative
      integer - parameters for slices

    EXAMPLES::

        sage: from sage.misc.lazy_list import lazy_list_explicit
        sage: from itertools import count
        sage: m = lazy_list_explicit(lambda n : n); m
        lazy list [0, 1, 2, ...]

        sage: m2 = lazy_list_explicit(lambda n: n, start=8, stop=20551, step=2)
        sage: m2
        lazy list [8, 10, 12, ...]

        sage: x = iter(m)
        sage: print x.next(), x.next(), x.next()
        0 1 2
        sage: y = iter(m)
        sage: print y.next(), y.next(), y.next()
        0 1 2
        sage: print x.next(), y.next()
        3 3
    .. NOTE::

        - :class:`lazy_list` interprets the constant ``(size_t)-1`` as infinity
        - all entry indices are stictly less than ``stop`` so that
          :class:`lazy_list` agrees with ``range(start, stop)``
    """
    def __init__(self, function, cache=None, start=None, stop=None, step=None):
        r"""
        Initialize ``self``.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_explicit
            sage: m = lazy_list_explicit(lambda n : n*(n+1)/2); m
            lazy list [0, 1, 3, ...]

        """
        
        super(lazy_list_explicit, self).__init__(cache, start, stop, step)

        if not hasattr(function, '__call__'):
            raise TypeError("%s object is not callable"%(type(function)))
        self.function = function

    cdef int update_cache_up_to(self, Py_ssize_t i) except -1:
        r"""
        Update the cache up to ``i``.

        If the iterator stops, the function silently return 0 (no error are
        raised). Otherwise it returns 1.

        .. TODO::

            This method should be implemented in such way that it does not raise
            StopIteration (using PyIter_Next from sage/ext/python_iterator.pxi).
            The fact that the iterator stops before than expected may be encoded
            in the returned value of the function:

            * 0 : function succeded
            * 1 : iterator stopped before stop was reached
            * -1 : an error occurred
        """
        cdef Py_ssize_t j = PyList_GET_SIZE(self.cache)
        
        if i > PY_SSIZE_T_MAX:
            raise StopIteration

        if i > j :
            self.cache.extend([None]*(i-j))
                
        return 0


    def __reduce__(self):
        r"""
        Pickling support

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_explicit
            sage: m = lazy_list_explicit(lambda n : n*(n+1)/2)
            sage: loads(dumps(m))
            ...
            PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed

        """

        return self.__class__, (self.function, self.cache, self.start, self.stop, self.step)

    def __call__(self, i):
        r"""
        Return the element at position ``i``.

        If the index is not an integer, then raise a ``TypeError``.  If the
        argument is negative then raise a ``ValueError``.  Finally, if the
        argument is beyond the size of that lazy list it raises a
        ``IndexError``.

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_explicit
            sage: f = lazy_list_from_explicit(lambda n : n*(n+1)/2)
            sage: f.get(0)
            0
            sage: f.get(3)
            6
            sage: f.get(0)
            0
            sage: f.get(4)
            10

            sage: g = f[:10]
            sage: g.get(5)
            15
            sage: g.get(10)
            Traceback (most recent call last):
            ...
            IndexError: lazy list index out of range
            sage: g.get(1/2)
            Traceback (most recent call last):
            ...
            TypeError: rational is not an integer
        """
        cdef Py_ssize_t j

        if not hasattr(i, '__index__'):
            raise TypeError("indices must be integers, not %s"%type(i).__name__)

        j = <Py_ssize_t> i.__index__()
        if j < 0:
            raise ValueError("indices must be non negative")

        j = self.start + j * self.step
        if j >= self.stop:
            raise IndexError("lazy list index out of range")
        self._fit(j) # may modify self.stop
        if j >= self.stop:
            raise IndexError("lazy list index out of range")

        obj = <object>PyList_GET_ITEM(self.cache, j)

        if obj is None :
            obj = self.function(j)
            Py_INCREF(obj)
            PyList_SET_ITEM(self.cache, j, obj)

        return obj

    get = __call__

    def list(self):
        r"""
        Return the list made of the elements of ``self``.

        .. NOTE::

            If self.stop is sufficiently large, this will build a list
            of length ``(size_t)-1`` which should be beyond the capacity of
            your RAM!

        EXAMPLES::

            sage: from sage.misc.lazy_list import lazy_list_explicit
            sage: P = lazy_list_explicit(lambda n : factorial(n))
            sage: P[2:17:3].list()
            [2, 120, 40320, 39916800, 87178291200]
            sage: P = lazy_list_explicit(lambda n : factorial(n), start = 0, stop = 10)
            sage: P.list()
            [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
            sage: P[:100000].list()
            [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
            sage: P[1:7:2].list()
            [1, 6, 120]

        TESTS:

        Check that the cache is immutable::

            sage: lazy = lazy_list_explicit(lambda n : 2*n+1)[:5]
            sage: l = lazy.list(); l
            [1, 3, 5, 7, 9]
            sage: l[0] = -1; l
            [-1, 3, 5, 7, 9]
            sage: lazy.list()
            [1, 3, 5, 7, 9]
        """
        try:
            self.update_cache_up_to(self.stop-1)
            for i in range((self.stop-self.start)/self.step):
                self.get(i) # force the computation
        except StopIteration:
            pass
        return self.cache[self.start:self.stop:self.step]
    
cdef class lazy_list_periodic(lazy_list):

    def __init__(self, period, pre_period=[], start=None, stop=None, step=None):
        super(lazy_list_periodic, self).__init__(start, stop, step)

        if not isinstance(period, list):
            raise TypeError("period must be a list, not %s"%type(period).__name__)
        else:
            self.period = period
        
        if not isinstance(pre_period, list):
            raise TypeError("pre_period must be a list, not %s"%type(pre_period).__name__)
        else:
            self.pre_period = pre_period

    def __repr__(self):

        cdef Py_ssize_t pp_size, p_size, i
        
        l = [ "lazy list [" ]

        pp_size = PyList_GET_SIZE(self.pre_period)
        p_size = PyList_GET_SIZE(self.period)
        
        for i in range(self.start, pp_size, self.step):
            l.append(repr(<object>PyList_GET_ITEM(self.pre_period, i)))
            l.append(", ")

        l.append("|")
            
        for i in range(pp_size % self.step, p_size, self.step):
            l.append(repr(<object>PyList_GET_ITEM(self.period, i)))
            l.append(", ")

        l.append("...]")
                
        return "".join(l)

    def __reduce__(self):
        
        return self.__class__, (self.period, self.pre_period, self.start, self.stop, self.step)

    def __call__(self, i):

        cdef Py_ssize_t j, pp_size, p_size
        
        if not hasattr(i, '__index__'):
            raise TypeError("indices must be integers, not %s"%type(i).__name__)

        j = i.__index__()
        
        if j < 0:
            raise ValueError("indices must be non negative")

        j = self.start + j * self.step

        if j >= self.stop:
            raise IndexError("lazy list index out of range")

        pp_size = PyList_GET_SIZE(self.pre_period)

        if j < pp_size:
            return <object> PyList_GET_ITEM(self.pre_period, j)
        
        p_size = PyList_GET_SIZE(self.period)

        j = (j - pp_size) % p_size
        
        return <object> PyList_GET_ITEM(self.period, j)

    get = __call__

    def list(self):

        cdef Py_ssize_t i
        cdef list l = <object> PyList_New((self.stop - self.start)/self.step)

        for i in range((self.stop - self.start)/self.step):
            PyList_SET_ITEM(l, i, self.get(i))

        return l
