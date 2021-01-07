"""
Mutability Cython Implementation
"""
##########################################################################
#
#   Sage: Open Source Mathematical Software
#
#       Copyright (C) 2006 William Stein <wstein@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
##########################################################################

from sage.misc.decorators import sage_wraps


cdef class Mutability:

    def __init__(self, is_immutable=False):
        self._is_immutable = is_immutable

    def _require_mutable(self):
        if self._is_immutable:
            raise ValueError("object is immutable; please change a copy instead.")

    cdef _require_mutable_cdef(self):
        if self._is_immutable:
            raise ValueError("object is immutable; please change a copy instead.")

    def set_immutable(self):
        """
        Make this object immutable, so it can never again be changed.

        EXAMPLES::

            sage: v = Sequence([1,2,3,4/5])
            sage: v[0] = 5
            sage: v
            [5, 2, 3, 4/5]
            sage: v.set_immutable()
            sage: v[3] = 7
            Traceback (most recent call last):
            ...
            ValueError: object is immutable; please change a copy instead.
        """
        self._is_immutable = 1

    cpdef bint is_immutable(self):
        """
        Return ``True`` if this object is immutable (cannot be changed)
        and ``False`` if it is not.

        To make this object immutable use self.set_immutable().

        EXAMPLES::

            sage: v = Sequence([1,2,3,4/5])
            sage: v[0] = 5
            sage: v
            [5, 2, 3, 4/5]
            sage: v.is_immutable()
            False
            sage: v.set_immutable()
            sage: v.is_immutable()
            True
        """
        self._is_immutable

    cpdef bint is_mutable(self):
        return not self._is_immutable

##########################################################################
## Method decorators for mutating methods resp. methods that assume immutability

def require_mutable(f):
    """
    A decorator that requires mutability for a method to be called.

    EXAMPLES::

        sage: from sage.structure.mutability import require_mutable, require_immutable
        sage: class A(object):
        ....:     def __init__(self, val):
        ....:         self._m = val
        ....:     @require_mutable
        ....:     def change(self, new_val):
        ....:         'change self'
        ....:         self._m = new_val
        ....:     @require_immutable
        ....:     def __hash__(self):
        ....:         'implement hash'
        ....:         return hash(self._m)
        sage: a = A(5)
        sage: a.change(6)
        sage: hash(a)
        Traceback (most recent call last):
        ...
        ValueError: object is mutable; please make it immutable first.
        sage: a._is_immutable = True
        sage: hash(a)
        6
        sage: a.change(7)   # indirect doctest
        Traceback (most recent call last):
        ...
        ValueError: object is immutable; please use a mutable copy instead.
        sage: from sage.misc.sageinspect import sage_getdoc
        sage: print(sage_getdoc(a.change))
        change self

    AUTHORS:

    - Simon King <simon.king@uni-jena.de>: initial version
    - Michael Jung <m.jung@vu.nl>: allow ``_is_mutable`` attribute and new
      error message

    """
    @sage_wraps(f)
    def new_f(self, *args, **kwds):
        if getattr(self, '_is_immutable', False) or not getattr(self, '_is_mutable', True):
            raise ValueError("object is immutable; please use a mutable copy instead.")
        return f(self, *args, **kwds)
    return new_f


def require_immutable(f):
    """
    A decorator that requires immutability for a method to be called.

    EXAMPLES::

        sage: from sage.structure.mutability import require_mutable, require_immutable
        sage: class A(object):
        ....:  def __init__(self, val):
        ....:      self._m = val
        ....:  @require_mutable
        ....:  def change(self, new_val):
        ....:      'change self'
        ....:      self._m = new_val
        ....:  @require_immutable
        ....:  def __hash__(self):
        ....:      'implement hash'
        ....:      return hash(self._m)
        sage: a = A(5)
        sage: a.change(6)
        sage: hash(a)   # indirect doctest
        Traceback (most recent call last):
        ...
        ValueError: object is mutable; please make it immutable first.
        sage: a._is_immutable = True
        sage: hash(a)
        6
        sage: a.change(7)
        Traceback (most recent call last):
        ...
        ValueError: object is immutable; please use a mutable copy instead.
        sage: from sage.misc.sageinspect import sage_getdoc
        sage: print(sage_getdoc(a.__hash__))
        implement hash

    AUTHORS:

    - Simon King <simon.king@uni-jena.de>: initial version
    - Michael Jung <m.jung@vu.nl>: allow ``_is_mutable`` attribute and new
      error message

    """
    @sage_wraps(f)
    def new_f(self, *args, **kwds):
        if not getattr(self, '_is_immutable', False) and getattr(self, '_is_mutable', True):
            raise ValueError("object is mutable; please make it immutable first.")
        return f(self, *args, **kwds)
    return new_f
