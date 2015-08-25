"""
Wrapper for Singular's Rings

AUTHORS:

- Martin Albrecht (2009-07): initial implementation

- Kwankyu Lee (2010-06): added matrix term order support
"""
#*****************************************************************************
#       Copyright (C) 2009 Martin Albrecht <malb@informatik.uni-bremen.de>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  http://www.gnu.org/licenses/
#*****************************************************************************


from sage.libs.gmp.types cimport __mpz_struct
from sage.libs.gmp.mpz cimport mpz_init_set_ui, mpz_init_set

#from sage.libs.singular.decl cimport number, lnumber, napoly, ring, currRing
from sage.libs.singular.decl cimport number,  napoly, ring, currRing
from sage.libs.singular.decl cimport rChangeCurrRing, rCopy0, rComplete, rDelete, idInit
from sage.libs.singular.decl cimport omAlloc0, omStrDup, omAlloc, omAlloc0Bin,  sip_sring_bin, rnumber_bin
from sage.libs.singular.decl cimport ringorder_dp, ringorder_Dp, ringorder_lp, ringorder_rp, ringorder_ds, ringorder_Ds, ringorder_ls, ringorder_M, ringorder_C, ringorder_wp, ringorder_Wp, ringorder_ws, ringorder_Ws, ringorder_a
from sage.libs.singular.decl cimport p_Copy
from sage.libs.singular.decl cimport n_unknown,  n_Zp,  n_Q,   n_R,   n_GF,  n_long_R,  n_algExt,n_transExt,n_long_C,   n_Z,   n_Zn,  n_Znm,  n_Z2m,  n_CF 
from sage.libs.singular.decl cimport n_coeffType, cfInitCharProc
from sage.libs.singular.decl cimport rDefault, GFInfo, ZnmInfo, nInitChar, AlgExtInfo, nRegister, naInitChar

from sage.rings.integer cimport Integer
from sage.rings.integer_ring cimport IntegerRing_class
from sage.rings.integer_ring import ZZ
from sage.rings.finite_rings.integer_mod_ring import is_IntegerModRing
from sage.rings.number_field.number_field_base cimport NumberField
from sage.rings.rational_field import RationalField
from sage.rings.finite_rings.finite_field_base import FiniteField as FiniteField_generic

from sage.rings.polynomial.term_order import TermOrder
from sage.rings.polynomial.multi_polynomial_libsingular cimport MPolynomial_libsingular, MPolynomialRing_libsingular
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing


# mapping str --> SINGULAR representation
order_dict = {
    "dp": ringorder_dp,
    "Dp": ringorder_Dp,
    "lp": ringorder_lp,
    "rp": ringorder_rp,
    "ds": ringorder_ds,
    "Ds": ringorder_Ds,
    "ls": ringorder_ls,
    "wp": ringorder_wp,
    "Wp": ringorder_Wp,
    "ws": ringorder_ws,
    "Ws": ringorder_Ws,
    "a":  ringorder_a,
}


#############################################################################
cdef ring *singular_ring_new(base_ring, n, names, term_order) except NULL:
    """
    Create a new Singular ring over the ``base_ring`` in ``n``
    variables with the names ``names`` and the term order
    ``term_order``.

    INPUT:

    - ``base_ring`` - a Sage ring

    - ``n`` - the number of variables (> 0)

    - ``names`` - a list of names of length ``n``

    - ``term_order`` - a term ordering

    EXAMPLES::

        sage: P.<x,y,z> = QQ[]
        sage: P
        Multivariate Polynomial Ring in x, y, z over Rational Field

        sage: P.term_order()
        Degree reverse lexicographic term order

        sage: P = PolynomialRing(GF(127),3,names='abc', order='lex')
        sage: P
        Multivariate Polynomial Ring in a, b, c over Finite Field of size 127

        sage: P.term_order()
        Lexicographic term order

        sage: z = QQ['z'].0
        sage: K.<s> = NumberField(z^2 - 2)
        sage: P.<x,y> = PolynomialRing(K, 2)

        sage: P.<x,y,z> = ZZ[]; P
        Multivariate Polynomial Ring in x, y, z over Integer Ring

        sage: P.<x,y,z> = Zmod(2^10)[]; P
        Multivariate Polynomial Ring in x, y, z over Ring of integers modulo 1024

        sage: P.<x,y,z> = Zmod(3^10)[]; P
        Multivariate Polynomial Ring in x, y, z over Ring of integers modulo 59049

        sage: P.<x,y,z> = Zmod(2^100)[]; P
        Multivariate Polynomial Ring in x, y, z over Ring of integers modulo 1267650600228229401496703205376

        sage: P.<x,y,z> = Zmod(2521352)[]; P
        Multivariate Polynomial Ring in x, y, z over Ring of integers modulo 2521352

        sage: P.<x,y,z> = Zmod(25213521351515232)[]; P
        Multivariate Polynomial Ring in x, y, z over Ring of integers modulo 25213521351515232
    """
    cdef long cexponent
    cdef GFInfo* _param
    cdef ZnmInfo _info
    cdef ring* _ring
    cdef char **_names
    cdef char **_ext_names
    cdef char *_name
    cdef int i,j
    cdef int nblcks
    cdef int offset
    cdef int nvars
    cdef int characteristic
    cdef int modbase

    cdef n_coeffType ringtype = n_unknown
    cdef MPolynomialRing_libsingular k
    cdef MPolynomial_libsingular minpoly
    cdef AlgExtInfo extParam
    cdef n_coeffType _type = n_unknown

    #cdef cfInitCharProc myfunctionptr;

    _ring  = NULL

    n = int(n)
    if n<1:
        raise ArithmeticError("The number of variables must be at least 1.")

    nvars = n
    order = TermOrder(term_order, n)

    cdef nbaseblcks = len(order.blocks())
    nblcks = nbaseblcks + order.singular_moreblocks()
    offset = 0


    _names = <char**>omAlloc0(sizeof(char*)*(len(names)))
    for i from 0 <= i < n:
        _name = names[i]
        _names[i] = omStrDup(_name)

    # from the SINGULAR source code documentation for the rInit function
    ##  characteristic --------------------------------------------------
    ##  input: 0 ch=0 : Q     parameter=NULL    ffChar=FALSE   float_len (done)
    ##         0    1 : Q(a,...)        *names         FALSE             (done)
    ##         0   -1 : R               NULL           FALSE  0
    ##         0   -1 : R               NULL           FALSE  prec. >6
    ##         0   -1 : C               *names         FALSE  prec. 0..?
    ##         p    p : Fp              NULL           FALSE             (done)
    ##         p   -p : Fp(a)           *names         FALSE             (done)
    ##         q    q : GF(q=p^n)       *names         TRUE              (todo)

    _wvhdl  = <int **>omAlloc0((nblcks + 2) * sizeof(int *))
    _order  = <int *>omAlloc0((nblcks + 2) * sizeof(int))
    _block0 = <int *>omAlloc0((nblcks + 2) * sizeof(int))
    _block1 = <int *>omAlloc0((nblcks + 2) * sizeof(int))



    cdef int idx = 0
    for i from 0 <= i < nbaseblcks:
        s = order[i].singular_str()
        if s[0] == 'M': # matrix order
            _order[idx] = ringorder_M
            mtx = order[i].matrix().list()
            wv = <int *>omAlloc0(len(mtx)*sizeof(int))
            for j in range(len(mtx)):
                wv[j] = int(mtx[j])
            _wvhdl[idx] = wv
        elif s[0] == 'w' or s[0] == 'W': # weighted degree orders
            _order[idx] = order_dict.get(s[:2], ringorder_dp)
            wts = order[i].weights()
            wv = <int *>omAlloc0(len(wts)*sizeof(int))
            for j in range(len(wts)):
                wv[j] = int(wts[j])
            _wvhdl[idx] = wv
        elif s[0] == '(' and order[i].name() == 'degneglex':  # "(a(1:n),ls(n))"
            _order[idx] = ringorder_a
            if len(order[i]) == 0:    # may be zero for arbitrary-length orders
                nlen = n
            else:
                nlen = len(order[i])

            _wvhdl[idx] = <int *>omAlloc0(len(order[i])*sizeof(int))
            for j in range(nlen):  _wvhdl[idx][j] = 1
            _block0[idx] = offset + 1     # same like subsequent rp block
            _block1[idx] = offset + nlen

            idx += 1;                   # we need one more block here
            _order[idx] = ringorder_rp

        else: # ordinary orders
            _order[idx] = order_dict.get(s, ringorder_dp)

        _block0[idx] = offset + 1
        if len(order[i]) == 0: # may be zero in some cases
            _block1[idx] = offset + n
        else:
            _block1[idx] = offset + len(order[i])
        offset = _block1[idx]
        idx += 1

    # TODO: if we construct a free module don't hardcode! This
    # position determines whether we break ties at monomials first or
    # whether we break at indices first!
    _order[nblcks] = ringorder_C
      
    
    if base_ring.is_field() and base_ring.is_finite() and base_ring.is_prime_field():

        if base_ring.characteristic() <= 2147483647:
            characteristic = base_ring.characteristic()
        else:
            raise TypeError, "Characteristic p must be <= 2147483647."
            
        # example for simpler ring creation interface without monomial orderings:
        #_ring = rDefault(characteristic, nvars, _names)         
        
        _ring = rDefault( characteristic , nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)
        #print "ring with prime coefficient field created"


    elif isinstance(base_ring, RationalField):
        characteristic = 0
        _ring = rDefault( characteristic ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)
        #print "ring with rational coefficient field created"
        

    elif isinstance(base_ring, IntegerRing_class):
        _cf = nInitChar( n_Z, NULL) # integer coefficient ring
        _ring = rDefault (_cf ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)
        #print "polynomial ring over integers created"
         

    elif isinstance(base_ring, FiniteField_generic):
        #print "creating generic finite field"
        # raise "Ring disabled "
        #print "Warning: minpoly in Sage and in Singular may differ(not checked yet) "
        if base_ring.characteristic() <= 2147483647:
            characteristic = -base_ring.characteristic() # note the negative characteristic
        else:
            raise TypeError, "characteristic must be <= 2147483647."
        # TODO: This is lazy, it should only call Singular stuff not MPolynomial stuff
        try:
            k = PolynomialRing(base_ring.prime_subfield(), 1, [base_ring.variable_name()], 'lex')
        except TypeError:
            raise TypeError, "The multivariate polynomial ring in a single variable %s in lex order over %s is supposed to be of type %s"%(base_ring.variable_name(), base_ring,MPolynomialRing_libsingular)
        minpoly = base_ring.polynomial()(k.gen())
        
        #print "k._singular_=",k._singular_()
        #print "GFcharacteristic=",characteristic

        ###################################### first variant: no control of minpoly
        # following example passes:
        # sage: K.<a> = GF(5^3) 
        # sage: R.<x,y,z> = PolynomialRing(K) 
        # sage: K( (4*R(a)^2 + R(a))^3 )

        #_param = <GFInfo *>omAlloc(sizeof(GFInfo))
           
        #_param.GFChar     = characteristic
        #_param.GFDegree   = base_ring.degree()      
        #_param.GFPar_name = omStrDup(base_ring.gen())

        #_cf = nInitChar( n_GF, _param )
        #_ring = rDefault( _cf ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)

        ###################################### second variant: using AlgExt and minpoly: 
        # following example eats all mem:
        # sage: K.<a> = GF(5^3) 
        # sage: R.<x,y,z> = PolynomialRing(K) 
        # sage: K( (4*R(a)^2 + R(a))^3 )

        ch = base_ring.characteristic()
        F = ch.factor()
        assert(len(F)==1)

        modbase = F[0][0]
        cexponent = F[0][1]

        _ext_names = <char**>omAlloc0(sizeof(char*))
        #extname = k.gen()
        #_name = k._names[0]      
        #_ext_names[0] = omStrDup(_name)
        _ext_names[0] = omStrDup(base_ring.gen())
        _cfr = rDefault( modbase, 1, _ext_names ); 

        _cfr.qideal = idInit(1,1)
        ##print "minpoly", minpoly;
        _cfr.qideal.m[0] = minpoly._poly;
        rComplete(_cfr, 1)
        extParam.r =  _cfr;

        _cf = nInitChar( n_algExt,  <void *>&extParam) 

        if (_cf is NULL):
            print "Failed to allocate _cf ring."
            raise "Failed to allocate _cf ring."

        _ring = rDefault (_cf ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)

        

    elif isinstance(base_ring, NumberField) and base_ring.is_absolute():
        #print  " creating NumberField "
        characteristic = 1
        try:
            k = PolynomialRing(RationalField(), 1, [base_ring.variable_name()], 'lex')
        except TypeError:
            print "falied  k = PolynomialRing(RationalField()"
            raise TypeError, "The multivariate polynomial ring in a single variable %s in lex order over Rational Field is supposed to be of type %s"%(base_ring.variable_name(), MPolynomialRing_libsingular)

        minpoly = base_ring.polynomial()(k.gen())
        #print "minpoly", minpoly

        _ext_names = <char**>omAlloc0(sizeof(char*))
        extname = k.gen()
        #print "extname",extname
        #_name = extname
        _name = k._names[0]      
        _ext_names[0] = omStrDup(_name)
        _cfr = rDefault( 0, 1, _ext_names ); 

        _cfr.qideal = idInit(1,1)
        _cfr.qideal.m[0] = minpoly._poly;
        # rComplete(_cfr, 1)
        extParam.r =  _cfr;

        # _type = nRegister(n_algExt, <cfInitCharProc> naInitChar);

        _cf = nInitChar( n_algExt,  <void *>&extParam) #  

        if (_cf is NULL):
            print "Failed to allocate _cf ring."
            raise "Failed to allocate _cf ring."

        _ring = rDefault (_cf ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)


    elif is_IntegerModRing(base_ring):
        #print  " creating IntegerModRing "

        ch = base_ring.characteristic()
        if ch.is_power_of(2):
            #print  " creating IntegerModRing : char is power of 2"
            exponent = ch.nbits() -1
            
            cexponent = exponent          
        
            if sizeof(long) > 4:
            
                # it seems Singular uses ints somewhere
                # internally, cf. #6051 (Sage) and #138 (Singular)
                
                if exponent <= 30:  ringtype = n_Z2m
                else:               ringtype = n_Znm
            else:
                if exponent <= 30: ringtype = n_Z2m
                else:              ringtype = n_Znm
                      
            if ringtype == n_Znm:

              F = ch.factor()

              modbase = F[0][0]
              cexponent = F[0][1]
              
              _info.base = <__mpz_struct*>omAlloc(sizeof(__mpz_struct))
              mpz_init_set_ui(_info.base, modbase)
              _info.exp = cexponent
              _cf = nInitChar( n_Znm, <void *>&_info )
              #print  " creating IntegerModRing : using n_Znm"

            elif  ringtype == n_Z2m:
                #print  " creating IntegerModRing : using n_Z2m"
                _cf = nInitChar( n_Z2m, <void *>cexponent )
                        

        elif base_ring.characteristic().is_prime_power()  and ch < ZZ(2)**160:
            #print  " creating IntegerModRing : char is prime power, using n_Znm"
            F = ch.factor()
            #print "base_ring.characteristic().is_prime_power()"
            #print "F ",F
            assert(len(F)==1)
           
            modbase = F[0][0]
            cexponent = F[0][1]

            _info.base = <__mpz_struct*>omAlloc(sizeof(__mpz_struct))
            mpz_init_set_ui(_info.base, modbase)
            _info.exp = cexponent
            _cf = nInitChar( n_Znm, <void *>&_info )

        else:
            #print "creating IntegerModRing: normal modulus n_Zn"
            # normal modulus
            try:
                characteristic = ch
            except OverflowError:
                raise NotImplementedError("Characteristic %d too big."%ch)          
           
            _info.base = <__mpz_struct*>omAlloc(sizeof(__mpz_struct))
            mpz_init_set_ui(_info.base, characteristic)
            _info.exp = 1
            _cf = nInitChar( n_Zn, <void *>&_info )
        _ring = rDefault( _cf ,nvars, _names, nblcks, _order, _block0, _block1, _wvhdl)
            
            
    else:
        #print "Base ring is not supported."
        raise NotImplementedError("Base ring is not supported.")

    
    if (_ring is NULL):
        print "Failed to allocate Singular ring."
        raise ValueError("Failed to allocate Singular ring.")
        
    #  assert( _ring.cf.ch == characteristic )
    #  _ring.cf.type = ringtype
    #  assert(_ring.N == nvars)
  

    #print "_ring.ShortOut",_ring.ShortOut
    #print "_ring.N",_ring.N

    
    _ring.ShortOut = 0   
    
    if order.is_local():
        assert(_ring.OrdSgn == -1)
    if order.is_global():
         assert(_ring.OrdSgn == 1)
         

    rChangeCurrRing(_ring)

    wrapped_ring = wrap_ring(_ring)
    if wrapped_ring in ring_refcount_dict:
        raise ValueError('newly created ring already in dictionary??')
    ring_refcount_dict[wrapped_ring] = 1
    
    
    return _ring


#############################################################################
ring_refcount_dict = {}


cdef class ring_wrapper_Py(object):
    r"""
    Python object wrapping the ring pointer.

    This is useful to store ring pointers in Python containers.

    You must not construct instances of this class yourself, use
    :func:`wrap_ring` instead.

    EXAMPLES::

        sage: from sage.libs.singular.ring import ring_wrapper_Py
        sage: ring_wrapper_Py
        <type 'sage.libs.singular.ring.ring_wrapper_Py'>
    """

    cdef ring* _ring

    def __cinit__(self):
        """
        The Cython constructor.

        EXAMPLES::

            sage: from sage.libs.singular.ring import ring_wrapper_Py
            sage: t = ring_wrapper_Py(); t
            The ring pointer 0x0
            sage: TestSuite(t).run()
        """
        self._ring = NULL

    def __hash__(self):
        """
        Return a hash value so that instances can be used as dictionary keys.

        OUTPUT:

        Integer.

        EXAMPLES::

            sage: from sage.libs.singular.ring import ring_wrapper_Py
            sage: t = ring_wrapper_Py()
            sage: t.__hash__()
            0
        """
        return <long>(self._ring)

    def __repr__(self):
        """
        Return a string representation.

        OUTPUT:

        String.

        EXAMPLES::

            sage: from sage.libs.singular.ring import ring_wrapper_Py
            sage: t = ring_wrapper_Py()
            sage: t
            The ring pointer 0x0
            sage: t.__repr__()
            'The ring pointer 0x0'
        """
        return 'The ring pointer '+hex(self.__hash__())

    def __cmp__(ring_wrapper_Py left, ring_wrapper_Py right):
        """
        Compare ``left`` and ``right`` so that instances can be used as dictionary keys.

        INPUT:

        - ``right`` -- a :class:`ring_wrapper_Py`

        OUTPUT:

        -1, 0, or +1 depending on whether ``left`` and ``right`` are
         less than, equal, or greather than.

        EXAMPLES::

            sage: from sage.libs.singular.ring import ring_wrapper_Py
            sage: t = ring_wrapper_Py()
            sage: t.__cmp__(t)
            0
        """
        if left._ring < right._ring:
            return -1
        if left._ring > right._ring:
            return +1
        return 0


cdef wrap_ring(ring* R):
    """
    Wrap a C ring pointer into a Python object.

    INPUT:

    - ``R`` -- a singular ring (a C datastructure).

    OUTPUT:

    A Python object :class:`ring_wrapper_Py` wrapping the C pointer.
    """
    cdef ring_wrapper_Py W = ring_wrapper_Py()
    W._ring = R
    return W


cdef ring *singular_ring_reference(ring *existing_ring) except NULL:
    """
    Refcount the ring ``existing_ring``.

    INPUT:

    - ``existing_ring`` -- an existing Singular ring.

    OUTPUT:

    The same ring with its refcount increased. After calling this
    function `n` times, you need to call :func:`singular_ring_delete`
    `n+1` times to actually deallocate the ring.

    EXAMPLE::

        sage: import gc
        sage: _ = gc.collect()
        sage: from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomialRing_libsingular
        sage: from sage.libs.singular.groebner_strategy import GroebnerStrategy
        sage: from sage.libs.singular.ring import ring_refcount_dict
        sage: n = len(ring_refcount_dict)
        sage: prev_rings = set(ring_refcount_dict.keys())
        sage: P = MPolynomialRing_libsingular(GF(541), 2, ('x', 'y'), TermOrder('degrevlex', 2))
        sage: ring_ptr = set(ring_refcount_dict.keys()).difference(prev_rings).pop()
        sage: ring_ptr  # random output
        The ring pointer 0x7f78a646b8d0
        sage: ring_refcount_dict[ring_ptr]
        4

        sage: strat = GroebnerStrategy(Ideal([P.gen(0) + P.gen(1)]))
        sage: ring_refcount_dict[ring_ptr]
        6

        sage: del strat
        sage: _ = gc.collect()
        sage: ring_refcount_dict[ring_ptr]
        4

        sage: del P
        sage: _ = gc.collect()
        sage: ring_ptr in ring_refcount_dict
        True
    """
    if existing_ring==NULL:
        raise ValueError('singular_ring_reference(ring*) called with NULL pointer.')
    cdef object r = wrap_ring(existing_ring)
    refcount = ring_refcount_dict.pop(r)
    ring_refcount_dict[r] = refcount+1
    return existing_ring


#############################################################################
cdef void singular_ring_delete(ring *doomed):
    """
    Carefully deallocate the ring, without changing "currRing" (since
    this method can be called at unpredictable times due to garbage
    collection).

    TESTS:

    This example caused a segmentation fault with a previous version
    of this method::

        sage: import gc
        sage: from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomialRing_libsingular
        sage: R1 = MPolynomialRing_libsingular(GF(5), 2, ('x', 'y'), TermOrder('degrevlex', 2))
        sage: R2 = MPolynomialRing_libsingular(GF(11), 2, ('x', 'y'), TermOrder('degrevlex', 2))
        sage: R3 = MPolynomialRing_libsingular(GF(13), 2, ('x', 'y'), TermOrder('degrevlex', 2))
        sage: _ = gc.collect()
        sage: foo = R1.gen(0)
        sage: del foo
        sage: del R1
        sage: _ = gc.collect()
        sage: del R2
        sage: _ = gc.collect()
        sage: del R3
        sage: _ = gc.collect()
    """
    if doomed==NULL:
        print 'singular_ring_delete(ring*) called with NULL pointer.'
        # this function is typically called in __deallocate__, so we can't raise an exception
        import traceback
        traceback.print_stack()

    if not ring_refcount_dict:  # arbitrary finalization order when we shut Sage down
        return

    cdef ring_wrapper_Py r = wrap_ring(doomed)
    refcount = ring_refcount_dict.pop(r)
    if refcount > 1:
        ring_refcount_dict[r] = refcount-1
        return

    global currRing
    cdef ring *oldRing = currRing
    if currRing == doomed:
        rDelete(doomed)
        currRing = <ring*>NULL
    else:
        rChangeCurrRing(doomed)
        rDelete(doomed)
        rChangeCurrRing(oldRing)




#############################################################################
# helpers for debugging

cpdef poison_currRing(frame, event, arg):
    """
    Poison the ``currRing`` pointer.

    This function sets the ``currRing`` to an illegal value. By
    setting it as the python debug hook, you can poison the currRing
    before every evaluated Python command (but not within Cython
    code).

    INPUT:

    - ``frame``, ``event``, ``arg`` -- the standard arguments for the
      CPython debugger hook. They are not used.

    OUTPUT:

    Returns itself, which ensures that :func:`poison_currRing` will
    stay in the debugger hook.

    EXAMPLES::

        sage: previous_trace_func = sys.gettrace()   # None if no debugger running
        sage: from sage.libs.singular.ring import poison_currRing
        sage: sys.settrace(poison_currRing)
        sage: sys.gettrace()
        <built-in function poison_currRing>
        sage: sys.settrace(previous_trace_func)  # switch it off again
    """
    #print "poisoning currRing"
    global currRing
    currRing = <ring*>NULL
    return poison_currRing


cpdef print_currRing():
    """
    Print the ``currRing`` pointer.

    EXAMPLES::

        sage: from sage.libs.singular.ring import print_currRing
        sage: print_currRing()   # random output
        DEBUG: currRing == 0x7fc6fa6ec480

        sage: from sage.libs.singular.ring import poison_currRing
        sage: _ = poison_currRing(None, None, None)
        sage: print_currRing()
        DEBUG: currRing == 0x0
    """
    cdef size_t addr = <size_t>currRing
    print "DEBUG: currRing == "+str(hex(addr))

def currRing_wrapper():
    """
    Returns a wrapper for the current ring, for use in debugging ring_refcount_dict.

    EXAMPLES::

        sage: from sage.libs.singular.ring import currRing_wrapper
        sage: currRing_wrapper()
        The ring pointer ...
    """
    return wrap_ring(currRing)
