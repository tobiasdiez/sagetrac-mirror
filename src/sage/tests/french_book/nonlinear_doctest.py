## -*- encoding: utf-8 -*-
"""
Doctests from French Sage book
Test file for chapter "Équations non linéaires" ("Nonlinear Equations")

Tests extracted from ./nonlinear.tex.

Sage example in ./nonlinear.tex, line 61::

    sage: R.<x> = PolynomialRing(RealFloatingPointField(prec=10))
    sage: p = 2*x^7 - 21*x^6 + 64*x^5 - 67*x^4 + 90*x^3 \
    ....:  + 265*x^2 - 900*x + 375
    sage: p.roots()
    [(-1.7, 1), (0.50, 1), (1.7, 1), (5.0, 2)]
    sage: p.roots(ring=ComplexField(10), multiplicities=False)
    [-1.7, 0.50, 1.7, 5.0, -2.2*I, 2.2*I]
    sage: p.roots(ring=RationalField())
    [(1/2, 1), (5, 2)]

Sage example in ./nonlinear.tex, line 158::

    sage: R.<x> = PolynomialRing(QQ, 'x')
    sage: p = x^4 + x^3 + x^2 + x + 1
    sage: K.<alpha> = p.root_field()
    sage: p.roots(ring=K, multiplicities=None)
    [alpha, alpha^2, alpha^3, -alpha^3 - alpha^2 - alpha - 1]
    sage: alpha^5
    1

Sage example in ./nonlinear.tex, line 202::

    sage: R.<x> = PolynomialRing(RR, 'x')
    sage: d = ZZ.random_element(1, 15)
    sage: p = R.random_element(d)
    sage: p.degree() == sum(r[1] for r in p.roots(CC))
    True

Sage example in ./nonlinear.tex, line 231::

    sage: from itertools import product
    sage: def build_complex_roots(degree):
    ....:     R.<x> = PolynomialRing(CDF, 'x')
    ....:     v = []
    ....:     for c in product([-1, 1], repeat=degree+1):
    ....:         v.extend(R(c).roots(multiplicities=False))
    ....:     return v
    sage: data = build_complex_roots(12) # long time
    sage: g = points(data, pointsize=1, aspect_ratio=1)  # long time

Sage example in ./nonlinear.tex, line 275::

    sage: a, b, c, x = var('a, b, c, x')
    sage: p = a * x^2 + b * x + c
    sage: type(p)
    <type 'sage.symbolic.expression.Expression'>
    sage: p.parent()
    Symbolic Ring
    sage: p.roots(x)
    [(-1/2*(b + sqrt(b^2 - 4*a*c))/a, 1),
    (-1/2*(b - sqrt(b^2 - 4*a*c))/a, 1)]

Sage example in ./nonlinear.tex, line 299::

    sage: a, b, c, d, e, f, x = var('a, b, c, d, e, f, x')
    sage: p = a*x^5+b*x^4+c*x^3+d*x^2+e*x+f
    sage: try:
    ....:    p.roots(x)
    ....: except RuntimeError:
    ....:    print('No explicit roots found')
    No explicit roots found

Sage example in ./nonlinear.tex, line 315::

    sage: x, a, b, c, d = var('x, a, b, c, d')
    sage: P = a * x^3 + b * x^2 + c * x + d
    sage: alpha = var('alpha')
    sage: P.subs(x=x + alpha).expand().coefficient(x, 2)
    3*a*alpha + b
    sage: P.subs(x = x - b / (3 * a)).expand().collect(x)
    a*x^3 - 1/3*(b^2/a - 3*c)*x + 2/27*b^3/a^2 - 1/3*b*c/a + d

Sage example in ./nonlinear.tex, line 328::

    sage: p, q, u, v = var('p, q, u, v')
    sage: P = x^3 + p * x + q
    sage: P.subs(x = u + v).expand()
    u^3 + 3*u^2*v + 3*u*v^2 + v^3 + p*u + p*v + q

Sage example in ./nonlinear.tex, line 340::

    sage: P.subs({x: u + v, q: -u^3 - v^3}).factor()
    (3*u*v + p)*(u + v)
    sage: P.subs({x: u+v, q: -u^3 - v^3, p: -3 * u * v}).expand()
    0
    sage: X = var('X')
    sage: solve([X^2 + q*X - p^3 / 27 == 0], X, solution_dict=True)
    [{X: -1/2*q - 1/18*sqrt(12*p^3 + 81*q^2)},
    {X: -1/2*q + 1/18*sqrt(12*p^3 + 81*q^2)}]

Sage example in ./nonlinear.tex, line 367::

     sage: e = sin(x) * (x^3 + 1) * (x^5 + x^4 + 1)
     sage: roots = e.roots()
     sage: print(len(roots))
     9
     sage: print(roots)
     [(0, 1),
     (-1/2*(1/18*sqrt(23)*sqrt(3) - 1/2)^(1/3)*(I*sqrt(3) + 1)
     - 1/6*(-I*sqrt(3) + 1)/(1/18*sqrt(23)*sqrt(3) - 1/2)^(1/3), 1),
     (-1/2*(1/18*sqrt(23)*sqrt(3) - 1/2)^(1/3)*(-I*sqrt(3) + 1)
     - 1/6*(I*sqrt(3) + 1)/(1/18*sqrt(23)*sqrt(3) - 1/2)^(1/3), 1),
     ((1/18*sqrt(23)*sqrt(3) - 1/2)^(1/3) + 1/3/(1/18*sqrt(23)*sqrt(3)
     - 1/2)^(1/3), 1),
     (-1/2*I*sqrt(3) - 1/2, 1),
     (1/2*I*sqrt(3) - 1/2, 1),
     (1/2*I*sqrt(3)*(-1)^(1/3) - 1/2*(-1)^(1/3), 1),
     (-1/2*I*sqrt(3)*(-1)^(1/3) - 1/2*(-1)^(1/3), 1), ((-1)^(1/3), 1)]

Sage example in ./nonlinear.tex, line 424::

    sage: alpha, m, x = var('alpha, m, x')
    sage: p = function('p')(x)
    sage: q = function('q')(x)
    sage: p = (x - alpha)^m * q
    sage: p.derivative(x)
    (-alpha + x)^(m - 1)*m*q(x) + (-alpha + x)^m*diff(q(x), x)
    sage: simplify(p.derivative(x)(x=alpha))
    0

Sage example in ./nonlinear.tex, line 450::

    sage: R.<x> = PolynomialRing(QQ, 'x')
    sage: p = 128 * x^13 - 1344 * x^12 + 6048 * x^11 \
    ....: - 15632 * x^10 + 28056 * x^9 - 44604 * x^8 \
    ....: + 71198 * x^7 - 98283 * x^6 + 105840 * x^5 \
    ....: - 101304 * x^4 + 99468 * x^3 - 81648 * x^2 \
    ....: + 40824 * x - 8748
    sage: d = gcd(p, p.derivative())
    sage: (p // d).degree()
    4
    sage: roots = SR(p // d).roots(multiplicities=False)
    sage: roots
    [1/2*I*sqrt(3)*2^(1/3) - 1/2*2^(1/3),
    -1/2*I*sqrt(3)*2^(1/3) - 1/2*2^(1/3),
    2^(1/3), 3/2]
    sage: [QQbar(p(alpha)).is_zero() for alpha in roots] # long time
    [True, True, True, True]

Sage example in ./nonlinear.tex, line 504::

    sage: R.<x> = PolynomialRing(RR, 'x')
    sage: p = x^7 - 131/3*x^6 + 1070/3*x^5 - 2927/3*x^4 \
    ....: + 2435/3*x^3 - 806/3*x^2 + 3188/3*x - 680
    sage: sign_changes = \
    ....: [p[i] * p[i + 1] < 0 \
    ....: for i in range(p.degree())].count(True)
    sage: real_positive_roots = \
    ....: sum([alpha[1] \
    ....: if alpha[0] > 0 else 0 for alpha in p.roots()])
    sage: sign_changes, real_positive_roots
    (7, 5)

Sage example in ./nonlinear.tex, line 567::

    sage: def count_sign_changes(l):
    ....:     changes = [l[i]*l[i + 1] < 0 \
    ....:                for i in range(len(l) - 1)]
    ....:     return changes.count(True)
    sage: def sturm(p, a, b):
    ....:     assert p.degree() > 2
    ....:     assert not (p(a) == 0)
    ....:     assert not (p(b) == 0)
    ....:     if a > b:
    ....:         a, b = b, a
    ....:     remains = [p, p.derivative()]
    ....:     for i in range(p.degree()):
    ....:         remains.append(-(remains[i] % remains[i + 1]))
    ....:     evals = [[], []]
    ....:     for q in remains:
    ....:         evals[0].append(q(a))
    ....:         evals[1].append(q(b))
    ....:     return count_sign_changes(evals[0]) \
    ....:            - count_sign_changes(evals[1])

Sage example in ./nonlinear.tex, line 591::

    sage: R.<x> = PolynomialRing(QQ, 'x')
    sage: p = (x - 34) * (x - 5) * (x - 3) * (x - 2) * (x - 2/3)
    sage: sturm(p, 1, 4)
    2
    sage: sturm(p, 1, 10)
    3
    sage: sturm(p, 1, 200)
    4
    sage: p.roots(multiplicities=False)
    [34, 5, 3, 2, 2/3]
    sage: sturm(p, 1/2, 35)
    5

Sage example in ./nonlinear.tex, line 651::

    sage: f(x) = 4 * sin(x) - exp(x) / 2 + 1
    sage: a, b = RR(-pi), RR(pi)
    sage: bool(f(a) * f(b) < 0)
    True

Sage example in ./nonlinear.tex, line 661::

    sage: solve(f(x) == 0, x)
    [sin(x) == 1/8*e^x - 1/4]

Sage example in ./nonlinear.tex, line 666::

    sage: f.roots()
    Traceback (most recent call last):
    ...
    RuntimeError: no explicit roots found

Sage example in ./nonlinear.tex, line 684::

    sage: a, b = RR(-pi), RR(pi)
    sage: g = plot(f, a, b, rgbcolor='blue')

Sage example in ./nonlinear.tex, line 720::

    sage: def phi(s, t): return (s + t) / 2
    sage: def intervalgen(f, phi, s, t):
    ....:     msg = 'Wrong arguments: f({0})*f({1})>=0)'.format(s, t)
    ....:     assert (f(s) * f(t) < 0), msg
    ....:     yield s
    ....:     yield t
    ....:     while True:
    ....:         u = phi(s, t)
    ....:         yield u
    ....:         if f(u) * f(s) < 0:
    ....:             t = u
    ....:         else:
    ....:             s = u

Sage example in ./nonlinear.tex, line 785::

    sage: a, b
    (-3.14159265358979, 3.14159265358979)
    sage: bisection = intervalgen(f, phi, a, b)
    sage: next(bisection)
    -3.14159265358979
    sage: next(bisection)
    3.14159265358979
    sage: next(bisection)
    0.000000000000000

Sage example in ./nonlinear.tex, line 805::

    sage: from types import GeneratorType, FunctionType
    sage: def checklength(u, v, w, prec):
    ....:     return abs(v - u) < 2 * prec
    sage: def iterate(series, check=checklength,prec=10^-5, maxit=100):
    ....:     assert isinstance(series, GeneratorType)
    ....:     assert isinstance(check, FunctionType)
    ....:     niter = 2
    ....:     v, w = next(series), next(series)
    ....:     while (niter <= maxit):
    ....:         niter += 1
    ....:         u, v, w = v, w, next(series)
    ....:         if check(u, v, w, prec):
    ....:             print('After {0} iterations: {1}'.format(niter, w))
    ....:             return
    ....:     print('Failed after {0} iterations'.format(maxit))

Sage example in ./nonlinear.tex, line 837::

    sage: bisection = intervalgen(f, phi, a, b)
    sage: iterate(bisection)
    After 22 iterations: 2.15847275559132

Sage example in ./nonlinear.tex, line 899::

    sage: phi(s, t) = t - f(t) * (s - t) / (f(s) - f(t))
    sage: falsepos = intervalgen(f, phi, a, b)
    sage: iterate(falsepos)
    After 8 iterations: -2.89603757331027

Sage example in ./nonlinear.tex, line 906::

    sage: a, b = RR(-pi), RR(pi)
    sage: g = plot(f, a, b, rgbcolor='blue')
    sage: phi(s, t) = t - f(t) * (s - t) / (f(s) - f(t))
    sage: falsepos = intervalgen(f, phi, a, b)
    sage: u, v, w = next(falsepos), next(falsepos), next(falsepos)
    sage: niter = 3
    sage: while niter < 9:
    ....:     g += line([(u, 0), (u, f(u))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += line([(u, f(u)), (v, f(v))], rgbcolor='red')
    ....:     g += line([(v, 0), (v, f(v))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += point((w, 0), rgbcolor='red')
    ....:     if (f(u) * f(w)) < 0:
    ....:         u, v = u, w
    ....:     else:
    ....:         u, v = w, v
    ....:     w = next(falsepos)
    ....:     niter += 1

Sage example in ./nonlinear.tex, line 942::

    sage: a, b = RR(pi/2), RR(pi)
    sage: phi(s, t) = t - f(t) * (s - t) / (f(s) - f(t))
    sage: falsepos = intervalgen(f, phi, a, b)
    sage: phi(s, t) = (s + t) / 2
    sage: bisection = intervalgen(f, phi, a, b)
    sage: iterate(falsepos)
    After 15 iterations: 2.15846441170219
    sage: iterate(bisection)
    After 20 iterations: 2.15847275559132

Sage example in ./nonlinear.tex, line 954::

    sage: a, b = RR(pi/2), RR(pi)
    sage: g = plot(f, a, b, rgbcolor='blue')
    sage: phi(s, t) = t - f(t) * (s - t) / (f(s) - f(t))
    sage: falsepos = intervalgen(f, phi, a, b)
    sage: u, v, w = next(falsepos), next(falsepos), next(falsepos)
    sage: niter = 3
    sage: while niter < 7:
    ....:     g += line([(u, 0), (u, f(u))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += line([(u, f(u)), (v, f(v))], rgbcolor='red')
    ....:     g += line([(v, 0), (v, f(v))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += point((w, 0), rgbcolor='red')
    ....:     if (f(u) * f(w)) < 0:
    ....:         u, v = u, w
    ....:     else:
    ....:         u, v = w, v
    ....:     w = next(falsepos)
    ....:     niter += 1

Sage example in ./nonlinear.tex, line 1025::

    sage: f.derivative()
    x |--> 4*cos(x) - 1/2*e^x
    sage: a, b = RR(pi/2), RR(pi)

Sage example in ./nonlinear.tex, line 1042::

    sage: def newtongen(f, u):
    ....:     while 1:
    ....:         yield u
    ....:         u -=  f(u) / (f.derivative()(u))
    sage: def checkconv(u, v, w, prec):
    ....:     return abs(w - v) / abs(w) <= prec

Sage example in ./nonlinear.tex, line 1053::

    sage: iterate(newtongen(f, a), check=checkconv)
    After 6 iterations: 2.15846852566756

Sage example in ./nonlinear.tex, line 1058::

    sage: generator = newtongen(f, a)
    sage: g = plot(f, a, b, rgbcolor='blue')
    sage: u, v = next(generator), next(generator)
    sage: niter = 2
    sage: while niter < 6:
    ....:     g += point((u, 0), rgbcolor='red')
    ....:     g += line([(u, 0), (u, f(u))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += line([(u, f(u)), (v, 0)], rgbcolor='red')
    ....:     u, v = v, next(generator)
    ....:     niter += 1

Sage example in ./nonlinear.tex, line 1109::

    sage: def secantgen(f, a):
    ....:     yield a
    ....:     estimate = f.derivative()(a)
    ....:     b = a - f(a) / estimate
    ....:     yield b
    ....:     while True:
    ....:         fa, fb = f(a), f(b)
    ....:         if fa == fb:
    ....:             estimate = f.derivative()(a)
    ....:         else:
    ....:             estimate = (fb - fa) / (b - a)
    ....:         a = b
    ....:         b -= fb / estimate
    ....:         yield b

Sage example in ./nonlinear.tex, line 1136::

    sage: iterate(secantgen(f, a), check=checkconv)
    After 8 iterations: 2.15846852557553

Sage example in ./nonlinear.tex, line 1148::

    sage: g = plot(f, a, b, rgbcolor='blue')
    sage: sequence = secantgen(f, a)
    sage: u, v = next(sequence), next(sequence)
    sage: niter = 2
    sage: while niter < 6:
    ....:     g += point((u, 0), rgbcolor='red')
    ....:     g += line([(u, 0), (u, f(u))], rgbcolor='red',
    ....:               linestyle=':')
    ....:     g += line([(u, f(u)), (v, 0)], rgbcolor='red')
    ....:     u, v = v, next(sequence)
    ....:     niter += 1

Sage example in ./nonlinear.tex, line 1198::

    sage: from collections import deque
    sage: basering = PolynomialRing(CC, 'x')
    sage: def quadraticgen(f, r, s):
    ....:     t = (r + s) / 2
    ....:     yield t
    ....:     points = deque([(r,f(r)), (s,f(s)), (t,f(t))], maxlen=3)
    ....:     while True:
    ....:         pol = basering.lagrange_polynomial(points)
    ....:         roots = pol.roots(ring=CC, multiplicities=False)
    ....:         u = min(roots, key=lambda x: abs(x - points[2][0]))
    ....:         points.append((u, f(u)))
    ....:         yield points[2][0]

Sage example in ./nonlinear.tex, line 1230::

    sage: generator = quadraticgen(f, a, b)
    sage: iterate(generator, check=checkconv)
    After 5 iterations: 2.15846852554764

Sage example in ./nonlinear.tex, line 1287::

    sage: rings = [ZZ, QQ, QQbar, RDF, RIF, RR, AA, CDF, CIF, CC]
    sage: for ring in rings:
    ....:   print("{0:50} {1}".format(ring, ring.is_exact()))
    Integer Ring                                       True
    Rational Field                                     True
    Algebraic Field                                    True
    Real Double Field                                  False
    Real Interval Field with 53 bits of precision      False
    Real Field with 53 bits of precision               False
    Algebraic Real Field                               True
    Complex Double Field                               False
    Complex Interval Field with 53 bits of precision   False
    Complex Field with 53 bits of precision            False

Sage example in ./nonlinear.tex, line 1403::

    sage: def steffensen(sequence):
    ....:  assert isinstance(sequence, GeneratorType)
    ....:  values = deque(maxlen=3)
    ....:  for i in range(3):
    ....:      values.append(next(sequence))
    ....:      yield values[i]
    ....:  while 1:
    ....:      values.append(next(sequence))
    ....:      u, v, w = values
    ....:      yield u - (v - u)^2 / (w - 2 * v + u)

Sage example in ./nonlinear.tex, line 1419::

    sage: g(x) = sin(x^2 - 2) * (x^2 - 2)
    sage: sequence = newtongen(g, RR(0.7))
    sage: accelseq = steffensen(newtongen(g, RR(0.7)))
    sage: iterate(sequence, check=checkconv)
    After 17 iterations: 1.41422192763287
    sage: iterate(accelseq, check=checkconv)
    After 10 iterations: 1.41421041980166

Sage example in ./nonlinear.tex, line 1432::

    sage: sequence = newtongen(f, RR(a))
    sage: accelseq = steffensen(newtongen(f, RR(a)))
    sage: iterate(sequence, check=checkconv)
    After 6 iterations: 2.15846852566756
    sage: iterate(accelseq, check=checkconv)
    After 7 iterations: 2.15846852554764

Sage example in ./nonlinear.tex, line 1457::

    sage: result = (f == 0).find_root(a, b, full_output=True)
    sage: result[0], result[1].iterations
    (2.1584685255476415, 9)

Sage example in ./nonlinear.tex, line 1494::

    sage: a, b = pi/2, pi
    sage: generator = newtongen(f, a)
    sage: next(generator)
    1/2*pi
    sage: next(generator)
    1/2*pi - (e^(1/2*pi) - 10)*e^(-1/2*pi)

"""
