from _pytest.fixtures import FixtureRequest
from sage.tensor.modules.format_utilities import FormattedExpansion
import pytest

# TODO: Remove sage.all import as soon as it's no longer necessary to load everything upfront
import sage.all
from sage.manifolds.manifold import Manifold
from sage.manifolds.differentiable.manifold import DifferentiableManifold
from sage.manifolds.catalog import Sphere
from sage.manifolds.differentiable.symplectic_form import SymplecticForm
from sage.manifolds.differentiable.examples.symplectic_vector_space import SymplecticVectorSpace
from sage.symbolic.function_factory import function


class TestGenericSymplecticForm:

    @pytest.fixture
    def omega(self):
        # Generic 6-dimensional manifold
        M = Manifold(6, 'M')
        return SymplecticForm(M, 'omega')

    def test_repr(self, omega: SymplecticForm):
        assert omega._repr_() == 'Symplectic form omega on the 6-dimensional differentiable manifold M'

    def test_new_instance_repr(self, omega: SymplecticForm):
        omega1 = omega._new_instance()
        assert omega1._repr_() == 'Symplectic form unnamed symplectic form on the 6-dimensional differentiable manifold M'

    def test_new_instance_same_type(self, omega: SymplecticForm):
        omega1 = omega._new_instance()
        assert type(omega1) == type(omega)

    def test_new_instance_same_parent(self, omega: SymplecticForm):
        omega1 = omega._new_instance()
        assert omega1.parent() == omega.parent()


class TestCoherenceOfFormulas:
    r"""
    Test correctness of the implementation, by checking that equivalent formulas give the correct result.
    We check it for the examples of `\R^2` and `S^2`, which should be enough.
    """

    @pytest.fixture(params=["R2", "S2"])
    def M(self, request: FixtureRequest):
        if request.param == "R2":
            return SymplecticVectorSpace(2, 'R2', symplectic_name='omega')
        elif request.param == "S2":
            # TODO: Return sphere here instead
            # Problem: we use cartesian_coordinates below
            # return Sphere(2)
            return SymplecticVectorSpace(4, 'R4', symplectic_name='omega')

    @pytest.fixture()
    def omega(self, M):
        if isinstance(M, SymplecticVectorSpace):
            return M.symplectic_form()
        else:
            return SymplecticForm.wrap(M.metric().volume_form(), 'omega')

    def test_flat_of_hamiltonian_vector_field(self, M: DifferentiableManifold, omega: SymplecticForm):
        q, p = M.cartesian_coordinates()[:]
        H = M.scalar_field(function('H')(q, p), name='H')
        assert omega.flat(omega.hamiltonian_vector_field(H)) == - H.differential()

    def test_poisson_bracket_as_contraction_symplectic_form(self, M: DifferentiableManifold, omega: SymplecticForm):
        q, p = M.cartesian_coordinates()[:]
        f = M.scalar_field(function('f')(q,p), name='f')
        g = M.scalar_field(function('g')(q,p), name='g')
        assert omega.poisson_bracket(f, g) == omega.contract(0, omega.hamiltonian_vector_field(f)).contract(0, omega.hamiltonian_vector_field(g))

    def test_poisson_bracket_as_contraction_poisson_tensor(self, M: DifferentiableManifold, omega: SymplecticForm):
        q, p = M.cartesian_coordinates()[:]
        f = M.scalar_field(function('f')(q,p), name='f')
        g = M.scalar_field(function('g')(q,p), name='g')
        assert omega.poisson_bracket(f, g) == omega.poisson().contract(0, f.exterior_derivative()).contract(0, g.exterior_derivative())


class TestR2VectorSpace:

    @pytest.fixture
    def M(self):
        return SymplecticVectorSpace(2, 'R2', symplectic_name='omega')

    @pytest.fixture
    def omega(self, M):
        return M.symplectic_form()

    def test_display(self, omega: SymplecticForm):
        assert str(omega.display()) == r'omega = -dq/\dp'

    def test_poisson(self, omega: SymplecticForm):
        assert str(omega.poisson().display()) == r'ee = '

    def test_hamiltonian_vector_field(self, M: SymplecticVectorSpace, omega: SymplecticForm):
        q, p = M.cartesian_coordinates()[:]
        H = M.scalar_field(function('H')(q, p), name='H')
        XH = omega.hamiltonian_vector_field(H)
        assert str(XH.display()) == r'XH = '
    
    def test_flat(self, M: SymplecticVectorSpace, omega: SymplecticForm):
        X = M.vector_field(1, 2, name='X')
        assert str(X.display()) == r'X = e_q + 2 e_p'
        assert str(omega.flat(X).display()) == r'X_flat = 2 dq - dp'

