"""Tests for the posteriordb Python API."""

from pathlib import Path

import pytest

from posteriordb import Data, Model, Posterior, PosteriorDatabase, ReferencePosterior


@pytest.fixture(scope="module")
def pdb() -> PosteriorDatabase:
    """Create a PosteriorDatabase instance for testing."""
    db_path = Path(__file__).parent.parent / "posterior_database"
    return PosteriorDatabase(db_path)


class TestPosteriorDatabase:
    """Tests for PosteriorDatabase class."""

    def test_init_with_valid_path(self, pdb: PosteriorDatabase) -> None:
        assert pdb.path.exists()

    def test_init_with_invalid_path(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            PosteriorDatabase("/nonexistent/path")

    def test_posterior_names(self, pdb: PosteriorDatabase) -> None:
        names = pdb.posterior_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "eight_schools-eight_schools_noncentered" in names

    def test_model_names(self, pdb: PosteriorDatabase) -> None:
        names = pdb.model_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "eight_schools_noncentered" in names

    def test_data_names(self, pdb: PosteriorDatabase) -> None:
        names = pdb.data_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "eight_schools" in names

    def test_reference_posterior_names(self, pdb: PosteriorDatabase) -> None:
        names = pdb.reference_posterior_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_alias_resolution(self, pdb: PosteriorDatabase) -> None:
        # "eight_schools" should resolve to full name
        posterior = pdb.posterior("eight_schools")
        assert posterior.name == "eight_schools-eight_schools_noncentered"


class TestPosterior:
    """Tests for Posterior class."""

    def test_posterior_name(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        assert posterior.name == "eight_schools-eight_schools_noncentered"

    def test_posterior_model_name(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        assert posterior.model_name == "eight_schools_noncentered"

    def test_posterior_data_name(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        assert posterior.data_name == "eight_schools"

    def test_posterior_model(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        model = posterior.model
        assert isinstance(model, Model)
        assert model.name == "eight_schools_noncentered"

    def test_posterior_data(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        data = posterior.data
        assert isinstance(data, Data)
        assert data.name == "eight_schools"

    def test_posterior_reference_posterior(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        ref = posterior.reference_posterior
        assert isinstance(ref, ReferencePosterior)

    def test_posterior_dimensions(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        dims = posterior.dimensions
        assert "mu" in dims
        assert "tau" in dims
        assert "theta" in dims

    def test_posterior_repr(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        assert "eight_schools-eight_schools_noncentered" in repr(posterior)


class TestModel:
    """Tests for Model class."""

    def test_model_name(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        assert model.name == "eight_schools_noncentered"

    def test_model_information(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        info = model.information
        assert isinstance(info, dict)
        assert "name" in info
        assert "model_implementations" in info

    def test_model_code_stan(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        code = model.code("stan")
        assert isinstance(code, str)
        assert "data" in code
        assert "parameters" in code

    def test_model_code_pymc(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        code = model.code("pymc")
        assert isinstance(code, str)
        assert "pymc" in code or "pm." in code

    def test_model_code_invalid_framework(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        with pytest.raises(ValueError, match="not available"):
            model.code("nonexistent")

    def test_model_frameworks(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        frameworks = model.frameworks
        assert "stan" in frameworks

    def test_model_code_path(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        path = model.code_path("stan")
        assert path is not None
        assert path.exists()
        assert path.suffix == ".stan"

    def test_model_repr(self, pdb: PosteriorDatabase) -> None:
        model = pdb.model("eight_schools_noncentered")
        assert "eight_schools_noncentered" in repr(model)


class TestData:
    """Tests for Data class."""

    def test_data_name(self, pdb: PosteriorDatabase) -> None:
        data = pdb.data("eight_schools")
        assert data.name == "eight_schools"

    def test_data_information(self, pdb: PosteriorDatabase) -> None:
        data = pdb.data("eight_schools")
        info = data.information
        assert isinstance(info, dict)
        assert "name" in info

    def test_data_values(self, pdb: PosteriorDatabase) -> None:
        data = pdb.data("eight_schools")
        values = data.values()
        assert isinstance(values, dict)
        assert "J" in values
        assert values["J"] == 8
        assert "y" in values
        assert "sigma" in values

    def test_data_title(self, pdb: PosteriorDatabase) -> None:
        data = pdb.data("eight_schools")
        assert data.title is not None

    def test_data_repr(self, pdb: PosteriorDatabase) -> None:
        data = pdb.data("eight_schools")
        assert "eight_schools" in repr(data)


class TestReferencePosterior:
    """Tests for ReferencePosterior class."""

    def test_reference_posterior_draws(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        draws = posterior.reference_draws()
        # Draws are a list of dicts, one per chain
        assert isinstance(draws, list)
        assert len(draws) == 10  # 10 chains
        assert isinstance(draws[0], dict)
        assert "mu" in draws[0] or "theta[1]" in draws[0]

    def test_reference_posterior_info(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        info = posterior.reference_draws_info()
        assert isinstance(info, dict)
        assert "diagnostics" in info or "inference" in info

    def test_reference_posterior_diagnostics(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        ref = posterior.reference_posterior
        assert ref is not None
        diagnostics = ref.diagnostics
        assert isinstance(diagnostics, dict)

    def test_reference_posterior_ndraws(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        ref = posterior.reference_posterior
        assert ref is not None
        assert ref.ndraws == 10000

    def test_reference_posterior_repr(self, pdb: PosteriorDatabase) -> None:
        posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        ref = posterior.reference_posterior
        assert ref is not None
        assert "eight_schools" in repr(ref)
