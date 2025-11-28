"""Pytest fixtures for posteriordb tests."""

from pathlib import Path

import pytest

from .posteriordb import PosteriorDB


@pytest.fixture(scope="session")
def pdb() -> PosteriorDB:
    """Provide a PosteriorDB instance for the local database."""
    # The database is in posterior_database/ relative to the repo root
    repo_root = Path(__file__).parent.parent
    db_path = repo_root / "posterior_database"
    return PosteriorDB(db_path)


@pytest.fixture(scope="session")
def posterior_names(pdb: PosteriorDB) -> list[str]:
    """List of all posterior names."""
    return pdb.posterior_names()


@pytest.fixture(scope="session")
def model_names(pdb: PosteriorDB) -> list[str]:
    """List of all model names."""
    return pdb.model_names()


@pytest.fixture(scope="session")
def data_names(pdb: PosteriorDB) -> list[str]:
    """List of all data names."""
    return pdb.data_names()


@pytest.fixture(scope="session")
def reference_posterior_names(pdb: PosteriorDB) -> list[str]:
    """List of all reference posterior names."""
    return pdb.reference_posterior_names()


@pytest.fixture(scope="session")
def bibtex_keys(pdb: PosteriorDB) -> set[str]:
    """Set of all BibTeX citation keys."""
    return pdb.get_bibtex_keys()


@pytest.fixture(scope="session")
def aliases(pdb: PosteriorDB) -> dict[str, str]:
    """Alias mapping from alias name to posterior name."""
    return pdb.get_aliases()
