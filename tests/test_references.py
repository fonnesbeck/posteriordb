"""Tests for bibliography and references."""

import pytest

from .posteriordb import PosteriorDB, get_all_cited_references


class TestBibliography:
    """Test that the bibliography file is valid."""

    def test_bibliography_exists(self, pdb: PosteriorDB):
        """The bibliography file must exist."""
        assert pdb.bibliography_path.exists(), (
            f"Bibliography file not found: {pdb.bibliography_path}"
        )

    def test_bibliography_parseable(self, pdb: PosteriorDB):
        """The bibliography must be valid BibTeX."""
        import bibtexparser

        with open(pdb.bibliography_path, encoding="utf-8") as f:
            try:
                bib_database = bibtexparser.load(f)
                assert len(bib_database.entries) > 0, "Bibliography is empty"
            except Exception as e:
                pytest.fail(f"Failed to parse bibliography: {e}")


class TestReferenceCitations:
    """Test that all cited references exist in the bibliography."""

    def test_all_cited_references_exist(self, pdb: PosteriorDB, bibtex_keys: set[str]):
        """All references cited in posteriors, models, and data must exist."""
        cited_refs = get_all_cited_references(pdb)
        missing_refs = cited_refs - bibtex_keys

        if missing_refs:
            pytest.fail(
                f"References cited but not in bibliography: {sorted(missing_refs)}"
            )
