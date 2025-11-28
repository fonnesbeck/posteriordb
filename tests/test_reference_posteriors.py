"""Tests for reference posteriors."""

import pytest

from .posteriordb import (
    REQUIRED_REFERENCE_POSTERIOR_FIELDS,
    PosteriorDB,
    validate_date_format,
)


class TestReferencePosteriorValidity:
    """Test that reference posterior info files are valid."""

    def test_all_reference_posteriors_have_required_fields(self, pdb: PosteriorDB):
        """Each reference posterior info must have all required fields."""
        errors = []
        for name in pdb.reference_posterior_names():
            ref_info = pdb.get_reference_posterior_info(name)
            missing = REQUIRED_REFERENCE_POSTERIOR_FIELDS - set(ref_info.keys())
            if missing:
                errors.append(f"{name}: missing fields {missing}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_reference_posterior_name_matches_filename(self, pdb: PosteriorDB):
        """The 'name' field must match the filename."""
        errors = []
        for name in pdb.reference_posterior_names():
            ref_info = pdb.get_reference_posterior_info(name)
            if ref_info.get("name") != name:
                errors.append(
                    f"{name}: name field '{ref_info.get('name')}' != filename '{name}'"
                )

        if errors:
            pytest.fail("\n".join(errors))

    def test_reference_posterior_date_format(self, pdb: PosteriorDB):
        """The 'added_date' field must be in YYYY-MM-DD format."""
        errors = []
        for name in pdb.reference_posterior_names():
            ref_info = pdb.get_reference_posterior_info(name)
            date = ref_info.get("added_date", "")
            if not validate_date_format(date):
                errors.append(f"{name}: invalid date format '{date}'")

        if errors:
            pytest.fail("\n".join(errors))


class TestReferencePosteriorCoverage:
    """Test that reference posteriors have matching posteriors."""

    def test_reference_posteriors_have_posterior(
        self, pdb: PosteriorDB, posterior_names: list[str]
    ):
        """Each reference posterior must have a corresponding posterior."""
        posterior_set = set(posterior_names)
        errors = []

        for name in pdb.reference_posterior_names():
            if name not in posterior_set:
                errors.append(f"Reference posterior '{name}' has no matching posterior")

        if errors:
            pytest.fail("\n".join(errors))


class TestReferencePosteriorDraws:
    """Test that reference posterior draw files exist."""

    def test_reference_posterior_draws_exist(self, pdb: PosteriorDB):
        """Reference posterior draws files should exist for each info file."""
        draws_path = pdb.path / "reference_posteriors" / "draws" / "draws"
        errors = []

        for name in pdb.reference_posterior_names():
            # Draws are typically stored as .json.zip
            draw_file = draws_path / f"{name}.json.zip"
            if not draw_file.exists():
                # Also check for uncompressed version
                draw_file_json = draws_path / f"{name}.json"
                if not draw_file_json.exists():
                    errors.append(f"{name}: draws file not found")

        if errors:
            pytest.fail("\n".join(errors))
