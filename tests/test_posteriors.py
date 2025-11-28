"""Tests for posterior definitions."""

import pytest

from .posteriordb import (
    REQUIRED_POSTERIOR_FIELDS,
    PosteriorDB,
    validate_date_format,
)


class TestPosteriorValidity:
    """Test that all posterior JSON files are valid."""

    def test_all_posteriors_have_required_fields(self, pdb: PosteriorDB):
        """Each posterior must have all required fields."""
        errors = []
        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            missing = REQUIRED_POSTERIOR_FIELDS - set(posterior.keys())
            if missing:
                errors.append(f"{name}: missing fields {missing}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_posterior_name_matches_filename(self, pdb: PosteriorDB):
        """The 'name' field must match the filename."""
        errors = []
        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            if posterior.get("name") != name:
                errors.append(
                    f"{name}: name field '{posterior.get('name')}' != filename '{name}'"
                )

        if errors:
            pytest.fail("\n".join(errors))

    def test_posterior_date_format(self, pdb: PosteriorDB):
        """The 'added_date' field must be in YYYY-MM-DD format."""
        errors = []
        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            date = posterior.get("added_date", "")
            if not validate_date_format(date):
                errors.append(f"{name}: invalid date format '{date}'")

        if errors:
            pytest.fail("\n".join(errors))

    def test_posterior_model_exists(self, pdb: PosteriorDB):
        """Each posterior must reference an existing model."""
        model_names = set(pdb.model_names())
        errors = []

        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            model_name = posterior.get("model_name")
            if model_name and model_name not in model_names:
                errors.append(f"{name}: model '{model_name}' does not exist")

        if errors:
            pytest.fail("\n".join(errors))

    def test_posterior_data_exists(self, pdb: PosteriorDB):
        """Each posterior must reference an existing dataset."""
        data_names = set(pdb.data_names())
        errors = []

        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            data_name = posterior.get("data_name")
            if data_name and data_name not in data_names:
                errors.append(f"{name}: data '{data_name}' does not exist")

        if errors:
            pytest.fail("\n".join(errors))

    def test_posterior_reference_posterior_exists(self, pdb: PosteriorDB):
        """If a posterior references a reference_posterior, it must exist."""
        ref_names = set(pdb.reference_posterior_names())
        errors = []

        for name in pdb.posterior_names():
            posterior = pdb.get_posterior(name)
            ref_name = posterior.get("reference_posterior_name")
            if ref_name and ref_name not in ref_names:
                errors.append(
                    f"{name}: reference_posterior '{ref_name}' does not exist"
                )

        if errors:
            pytest.fail("\n".join(errors))
