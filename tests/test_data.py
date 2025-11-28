"""Tests for data definitions and files."""

import pytest

from .posteriordb import (
    REQUIRED_DATA_INFO_FIELDS,
    PosteriorDB,
    get_data_used_by_posteriors,
    validate_date_format,
)


class TestDataInfoValidity:
    """Test that all data info JSON files are valid."""

    def test_all_data_have_required_fields(self, pdb: PosteriorDB):
        """Each data info must have all required fields."""
        errors = []
        for name in pdb.data_names():
            data_info = pdb.get_data_info(name)
            missing = REQUIRED_DATA_INFO_FIELDS - set(data_info.keys())
            if missing:
                errors.append(f"{name}: missing fields {missing}")

        if errors:
            pytest.fail("\n".join(errors))

    def test_data_name_matches_filename(self, pdb: PosteriorDB):
        """The 'name' field must match the filename."""
        errors = []
        for name in pdb.data_names():
            data_info = pdb.get_data_info(name)
            if data_info.get("name") != name:
                errors.append(
                    f"{name}: name field '{data_info.get('name')}' != filename '{name}'"
                )

        if errors:
            pytest.fail("\n".join(errors))

    def test_data_date_format(self, pdb: PosteriorDB):
        """The 'added_date' field must be in YYYY-MM-DD format."""
        errors = []
        for name in pdb.data_names():
            data_info = pdb.get_data_info(name)
            date = data_info.get("added_date", "")
            if not validate_date_format(date):
                errors.append(f"{name}: invalid date format '{date}'")

        if errors:
            pytest.fail("\n".join(errors))


class TestDataFiles:
    """Test that data files exist."""

    def test_data_files_exist(self, pdb: PosteriorDB):
        """Data files referenced in info must exist."""
        errors = []
        for name in pdb.data_names():
            data_path = pdb.data_file_path(name)
            if data_path is None:
                errors.append(f"{name}: no data_file specified")
            elif not data_path.exists():
                # Also check for .zip variant
                zip_path = data_path.parent / f"{data_path.name}.zip"
                if not zip_path.exists():
                    errors.append(f"{name}: data file not found: {data_path}")

        if errors:
            pytest.fail("\n".join(errors))


class TestDataCoverage:
    """Test that all datasets are used by at least one posterior."""

    def test_all_data_have_posterior(self, pdb: PosteriorDB):
        """Every dataset must be referenced by at least one posterior."""
        all_data = set(pdb.data_names())
        used_data = get_data_used_by_posteriors(pdb)
        unused_data = all_data - used_data

        if unused_data:
            pytest.fail(
                f"Datasets not used by any posterior: {sorted(unused_data)}"
            )
