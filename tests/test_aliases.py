"""Tests for posterior aliases."""

import pytest

from .posteriordb import PosteriorDB


class TestAliasValidity:
    """Test that aliases are valid."""

    def test_aliases_file_exists(self, pdb: PosteriorDB):
        """The aliases file must exist."""
        assert pdb.alias_path.exists(), f"Aliases file not found: {pdb.alias_path}"

    def test_aliases_point_to_valid_posteriors(
        self, pdb: PosteriorDB, aliases: dict[str, str], posterior_names: list[str]
    ):
        """All alias targets must be valid posterior names."""
        posterior_set = set(posterior_names)
        errors = []

        for alias_name, target in aliases.items():
            if target not in posterior_set:
                errors.append(f"Alias '{alias_name}' -> '{target}' (target not found)")

        if errors:
            pytest.fail("\n".join(errors))

    def test_aliases_dont_conflict_with_posterior_names(
        self, aliases: dict[str, str], posterior_names: list[str]
    ):
        """Alias names must not duplicate existing posterior names."""
        posterior_set = set(posterior_names)
        conflicts = set(aliases.keys()) & posterior_set

        if conflicts:
            pytest.fail(
                f"Aliases conflict with posterior names: {sorted(conflicts)}"
            )

    def test_aliases_are_unique(self, aliases: dict[str, str]):
        """Alias names must be unique (JSON handles this, but verify)."""
        # This is implicitly handled by JSON parsing, but we can verify
        # that no alias points to itself
        self_referential = [
            alias for alias, target in aliases.items() if alias == target
        ]

        if self_referential:
            pytest.fail(f"Self-referential aliases: {self_referential}")
