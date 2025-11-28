"""
Utility module for accessing and validating the posteriordb database.

This module provides a PosteriorDB class for accessing the database structure
and helper functions for validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import bibtexparser


class PosteriorDB:
    """Access layer for a posterior database directory structure."""

    def __init__(self, path: Path | str):
        """Initialize the database connection.

        Args:
            path: Path to the posterior_database directory
        """
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Database path does not exist: {self.path}")

        # Cache for expensive operations
        self._bibtex_keys: set[str] | None = None
        self._posteriors_cache: dict[str, dict] | None = None
        self._model_infos_cache: dict[str, dict] | None = None
        self._data_infos_cache: dict[str, dict] | None = None

    # -------------------------------------------------------------------------
    # Path accessors
    # -------------------------------------------------------------------------

    @property
    def posteriors_path(self) -> Path:
        return self.path / "posteriors"

    @property
    def models_info_path(self) -> Path:
        return self.path / "models" / "info"

    @property
    def models_stan_path(self) -> Path:
        return self.path / "models" / "stan"

    @property
    def models_pymc_path(self) -> Path:
        return self.path / "models" / "pymc"

    @property
    def data_info_path(self) -> Path:
        return self.path / "data" / "info"

    @property
    def data_data_path(self) -> Path:
        return self.path / "data" / "data"

    @property
    def reference_posteriors_info_path(self) -> Path:
        return self.path / "reference_posteriors" / "draws" / "info"

    @property
    def alias_path(self) -> Path:
        return self.path / "alias" / "posteriors.json"

    @property
    def bibliography_path(self) -> Path:
        return self.path / "bibliography" / "references.bib"

    # -------------------------------------------------------------------------
    # List methods
    # -------------------------------------------------------------------------

    def posterior_names(self) -> list[str]:
        """List all posterior names (derived from filenames)."""
        return sorted(
            p.stem for p in self.posteriors_path.glob("*.json")
        )

    def model_names(self) -> list[str]:
        """List all model names (derived from info filenames)."""
        return sorted(
            p.stem.replace(".info", "")
            for p in self.models_info_path.glob("*.info.json")
        )

    def data_names(self) -> list[str]:
        """List all data names (derived from info filenames)."""
        return sorted(
            p.stem.replace(".info", "")
            for p in self.data_info_path.glob("*.info.json")
        )

    def reference_posterior_names(self) -> list[str]:
        """List all reference posterior names (derived from info filenames)."""
        return sorted(
            p.stem.replace(".info", "")
            for p in self.reference_posteriors_info_path.glob("*.info.json")
        )

    def stan_model_files(self) -> list[Path]:
        """List all Stan model files."""
        return sorted(self.models_stan_path.glob("*.stan"))

    def pymc_model_files(self) -> list[Path]:
        """List all PyMC model files."""
        return sorted(self.models_pymc_path.glob("*.py"))

    # -------------------------------------------------------------------------
    # Get methods (load JSON)
    # -------------------------------------------------------------------------

    def get_posterior(self, name: str) -> dict:
        """Load a posterior definition by name."""
        path = self.posteriors_path / f"{name}.json"
        return self._load_json(path)

    def get_model_info(self, name: str) -> dict:
        """Load a model info file by name."""
        path = self.models_info_path / f"{name}.info.json"
        return self._load_json(path)

    def get_data_info(self, name: str) -> dict:
        """Load a data info file by name."""
        path = self.data_info_path / f"{name}.info.json"
        return self._load_json(path)

    def get_reference_posterior_info(self, name: str) -> dict:
        """Load a reference posterior info file by name."""
        path = self.reference_posteriors_info_path / f"{name}.info.json"
        return self._load_json(path)

    def get_aliases(self) -> dict[str, str]:
        """Load the alias mapping."""
        if self.alias_path.exists():
            return self._load_json(self.alias_path)
        return {}

    # -------------------------------------------------------------------------
    # Bulk loaders (with caching)
    # -------------------------------------------------------------------------

    def all_posteriors(self) -> dict[str, dict]:
        """Load all posteriors (cached)."""
        if self._posteriors_cache is None:
            self._posteriors_cache = {
                name: self.get_posterior(name)
                for name in self.posterior_names()
            }
        return self._posteriors_cache

    def all_model_infos(self) -> dict[str, dict]:
        """Load all model info files (cached)."""
        if self._model_infos_cache is None:
            self._model_infos_cache = {
                name: self.get_model_info(name)
                for name in self.model_names()
            }
        return self._model_infos_cache

    def all_data_infos(self) -> dict[str, dict]:
        """Load all data info files (cached)."""
        if self._data_infos_cache is None:
            self._data_infos_cache = {
                name: self.get_data_info(name)
                for name in self.data_names()
            }
        return self._data_infos_cache

    # -------------------------------------------------------------------------
    # Bibliography
    # -------------------------------------------------------------------------

    def get_bibtex_keys(self) -> set[str]:
        """Parse bibliography and return all citation keys."""
        if self._bibtex_keys is not None:
            return self._bibtex_keys

        if not self.bibliography_path.exists():
            self._bibtex_keys = set()
            return self._bibtex_keys

        with open(self.bibliography_path, encoding="utf-8") as f:
            bib_database = bibtexparser.load(f)

        self._bibtex_keys = {entry["ID"] for entry in bib_database.entries}
        return self._bibtex_keys

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _load_json(self, path: Path) -> dict:
        """Load a JSON file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def model_code_path(self, model_name: str, framework: str) -> Path | None:
        """Get the path to model code for a given framework.

        Args:
            model_name: The model name
            framework: 'stan' or 'pymc'

        Returns:
            Path to the model code file, or None if not found
        """
        info = self.get_model_info(model_name)
        implementations = info.get("model_implementations", {})

        if framework not in implementations:
            return None

        code_path = implementations[framework].get("model_code")
        if code_path:
            # Path is relative to the database root
            return self.path / code_path
        return None

    def data_file_path(self, data_name: str) -> Path | None:
        """Get the path to data file.

        Args:
            data_name: The data name

        Returns:
            Path to the data file, or None if not specified
        """
        info = self.get_data_info(data_name)
        data_file = info.get("data_file")
        if data_file:
            # Path is relative to the database root
            full_path = self.path / data_file
            # Check for .zip version if plain doesn't exist
            if not full_path.exists() and not str(data_file).endswith(".zip"):
                zip_path = self.path / f"{data_file}.zip"
                if zip_path.exists():
                    return zip_path
            return full_path
        return None


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

REQUIRED_POSTERIOR_FIELDS = {"name", "model_name", "data_name", "added_by", "added_date"}
REQUIRED_MODEL_INFO_FIELDS = {"name", "model_implementations", "added_by", "added_date"}
REQUIRED_DATA_INFO_FIELDS = {"name", "data_file", "added_by", "added_date"}
REQUIRED_REFERENCE_POSTERIOR_FIELDS = {"name", "added_by", "added_date"}

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date_format(date_str: str) -> bool:
    """Check if date is in YYYY-MM-DD format."""
    return bool(DATE_PATTERN.match(date_str))


def _collect_references(obj: dict) -> set[str]:
    """Extract references from an object, handling both list and string formats."""
    refs = obj.get("references")
    if not refs:
        return set()
    if isinstance(refs, str):
        # Single reference as string
        return {refs}
    # List of references
    return set(refs)


def get_all_cited_references(pdb: PosteriorDB) -> set[str]:
    """Collect all references cited in posteriors, models, and data."""
    refs = set()

    # From posteriors
    for posterior in pdb.all_posteriors().values():
        refs.update(_collect_references(posterior))

    # From models
    for model_info in pdb.all_model_infos().values():
        refs.update(_collect_references(model_info))

    # From data
    for data_info in pdb.all_data_infos().values():
        refs.update(_collect_references(data_info))

    return refs


def get_models_used_by_posteriors(pdb: PosteriorDB) -> set[str]:
    """Get the set of model names referenced by posteriors."""
    return {p["model_name"] for p in pdb.all_posteriors().values()}


def get_data_used_by_posteriors(pdb: PosteriorDB) -> set[str]:
    """Get the set of data names referenced by posteriors."""
    return {p["data_name"] for p in pdb.all_posteriors().values()}
