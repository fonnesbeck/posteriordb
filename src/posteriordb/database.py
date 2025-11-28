"""PosteriorDatabase class for accessing the posterior database."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from posteriordb.utils import load_json

if TYPE_CHECKING:
    from posteriordb.data import Data
    from posteriordb.model import Model
    from posteriordb.posterior import Posterior


class PosteriorDatabase:
    """Access layer for a posterior database directory.

    The posterior database contains Bayesian statistical models, datasets,
    and reference posterior draws organized in a standard directory structure.

    Example:
        >>> pdb = PosteriorDatabase("posterior_database")
        >>> posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        >>> print(posterior.data.values())
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize the database connection.

        Args:
            path: Path to the posterior_database directory.

        Raises:
            ValueError: If the path does not exist.
        """
        self._path = Path(path)
        if not self._path.exists():
            raise ValueError(f"Database path does not exist: {self._path}")

        self._aliases: dict[str, str] | None = None

    @property
    def path(self) -> Path:
        """Path to the posterior database directory."""
        return self._path

    # -------------------------------------------------------------------------
    # Path accessors
    # -------------------------------------------------------------------------

    @property
    def posteriors_path(self) -> Path:
        """Path to the posteriors directory."""
        return self._path / "posteriors"

    @property
    def models_info_path(self) -> Path:
        """Path to the models info directory."""
        return self._path / "models" / "info"

    @property
    def models_stan_path(self) -> Path:
        """Path to the Stan models directory."""
        return self._path / "models" / "stan"

    @property
    def models_pymc_path(self) -> Path:
        """Path to the PyMC models directory."""
        return self._path / "models" / "pymc"

    @property
    def data_info_path(self) -> Path:
        """Path to the data info directory."""
        return self._path / "data" / "info"

    @property
    def data_path(self) -> Path:
        """Path to the data files directory."""
        return self._path / "data" / "data"

    @property
    def reference_posteriors_path(self) -> Path:
        """Path to the reference posteriors directory."""
        return self._path / "reference_posteriors" / "draws"

    @property
    def alias_path(self) -> Path:
        """Path to the alias file."""
        return self._path / "alias" / "posteriors.json"

    # -------------------------------------------------------------------------
    # Alias resolution
    # -------------------------------------------------------------------------

    def _load_aliases(self) -> dict[str, str]:
        """Load and cache the alias mapping."""
        if self._aliases is None:
            if self.alias_path.exists():
                self._aliases = load_json(self.alias_path)
            else:
                self._aliases = {}
        return self._aliases

    def _resolve_alias(self, name: str) -> str:
        """Resolve a posterior name through aliases.

        Args:
            name: Posterior name or alias.

        Returns:
            Canonical posterior name.
        """
        aliases = self._load_aliases()
        return aliases.get(name, name)

    # -------------------------------------------------------------------------
    # Listing methods
    # -------------------------------------------------------------------------

    def posterior_names(self) -> list[str]:
        """List all available posterior names.

        Returns:
            Sorted list of posterior names.
        """
        return sorted(p.stem for p in self.posteriors_path.glob("*.json"))

    def model_names(self) -> list[str]:
        """List all available model names.

        Returns:
            Sorted list of model names.
        """
        return sorted(
            p.stem.replace(".info", "")
            for p in self.models_info_path.glob("*.info.json")
        )

    def data_names(self) -> list[str]:
        """List all available dataset names.

        Returns:
            Sorted list of dataset names.
        """
        return sorted(
            p.stem.replace(".info", "") for p in self.data_info_path.glob("*.info.json")
        )

    def reference_posterior_names(self) -> list[str]:
        """List all available reference posterior names.

        Returns:
            Sorted list of reference posterior names.
        """
        info_path = self.reference_posteriors_path / "info"
        return sorted(
            p.stem.replace(".info", "") for p in info_path.glob("*.info.json")
        )

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    def posterior(self, name: str) -> Posterior:
        """Get a posterior by name.

        Args:
            name: Posterior name or alias.

        Returns:
            Posterior object.

        Raises:
            FileNotFoundError: If the posterior does not exist.
        """
        from posteriordb.posterior import Posterior

        resolved_name = self._resolve_alias(name)
        return Posterior(resolved_name, self)

    def model(self, name: str) -> Model:
        """Get a model by name.

        Args:
            name: Model name.

        Returns:
            Model object.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        from posteriordb.model import Model

        return Model(name, self)

    def data(self, name: str) -> Data:
        """Get a dataset by name.

        Args:
            name: Dataset name.

        Returns:
            Data object.

        Raises:
            FileNotFoundError: If the dataset does not exist.
        """
        from posteriordb.data import Data

        return Data(name, self)
