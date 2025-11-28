"""Posterior class for accessing posteriors in the posterior database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from posteriordb.data import Data
from posteriordb.model import Model
from posteriordb.reference_posterior import ReferencePosterior
from posteriordb.utils import load_json

if TYPE_CHECKING:
    from posteriordb.database import PosteriorDatabase


class Posterior:
    """A posterior in the posterior database.

    Links together a model, dataset, and optionally reference posterior draws.

    Example:
        >>> pdb = PosteriorDatabase("posterior_database")
        >>> posterior = pdb.posterior("eight_schools-eight_schools_noncentered")
        >>> print(posterior.model.name)
        eight_schools_noncentered
        >>> print(posterior.data.values())
        {'J': 8, 'y': [...], 'sigma': [...]}
    """

    def __init__(self, name: str, db: PosteriorDatabase) -> None:
        """Initialize the Posterior object.

        Args:
            name: Posterior name.
            db: PosteriorDatabase instance.
        """
        self._name = name
        self._db = db
        self._definition: dict[str, Any] | None = None
        self._model: Model | None = None
        self._data: Data | None = None
        self._reference_posterior: ReferencePosterior | None = None

    @property
    def name(self) -> str:
        """Posterior name."""
        return self._name

    @property
    def definition(self) -> dict[str, Any]:
        """Raw posterior definition from JSON.

        Returns:
            Dictionary containing the posterior definition.
        """
        if self._definition is None:
            path = self._db.posteriors_path / f"{self._name}.json"
            self._definition = load_json(path)
        return self._definition

    @property
    def model_name(self) -> str:
        """Name of the model used by this posterior."""
        return self.definition["model_name"]

    @property
    def data_name(self) -> str:
        """Name of the dataset used by this posterior."""
        return self.definition["data_name"]

    @property
    def reference_posterior_name(self) -> str | None:
        """Name of the reference posterior, if available."""
        return self.definition.get("reference_posterior_name")

    @property
    def model(self) -> Model:
        """Get the Model object for this posterior.

        Returns:
            Model instance.
        """
        if self._model is None:
            self._model = Model(self.model_name, self._db)
        return self._model

    @property
    def data(self) -> Data:
        """Get the Data object for this posterior.

        Returns:
            Data instance.
        """
        if self._data is None:
            self._data = Data(self.data_name, self._db)
        return self._data

    @property
    def reference_posterior(self) -> ReferencePosterior | None:
        """Get the ReferencePosterior object, if available.

        Returns:
            ReferencePosterior instance, or None if not available.
        """
        ref_name = self.reference_posterior_name
        if ref_name is None:
            return None
        if self._reference_posterior is None:
            self._reference_posterior = ReferencePosterior(ref_name, self._db)
        return self._reference_posterior

    def reference_draws(self) -> dict[str, list[float]] | None:
        """Get the reference posterior draws, if available.

        Returns:
            Dictionary mapping parameter names to lists of draws,
            or None if no reference posterior is available.
        """
        ref = self.reference_posterior
        if ref is None:
            return None
        return ref.draws()

    def reference_draws_info(self) -> dict[str, Any] | None:
        """Get the reference posterior metadata, if available.

        Returns:
            Dictionary containing inference settings and diagnostics,
            or None if no reference posterior is available.
        """
        ref = self.reference_posterior
        if ref is None:
            return None
        return ref.information

    @property
    def dimensions(self) -> dict[str, int]:
        """Parameter dimensions for this posterior.

        Returns:
            Dictionary mapping parameter names to their dimensions.
        """
        return self.definition.get("dimensions", {})

    @property
    def keywords(self) -> list[str]:
        """Keywords associated with this posterior."""
        kw = self.definition.get("keywords", [])
        if isinstance(kw, str):
            return [kw]
        return kw

    @property
    def references(self) -> list[str]:
        """Bibliography references for this posterior."""
        refs = self.definition.get("references", [])
        if isinstance(refs, str):
            return [refs]
        return refs

    def __repr__(self) -> str:
        return f"Posterior({self._name!r})"
